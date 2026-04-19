"""
MODULE F — TWO-LAYER BIDIRECTIONAL LSTM FOR POS (CE) AND NER (LINEAR-CHAIN CRF + VITERBI).
WORD2VEC INITIALIZATION WITH FROZEN AND FINETUNED RUNS, VAL MACRO-F1 EARLY STOPPING, LOSS PLOTS.
"""

import os
import sys
import json
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader


def ConfigureStdoutUtf8():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def LoadConllSentences(PATH):
    SENTS = []
    CUR_W = []
    CUR_T = []
    with open(PATH, encoding="utf-8") as HANDLE:
        for RAW in HANDLE:
            RAW = RAW.rstrip("\n")
            if not RAW.strip():
                if CUR_W:
                    SENTS.append((CUR_W, CUR_T))
                    CUR_W = []
                    CUR_T = []
                continue
            PARTS = RAW.split("\t")
            if len(PARTS) >= 2:
                CUR_W.append(PARTS[0])
                CUR_T.append(PARTS[1])
        if CUR_W:
            SENTS.append((CUR_W, CUR_T))
    return SENTS


def BuildWordToNewIndex(WORD_TO_INDEX):
    """
    PAD ROW IS INDEX 0; ORIGINAL word2idx K MAPS TO K+1 SO EMBEDDING[0] IS PADDING ZEROS.
    """
    W2NEW = {}
    for W, OLD in WORD_TO_INDEX.items():
        W2NEW[W] = int(OLD) + 1
    NEW_UNK = int(WORD_TO_INDEX["<UNK>"]) + 1
    VOC_PLUS = len(WORD_TO_INDEX) + 1
    return W2NEW, NEW_UNK, VOC_PLUS


def StackEmbeddingRows(W2V_MATRIX, VOC_PLUS, EMB_DIM):
    OUT = np.zeros((VOC_PLUS, EMB_DIM), dtype=np.float32)
    OUT[1:] = W2V_MATRIX.astype(np.float32)
    return torch.from_numpy(OUT)


def BuildTagDicts(SENTENCES):
    TAGS = sorted({T for W, TLIST in SENTENCES for T in TLIST})
    T2I = {T: I for I, T in enumerate(TAGS)}
    return T2I, TAGS


def SplitTrainVal(SENTENCES, VAL_FRAC, SEED):
    RNG = random.Random(SEED)
    IDX = list(range(len(SENTENCES)))
    RNG.shuffle(IDX)
    N_VAL = max(1, int(round(len(IDX) * VAL_FRAC)))
    VAL_SET = set(IDX[:N_VAL])
    TR = [SENTENCES[I] for I in range(len(SENTENCES)) if I not in VAL_SET]
    VA = [SENTENCES[I] for I in range(len(SENTENCES)) if I in VAL_SET]
    return TR, VA


class SentenceDataset(Dataset):
    def __init__(self, SENTENCES, WORD_TO_NEW, NEW_UNK, TAG_TO_IDX):
        self.SENTS = SENTENCES
        self.W2N = WORD_TO_NEW
        self.UNK = int(NEW_UNK)
        self.T2I = TAG_TO_IDX

    def __len__(self):
        return len(self.SENTS)

    def __getitem__(self, I):
        W, T = self.SENTS[I]
        WI = [self.W2N.get(X, self.UNK) for X in W]
        TI = [self.T2I[Y] for Y in T]
        return torch.tensor(WI, dtype=torch.long), torch.tensor(TI, dtype=torch.long)


def CollatePadWordTag(BATCH):
    MAXL = max(int(B[0].shape[0]) for B in BATCH)
    BSIZE = len(BATCH)
    PAD_W = torch.zeros((BSIZE, MAXL), dtype=torch.long)
    PAD_T = torch.full((BSIZE, MAXL), -100, dtype=torch.long)
    MASK = torch.zeros((BSIZE, MAXL), dtype=torch.bool)
    for I, (W, T) in enumerate(BATCH):
        L = int(W.shape[0])
        PAD_W[I, :L] = W
        PAD_T[I, :L] = T
        MASK[I, :L] = True
    return PAD_W, PAD_T, MASK


class LinearChainCRF(nn.Module):
    def __init__(self, NUM_TAGS):
        super().__init__()
        self.K = NUM_TAGS
        self.START = nn.Parameter(torch.randn(NUM_TAGS) * 0.01)
        self.END = nn.Parameter(torch.randn(NUM_TAGS) * 0.01)
        self.TRANS = nn.Parameter(torch.randn(NUM_TAGS, NUM_TAGS) * 0.01)

    def ForwardLogZ(self, EMISSIONS, MASK):
        """
        EMISSIONS (B, T, K); MASK (B, T) BOOL. RETURNS LOGZ (B,).
        """
        B, T, K = EMISSIONS.shape
        BA = torch.arange(B, device=EMISSIONS.device)
        ALPHA = self.START.unsqueeze(0) + EMISSIONS[:, 0, :]
        for TT in range(1, T):
            EMIT = EMISSIONS[:, TT, :]
            M = MASK[:, TT].unsqueeze(1).float()
            PACK = ALPHA.unsqueeze(2) + self.TRANS.unsqueeze(0)
            NEW = torch.logsumexp(PACK, dim=1) + EMIT
            ALPHA = NEW * M + ALPHA * (1.0 - M)
        END_SCORES = ALPHA + self.END.unsqueeze(0)
        return torch.logsumexp(END_SCORES, dim=1)

    def GoldPathScore(self, EMISSIONS, TAGS, MASK):
        B, T, K = EMISSIONS.shape
        BA = torch.arange(B, device=EMISSIONS.device)
        SC = (self.START[TAGS[:, 0]] + EMISSIONS[BA, 0, TAGS[:, 0]]) * MASK[:, 0].float()
        for TT in range(1, T):
            M = (MASK[:, TT] & MASK[:, TT - 1]).float()
            PR = TAGS[:, TT - 1].clamp(min=0)
            CR = TAGS[:, TT].clamp(min=0)
            SC = SC + (self.TRANS[PR, CR] + EMISSIONS[BA, TT, CR]) * M
        LENM1 = MASK.long().sum(dim=1) - 1
        LAST = TAGS[BA, LENM1]
        SC = SC + self.END[LAST]
        return SC

    def NegLogLikelihood(self, EMISSIONS, TAGS, MASK):
        LOGZ = self.ForwardLogZ(EMISSIONS, MASK)
        GOLD = self.GoldPathScore(EMISSIONS, TAGS, MASK)
        return (LOGZ - GOLD).mean()

    def ViterbiDecode(self, EMISSIONS, MASK):
        B, T, K = EMISSIONS.shape
        BA = torch.arange(B, device=EMISSIONS.device)
        SCORE = self.START.unsqueeze(0) + EMISSIONS[:, 0, :]
        BACK = torch.zeros((B, T, K), dtype=torch.long, device=EMISSIONS.device)
        for TT in range(1, T):
            PACK = SCORE.unsqueeze(2) + self.TRANS.unsqueeze(0)
            BEST, IND = PACK.max(dim=1)
            NEW = BEST + EMISSIONS[:, TT, :]
            M = MASK[:, TT].unsqueeze(1).float()
            SCORE = NEW * M + SCORE * (1.0 - M)
            BACK[:, TT, :] = IND
        END_SCORES = SCORE + self.END.unsqueeze(0)
        BEST_LAST = END_SCORES.argmax(dim=1)
        OUT = torch.zeros((B, T), dtype=torch.long, device=EMISSIONS.device)
        OUT[BA, T - 1] = BEST_LAST
        for TT in range(T - 2, -1, -1):
            PREV_IDX = OUT[:, TT + 1].unsqueeze(1)
            OUT[:, TT] = torch.gather(BACK[:, TT + 1, :], 1, PREV_IDX).squeeze(1)
        return OUT


class BiLstmPosModel(nn.Module):
    def __init__(self, EMB_WEIGHT, FREEZE_EMB, NUM_TAGS, HID, LAYERS, PAD_IDX, DROPOUT):
        super().__init__()
        VOC, DIM = EMB_WEIGHT.shape
        self.EMB = nn.Embedding(VOC, DIM, padding_idx=PAD_IDX)
        self.EMB.weight.data.copy_(EMB_WEIGHT)
        self.EMB.weight.requires_grad = not FREEZE_EMB
        self.LSTM = nn.LSTM(
            DIM,
            HID,
            num_layers=LAYERS,
            bidirectional=True,
            batch_first=True,
            dropout=DROPOUT if LAYERS > 1 else 0.0,
        )
        self.HEAD = nn.Linear(2 * HID, NUM_TAGS)

    def forward(self, WORD_IDS, MASK):
        E = self.EMB(WORD_IDS)
        LENS = MASK.long().sum(dim=1).clamp(min=1).cpu()
        PACK = nn.utils.rnn.pack_padded_sequence(E, LENS, batch_first=True, enforce_sorted=False)
        OUT, _ = self.LSTM(PACK)
        OUT, _ = nn.utils.rnn.pad_packed_sequence(OUT, batch_first=True)
        return self.HEAD(OUT)


class BiLstmNerModel(nn.Module):
    def __init__(self, EMB_WEIGHT, FREEZE_EMB, NUM_TAGS, HID, LAYERS, PAD_IDX, DROPOUT):
        super().__init__()
        VOC, DIM = EMB_WEIGHT.shape
        self.EMB = nn.Embedding(VOC, DIM, padding_idx=PAD_IDX)
        self.EMB.weight.data.copy_(EMB_WEIGHT)
        self.EMB.weight.requires_grad = not FREEZE_EMB
        self.LSTM = nn.LSTM(
            DIM,
            HID,
            num_layers=LAYERS,
            bidirectional=True,
            batch_first=True,
            dropout=DROPOUT if LAYERS > 1 else 0.0,
        )
        self.EMIT = nn.Linear(2 * HID, NUM_TAGS)
        self.CRF = LinearChainCRF(NUM_TAGS)

    def forward(self, WORD_IDS, MASK):
        E = self.EMB(WORD_IDS)
        LENS = MASK.long().sum(dim=1).clamp(min=1).cpu()
        PACK = nn.utils.rnn.pack_padded_sequence(E, LENS, batch_first=True, enforce_sorted=False)
        OUT, _ = self.LSTM(PACK)
        OUT, _ = nn.utils.rnn.pad_packed_sequence(OUT, batch_first=True)
        return self.EMIT(OUT)


def FlattenPredsTrue(LOGITS, TAGS, MASK):
    PRED = LOGITS.argmax(dim=-1)
    PT = []
    PP = []
    for B in range(TAGS.shape[0]):
        for TT in range(TAGS.shape[1]):
            if not MASK[B, TT]:
                continue
            TV = int(TAGS[B, TT].item())
            if TV == -100:
                continue
            PT.append(TV)
            PP.append(int(PRED[B, TT].item()))
    return PT, PP


def MacroF1FromLists(TRUE_IDX, PRED_IDX, NUM_LABELS):
    if not TRUE_IDX:
        return 0.0
    CM = np.zeros((NUM_LABELS, NUM_LABELS), dtype=np.float64)
    for T, P in zip(TRUE_IDX, PRED_IDX):
        if 0 <= T < NUM_LABELS and 0 <= P < NUM_LABELS:
            CM[T, P] += 1.0
    F1S = []
    for K in range(NUM_LABELS):
        TP = CM[K, K]
        FP = CM[:, K].sum() - TP
        FN = CM[K, :].sum() - TP
        if TP + FP + FN <= 0:
            continue
        PR = TP / (TP + FP + 1e-12)
        RC = TP / (TP + FN + 1e-12)
        F1S.append(2 * PR * RC / (PR + RC + 1e-12))
    return float(np.mean(F1S)) if F1S else 0.0


def NerDecodeF1(MODEL, LOADER, DEVICE, NUM_TAGS):
    MODEL.eval()
    TRUE_ALL = []
    PRED_ALL = []
    with torch.no_grad():
        for WORD_IDS, TAGS, MASK in LOADER:
            WORD_IDS = WORD_IDS.to(DEVICE)
            TAGS = TAGS.to(DEVICE)
            MASK = MASK.to(DEVICE)
            LOGITS = MODEL(WORD_IDS, MASK)
            LOGITS = LOGITS * MASK.unsqueeze(-1).float()
            BEST = MODEL.CRF.ViterbiDecode(LOGITS, MASK)
            for B in range(WORD_IDS.shape[0]):
                for TT in range(WORD_IDS.shape[1]):
                    if not MASK[B, TT]:
                        continue
                    TRUE_ALL.append(int(TAGS[B, TT].item()))
                    PRED_ALL.append(int(BEST[B, TT].item()))
    return MacroF1FromLists(TRUE_ALL, PRED_ALL, NUM_TAGS)


def TrainPosOneMode(FREEZE, TRAIN_DS, VAL_DS, EMB_MATRIX, DEVICE, HID, LAYERS, DROPOUT, LR, WD, PATIENCE, MAX_EPOCHS, PLOT_PATH):
    TR_LD = DataLoader(TRAIN_DS, batch_size=16, shuffle=True, collate_fn=CollatePadWordTag)
    VA_LD = DataLoader(VAL_DS, batch_size=16, shuffle=False, collate_fn=CollatePadWordTag)
    NUM_TAGS = len(TRAIN_DS.T2I)
    MODEL = BiLstmPosModel(EMB_MATRIX, FREEZE, NUM_TAGS, HID, LAYERS, 0, DROPOUT).to(DEVICE)
    OPT = torch.optim.Adam(filter(lambda P: P.requires_grad, MODEL.parameters()), lr=LR, weight_decay=WD)
    CE = nn.CrossEntropyLoss(ignore_index=-100)
    TR_LOSS_HIST = []
    VA_LOSS_HIST = []
    VA_F1_HIST = []
    BEST_F1 = -1.0
    BEST_STATE = None
    BAD = 0
    for EP in range(MAX_EPOCHS):
        MODEL.train()
        RUN = 0.0
        STEPS = 0
        for WORD_IDS, TAGS, MASK in TR_LD:
            WORD_IDS = WORD_IDS.to(DEVICE)
            TAGS = TAGS.to(DEVICE)
            MASK = MASK.to(DEVICE)
            OPT.zero_grad()
            LOGITS = MODEL(WORD_IDS, MASK)
            LOSS = CE(LOGITS.view(-1, NUM_TAGS), TAGS.view(-1))
            LOSS.backward()
            OPT.step()
            RUN += float(LOSS.item())
            STEPS += 1
        TR_LOSS_HIST.append(RUN / max(1, STEPS))
        MODEL.eval()
        VA_LOSS = 0.0
        VST = 0
        PT_ALL = []
        PP_ALL = []
        with torch.no_grad():
            for WORD_IDS, TAGS, MASK in VA_LD:
                WORD_IDS = WORD_IDS.to(DEVICE)
                TAGS = TAGS.to(DEVICE)
                MASK = MASK.to(DEVICE)
                LOGITS = MODEL(WORD_IDS, MASK)
                VA_LOSS += float(CE(LOGITS.view(-1, NUM_TAGS), TAGS.view(-1)).item())
                VST += 1
                PT, PP = FlattenPredsTrue(LOGITS, TAGS, MASK)
                PT_ALL.extend(PT)
                PP_ALL.extend(PP)
        VA_LOSS_HIST.append(VA_LOSS / max(1, VST))
        F1V = MacroF1FromLists(PT_ALL, PP_ALL, NUM_TAGS)
        VA_F1_HIST.append(F1V)
        MODE_STR = "FROZEN_EMB" if FREEZE else "FINETUNE_EMB"
        print(
            "POS",
            MODE_STR,
            "epoch",
            EP + 1,
            "train_loss",
            round(TR_LOSS_HIST[-1], 5),
            "val_loss",
            round(VA_LOSS_HIST[-1], 5),
            "val_macro_f1",
            round(F1V, 5),
            flush=True,
        )
        if F1V > BEST_F1 + 1e-6:
            BEST_F1 = F1V
            BEST_STATE = copy.deepcopy(MODEL.state_dict())
            BAD = 0
        else:
            BAD += 1
            if BAD >= PATIENCE:
                print("POS", MODE_STR, "early_stop epoch", EP + 1, flush=True)
                break
    FIG, AX = plt.subplots(figsize=(8, 5))
    AX.plot(range(1, len(TR_LOSS_HIST) + 1), TR_LOSS_HIST, label="train_loss")
    AX.plot(range(1, len(VA_LOSS_HIST) + 1), VA_LOSS_HIST, label="val_loss")
    AX.set_xlabel("epoch")
    AX.set_ylabel("loss")
    AX.set_title("POS BiLSTM train vs val loss (" + ("frozen" if FREEZE else "finetuned") + " embeddings)")
    AX.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close(FIG)
    return BEST_STATE, BEST_F1


def TrainNerOneMode(FREEZE, TRAIN_DS, VAL_DS, EMB_MATRIX, DEVICE, HID, LAYERS, DROPOUT, LR, WD, PATIENCE, MAX_EPOCHS, PLOT_PATH):
    TR_LD = DataLoader(TRAIN_DS, batch_size=8, shuffle=True, collate_fn=CollatePadWordTag)
    VA_LD = DataLoader(VAL_DS, batch_size=8, shuffle=False, collate_fn=CollatePadWordTag)
    NUM_TAGS = len(TRAIN_DS.T2I)
    MODEL = BiLstmNerModel(EMB_MATRIX, FREEZE, NUM_TAGS, HID, LAYERS, 0, DROPOUT).to(DEVICE)
    OPT = torch.optim.Adam(filter(lambda P: P.requires_grad, MODEL.parameters()), lr=LR, weight_decay=WD)
    TR_LOSS_HIST = []
    VA_LOSS_HIST = []
    VA_F1_HIST = []
    BEST_F1 = -1.0
    BEST_STATE = None
    BAD = 0
    for EP in range(MAX_EPOCHS):
        MODEL.train()
        RUN = 0.0
        STEPS = 0
        for WORD_IDS, TAGS, MASK in TR_LD:
            WORD_IDS = WORD_IDS.to(DEVICE)
            TAGS = TAGS.to(DEVICE)
            MASK = MASK.to(DEVICE)
            OPT.zero_grad()
            LOGITS = MODEL(WORD_IDS, MASK)
            LOGITS = LOGITS * MASK.unsqueeze(-1).float()
            LOSS = MODEL.CRF.NegLogLikelihood(LOGITS, TAGS, MASK)
            LOSS.backward()
            OPT.step()
            RUN += float(LOSS.item())
            STEPS += 1
        TR_LOSS_HIST.append(RUN / max(1, STEPS))
        MODEL.eval()
        VA_LOSS = 0.0
        VST = 0
        with torch.no_grad():
            for WORD_IDS, TAGS, MASK in VA_LD:
                WORD_IDS = WORD_IDS.to(DEVICE)
                TAGS = TAGS.to(DEVICE)
                MASK = MASK.to(DEVICE)
                LOGITS = MODEL(WORD_IDS, MASK)
                LOGITS = LOGITS * MASK.unsqueeze(-1).float()
                VA_LOSS += float(MODEL.CRF.NegLogLikelihood(LOGITS, TAGS, MASK).item())
                VST += 1
        VA_LOSS_HIST.append(VA_LOSS / max(1, VST))
        F1V = NerDecodeF1(MODEL, VA_LD, DEVICE, NUM_TAGS)
        VA_F1_HIST.append(F1V)
        MODE_STR = "FROZEN_EMB" if FREEZE else "FINETUNE_EMB"
        print(
            "NER",
            MODE_STR,
            "epoch",
            EP + 1,
            "train_loss",
            round(TR_LOSS_HIST[-1], 5),
            "val_loss",
            round(VA_LOSS_HIST[-1], 5),
            "val_macro_f1",
            round(F1V, 5),
            flush=True,
        )
        if F1V > BEST_F1 + 1e-6:
            BEST_F1 = F1V
            BEST_STATE = copy.deepcopy(MODEL.state_dict())
            BAD = 0
        else:
            BAD += 1
            if BAD >= PATIENCE:
                print("NER", MODE_STR, "early_stop epoch", EP + 1, flush=True)
                break
    FIG, AX = plt.subplots(figsize=(8, 5))
    AX.plot(range(1, len(TR_LOSS_HIST) + 1), TR_LOSS_HIST, label="train_loss")
    AX.plot(range(1, len(VA_LOSS_HIST) + 1), VA_LOSS_HIST, label="val_loss")
    AX.set_xlabel("epoch")
    AX.set_ylabel("loss")
    AX.set_title("NER BiLSTM+CRF train vs val loss (" + ("frozen" if FREEZE else "finetuned") + " embeddings)")
    AX.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close(FIG)
    return BEST_STATE, BEST_F1


def Main():
    ConfigureStdoutUtf8()
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WORD_TO_INDEX = json.load(open("embeddings/word2idx.json", encoding="utf-8"))
    W2V = np.load("embeddings/embeddings_w2v.npy")
    W2NEW, NEW_UNK, VOC_PLUS = BuildWordToNewIndex(WORD_TO_INDEX)
    EMB_TENSOR = StackEmbeddingRows(W2V, VOC_PLUS, W2V.shape[1])

    POS_ALL = LoadConllSentences("data/pos_train.conll")
    NER_ALL = LoadConllSentences("data/ner_train.conll")
    if len(POS_ALL) != len(NER_ALL):
        raise SystemExit("POS AND NER TRAIN SENTENCE COUNTS MUST MATCH.")
    for I in range(len(POS_ALL)):
        if POS_ALL[I][0] != NER_ALL[I][0]:
            raise SystemExit("WORD MISMATCH AT SENTENCE " + str(I))

    POS_TR, POS_VA = SplitTrainVal(POS_ALL, 0.15, 42)
    NER_TR, NER_VA = SplitTrainVal(NER_ALL, 0.15, 42)

    POS_T2I, POS_LABELS = BuildTagDicts(POS_TR + POS_VA)
    NER_T2I, NER_LABELS = BuildTagDicts(NER_TR + NER_VA)

    POS_TR_DS = SentenceDataset(POS_TR, W2NEW, NEW_UNK, POS_T2I)
    POS_VA_DS = SentenceDataset(POS_VA, W2NEW, NEW_UNK, POS_T2I)
    NER_TR_DS = SentenceDataset(NER_TR, W2NEW, NEW_UNK, NER_T2I)
    NER_VA_DS = SentenceDataset(NER_VA, W2NEW, NEW_UNK, NER_T2I)

    HID = 100
    LAYERS = 2
    DROPOUT = 0.5
    LR = 1e-3
    WD = 1e-4
    PAT = 5
    MAX_EP = 60

    os.makedirs("models", exist_ok=True)

    print("=== POS FROZEN ===", flush=True)
    ST_POS_FR, F1_POS_FR = TrainPosOneMode(
        True, POS_TR_DS, POS_VA_DS, EMB_TENSOR, DEVICE, HID, LAYERS, DROPOUT, LR, WD, PAT, MAX_EP, "models/pos_loss_frozen.png"
    )
    print("BEST_POS_VAL_F1_FROZEN", round(F1_POS_FR, 5), flush=True)

    print("=== POS FINETUNE ===", flush=True)
    ST_POS_FT, F1_POS_FT = TrainPosOneMode(
        False, POS_TR_DS, POS_VA_DS, EMB_TENSOR, DEVICE, HID, LAYERS, DROPOUT, LR, WD, PAT, MAX_EP, "models/pos_loss_finetune.png"
    )
    print("BEST_POS_VAL_F1_FINETUNE", round(F1_POS_FT, 5), flush=True)

    torch.save(
        {
            "state_dict": ST_POS_FT,
            "pos_tag_to_idx": POS_T2I,
            "pos_idx_to_tag": POS_LABELS,
            "hid": HID,
            "layers": LAYERS,
            "dropout": DROPOUT,
            "vocab_plus_one": VOC_PLUS,
            "emb_dim": int(W2V.shape[1]),
            "val_f1_frozen_best": F1_POS_FR,
            "val_f1_finetune_best": F1_POS_FT,
        },
        "models/bilstm_pos.pt",
    )

    print("=== NER FROZEN ===", flush=True)
    ST_NER_FR, F1_NER_FR = TrainNerOneMode(
        True, NER_TR_DS, NER_VA_DS, EMB_TENSOR, DEVICE, HID, LAYERS, DROPOUT, LR, WD, PAT + 3, MAX_EP, "models/ner_loss_frozen.png"
    )
    print("BEST_NER_VAL_F1_FROZEN", round(F1_NER_FR, 5), flush=True)

    print("=== NER FINETUNE ===", flush=True)
    ST_NER_FT, F1_NER_FT = TrainNerOneMode(
        False, NER_TR_DS, NER_VA_DS, EMB_TENSOR, DEVICE, HID, LAYERS, DROPOUT, LR, WD, PAT + 3, MAX_EP, "models/ner_loss_finetune.png"
    )
    print("BEST_NER_VAL_F1_FINETUNE", round(F1_NER_FT, 5), flush=True)

    torch.save(
        {
            "state_dict": ST_NER_FT,
            "ner_tag_to_idx": NER_T2I,
            "ner_idx_to_tag": NER_LABELS,
            "hid": HID,
            "layers": LAYERS,
            "dropout": DROPOUT,
            "vocab_plus_one": VOC_PLUS,
            "emb_dim": int(W2V.shape[1]),
            "val_f1_frozen_best": F1_NER_FR,
            "val_f1_finetune_best": F1_NER_FT,
        },
        "models/bilstm_ner.pt",
    )

    print("saved models/bilstm_pos.pt and models/bilstm_ner.pt", flush=True)


if __name__ == "__main__":
    Main()
