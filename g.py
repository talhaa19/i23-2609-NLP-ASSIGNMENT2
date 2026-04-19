"""
MODULE G — LOAD SAVED BILSTM CHECKPOINTS, FULL TEST EVALUATION, POS CONFUSION PLOT,
NER ENTITY METRICS (CRF VS ARGMAX), ERROR SAMPLES, AND ABLATIONS A1–A4 RETRAINED ON THE SAME SPLIT.
"""

import os
import sys
import json
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import f as F


def ConfigureStdoutUtf8():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def EntitySpansFromTagStrings(TAGS):
    N = len(TAGS)
    OUT = []
    I = 0
    while I < N:
        T = TAGS[I]
        if T == "O" or (not T.startswith("B-") and not T.startswith("I-")):
            I += 1
            continue
        if T.startswith("B-"):
            TYP = T[2:]
            J = I + 1
            while J < N and TAGS[J] == "I-" + TYP:
                J += 1
            OUT.append((I, J, TYP))
            I = J
        else:
            I += 1
    return OUT


def EntityPrfPerType(GOLD_BY_SENT, PRED_BY_SENT, TYPES):
    TP = {T: 0 for T in TYPES}
    FP = {T: 0 for T in TYPES}
    FN = {T: 0 for T in TYPES}
    for GS, PS in zip(GOLD_BY_SENT, PRED_BY_SENT):
        GSET = set(GS)
        PSET = set(PS)
        for E in PSET:
            TY = E[2]
            if TY not in TP:
                continue
            if E in GSET:
                TP[TY] += 1
            else:
                FP[TY] += 1
        for E in GSET:
            TY = E[2]
            if TY not in FN:
                continue
            if E not in PSET:
                FN[TY] += 1
    ROWS = []
    TPS = FPS = FNS = 0
    for T in TYPES:
        TPS += TP[T]
        FPS += FP[T]
        FNS += FN[T]
        PR = TP[T] / (TP[T] + FP[T] + 1e-12)
        RC = TP[T] / (TP[T] + FN[T] + 1e-12)
        F1 = 2 * PR * RC / (PR + RC + 1e-12)
        ROWS.append((T, TP[T], FP[T], FN[T], PR, RC, F1))
    OPR = TPS / (TPS + FPS + 1e-12)
    ORC = TPS / (TPS + FNS + 1e-12)
    OF1 = 2 * OPR * ORC / (OPR + ORC + 1e-12)
    return ROWS, (OPR, ORC, OF1)


def NerSentenceDecodes(MODEL, NER_LABELS, W2N, UNK, T2I, DEVICE, SENTS, USE_CRF):
    MODEL.eval()
    GOLD_BY = []
    PRED_BY = []
    STRS = []
    with torch.no_grad():
        for W, TSTR in SENTS:
            WI = torch.tensor([W2N.get(X, UNK) for X in W], dtype=torch.long).unsqueeze(0).to(DEVICE)
            MASK = torch.ones((1, WI.shape[1]), dtype=torch.bool, device=DEVICE)
            LOGITS = MODEL(WI, MASK)
            LOGITS = LOGITS * MASK.unsqueeze(-1).float()
            if USE_CRF:
                BEST = MODEL.CRF.ViterbiDecode(LOGITS, MASK)
            else:
                BEST = LOGITS.argmax(dim=-1)
            TRUE_TAGS = [TSTR[J] for J in range(len(W))]
            PRED_TAGS = [NER_LABELS[int(BEST[0, J].item())] for J in range(len(W))]
            GOLD_BY.append(EntitySpansFromTagStrings(TRUE_TAGS))
            PRED_BY.append(EntitySpansFromTagStrings(PRED_TAGS))
            STRS.append(" ".join(W))
    return GOLD_BY, PRED_BY, STRS


class BiLstmPosUni(nn.Module):
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
            bidirectional=False,
            batch_first=True,
            dropout=DROPOUT if LAYERS > 1 else 0.0,
        )
        self.HEAD = nn.Linear(HID, NUM_TAGS)

    def forward(self, WORD_IDS, MASK):
        E = self.EMB(WORD_IDS)
        LENS = MASK.long().sum(dim=1).clamp(min=1).cpu()
        PACK = nn.utils.rnn.pack_padded_sequence(E, LENS, batch_first=True, enforce_sorted=False)
        OUT, _ = self.LSTM(PACK)
        OUT, _ = nn.utils.rnn.pad_packed_sequence(OUT, batch_first=True)
        return self.HEAD(OUT)


class BiLstmNerUni(nn.Module):
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
            bidirectional=False,
            batch_first=True,
            dropout=DROPOUT if LAYERS > 1 else 0.0,
        )
        self.EMIT = nn.Linear(HID, NUM_TAGS)
        self.CRF = F.LinearChainCRF(NUM_TAGS)

    def forward(self, WORD_IDS, MASK):
        E = self.EMB(WORD_IDS)
        LENS = MASK.long().sum(dim=1).clamp(min=1).cpu()
        PACK = nn.utils.rnn.pack_padded_sequence(E, LENS, batch_first=True, enforce_sorted=False)
        OUT, _ = self.LSTM(PACK)
        OUT, _ = nn.utils.rnn.pad_packed_sequence(OUT, batch_first=True)
        return self.EMIT(OUT)


class BiLstmNerCe(nn.Module):
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

    def forward(self, WORD_IDS, MASK):
        E = self.EMB(WORD_IDS)
        LENS = MASK.long().sum(dim=1).clamp(min=1).cpu()
        PACK = nn.utils.rnn.pack_padded_sequence(E, LENS, batch_first=True, enforce_sorted=False)
        OUT, _ = self.LSTM(PACK)
        OUT, _ = nn.utils.rnn.pad_packed_sequence(OUT, batch_first=True)
        return self.EMIT(OUT)


def TrainPosUntil(MODEL, TR_DS, VA_DS, DEVICE, MAX_EP, PAT, LR, WD):
    TR_LD = DataLoader(TR_DS, batch_size=16, shuffle=True, collate_fn=F.CollatePadWordTag)
    VA_LD = DataLoader(VA_DS, batch_size=16, shuffle=False, collate_fn=F.CollatePadWordTag)
    NUM_TAGS = len(TR_DS.T2I)
    OPT = torch.optim.Adam(filter(lambda P: P.requires_grad, MODEL.parameters()), lr=LR, weight_decay=WD)
    CE = nn.CrossEntropyLoss(ignore_index=-100)
    BEST = -1.0
    BAD = 0
    BEST_ST = None
    for EP in range(MAX_EP):
        MODEL.train()
        for WORD_IDS, TAGS, MASK in TR_LD:
            WORD_IDS = WORD_IDS.to(DEVICE)
            TAGS = TAGS.to(DEVICE)
            MASK = MASK.to(DEVICE)
            OPT.zero_grad()
            LOGITS = MODEL(WORD_IDS, MASK)
            LOSS = CE(LOGITS.view(-1, NUM_TAGS), TAGS.view(-1))
            LOSS.backward()
            OPT.step()
        MODEL.eval()
        PT_ALL = []
        PP_ALL = []
        with torch.no_grad():
            for WORD_IDS, TAGS, MASK in VA_LD:
                WORD_IDS = WORD_IDS.to(DEVICE)
                TAGS = TAGS.to(DEVICE)
                MASK = MASK.to(DEVICE)
                LOGITS = MODEL(WORD_IDS, MASK)
                PT, PP = F.FlattenPredsTrue(LOGITS, TAGS, MASK)
                PT_ALL.extend(PT)
                PP_ALL.extend(PP)
        F1V = F.MacroF1FromLists(PT_ALL, PP_ALL, NUM_TAGS)
        print("ABL_POS_EP", EP + 1, "val_macro_f1", round(F1V, 5), flush=True)
        if F1V > BEST + 1e-6:
            BEST = F1V
            BEST_ST = copy.deepcopy(MODEL.state_dict())
            BAD = 0
        else:
            BAD += 1
            if BAD >= PAT:
                break
    if BEST_ST is not None:
        MODEL.load_state_dict(BEST_ST)
    return BEST


def TrainNerCrfUntil(MODEL, TR_DS, VA_DS, DEVICE, MAX_EP, PAT, LR, WD):
    TR_LD = DataLoader(TR_DS, batch_size=8, shuffle=True, collate_fn=F.CollatePadWordTag)
    VA_LD = DataLoader(VA_DS, batch_size=8, shuffle=False, collate_fn=F.CollatePadWordTag)
    NUM_TAGS = len(TR_DS.T2I)
    OPT = torch.optim.Adam(filter(lambda P: P.requires_grad, MODEL.parameters()), lr=LR, weight_decay=WD)
    BEST = -1.0
    BAD = 0
    BEST_ST = None
    for EP in range(MAX_EP):
        MODEL.train()
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
        MODEL.eval()
        F1V = F.NerDecodeF1(MODEL, VA_LD, DEVICE, NUM_TAGS)
        print("ABL_NER_CRF_EP", EP + 1, "val_macro_f1", round(F1V, 5), flush=True)
        if F1V > BEST + 1e-6:
            BEST = F1V
            BEST_ST = copy.deepcopy(MODEL.state_dict())
            BAD = 0
        else:
            BAD += 1
            if BAD >= PAT:
                break
    if BEST_ST is not None:
        MODEL.load_state_dict(BEST_ST)
    return BEST


def TrainNerCeUntil(MODEL, TR_DS, VA_DS, DEVICE, MAX_EP, PAT, LR, WD):
    TR_LD = DataLoader(TR_DS, batch_size=8, shuffle=True, collate_fn=F.CollatePadWordTag)
    VA_LD = DataLoader(VA_DS, batch_size=8, shuffle=False, collate_fn=F.CollatePadWordTag)
    NUM_TAGS = len(TR_DS.T2I)
    OPT = torch.optim.Adam(filter(lambda P: P.requires_grad, MODEL.parameters()), lr=LR, weight_decay=WD)
    CE = nn.CrossEntropyLoss(ignore_index=-100)
    BEST = -1.0
    BAD = 0
    BEST_ST = None
    for EP in range(MAX_EP):
        MODEL.train()
        for WORD_IDS, TAGS, MASK in TR_LD:
            WORD_IDS = WORD_IDS.to(DEVICE)
            TAGS = TAGS.to(DEVICE)
            MASK = MASK.to(DEVICE)
            OPT.zero_grad()
            LOGITS = MODEL(WORD_IDS, MASK)
            LOSS = CE(LOGITS.view(-1, NUM_TAGS), TAGS.view(-1))
            LOSS.backward()
            OPT.step()
        MODEL.eval()
        PT_ALL = []
        PP_ALL = []
        with torch.no_grad():
            for WORD_IDS, TAGS, MASK in VA_LD:
                WORD_IDS = WORD_IDS.to(DEVICE)
                TAGS = TAGS.to(DEVICE)
                MASK = MASK.to(DEVICE)
                LOGITS = MODEL(WORD_IDS, MASK)
                PT, PP = F.FlattenPredsTrue(LOGITS, TAGS, MASK)
                PT_ALL.extend(PT)
                PP_ALL.extend(PP)
        F1V = F.MacroF1FromLists(PT_ALL, PP_ALL, NUM_TAGS)
        print("ABL_NER_CE_EP", EP + 1, "val_token_macro_f1", round(F1V, 5), flush=True)
        if F1V > BEST + 1e-6:
            BEST = F1V
            BEST_ST = copy.deepcopy(MODEL.state_dict())
            BAD = 0
        else:
            BAD += 1
            if BAD >= PAT:
                break
    if BEST_ST is not None:
        MODEL.load_state_dict(BEST_ST)
    return BEST


def PosTestMetrics(MODEL, DS, DEVICE):
    LD = DataLoader(DS, batch_size=16, shuffle=False, collate_fn=F.CollatePadWordTag)
    NUM = len(DS.T2I)
    CM = np.zeros((NUM, NUM), dtype=np.int64)
    MODEL.eval()
    COR = 0
    TOT = 0
    with torch.no_grad():
        for WORD_IDS, TAGS, MASK in LD:
            WORD_IDS = WORD_IDS.to(DEVICE)
            TAGS = TAGS.to(DEVICE)
            MASK = MASK.to(DEVICE)
            LOGITS = MODEL(WORD_IDS, MASK)
            PRED = LOGITS.argmax(dim=-1)
            for B in range(WORD_IDS.shape[0]):
                for TT in range(WORD_IDS.shape[1]):
                    if not MASK[B, TT]:
                        continue
                    TV = int(TAGS[B, TT].item())
                    if TV == -100:
                        continue
                    PV = int(PRED[B, TT].item())
                    CM[TV, PV] += 1
                    COR += int(TV == PV)
                    TOT += 1
    PT = []
    PP = []
    for GI in range(NUM):
        for PI in range(NUM):
            for _ in range(int(CM[GI, PI])):
                PT.append(GI)
                PP.append(PI)
    MF1 = F.MacroF1FromLists(PT, PP, NUM)
    return COR / max(1, TOT), MF1, CM


def PosConfusedExamples(POS_LABELS, PAIRS, SENTS, MODEL, W2N, UNK, T2I, DEVICE):
    MODEL.eval()
    OUT = {P: [] for P in PAIRS}
    with torch.no_grad():
        for W, TSTR in SENTS:
            WI = torch.tensor([W2N.get(X, UNK) for X in W], dtype=torch.long).unsqueeze(0).to(DEVICE)
            TI = torch.tensor([T2I[Y] for Y in TSTR], dtype=torch.long).unsqueeze(0).to(DEVICE)
            MASK = torch.ones((1, WI.shape[1]), dtype=torch.bool, device=DEVICE)
            LOGITS = MODEL(WI, MASK)
            PRED = LOGITS.argmax(dim=-1)
            for J in range(WI.shape[1]):
                TV = int(TI[0, J].item())
                PV = int(PRED[0, J].item())
                if TV == PV:
                    continue
                PR = (POS_LABELS[TV], POS_LABELS[PV])
                if PR not in OUT:
                    continue
                if len(OUT[PR]) >= 2:
                    continue
                OUT[PR].append(" ".join(W))
    return OUT


def Main():
    ConfigureStdoutUtf8()
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile("models/bilstm_pos.pt") or not os.path.isfile("models/bilstm_ner.pt"):
        raise SystemExit("MISSING models/bilstm_pos.pt OR models/bilstm_ner.pt — RUN f.py FIRST.")

    WORD_TO_INDEX = json.load(open("embeddings/word2idx.json", encoding="utf-8"))
    W2V = np.load("embeddings/embeddings_w2v.npy")
    W2NEW, NEW_UNK, VOC_PLUS = F.BuildWordToNewIndex(WORD_TO_INDEX)
    EMB_TENSOR = F.StackEmbeddingRows(W2V, VOC_PLUS, W2V.shape[1])

    try:
        CK_POS = torch.load("models/bilstm_pos.pt", map_location="cpu", weights_only=False)
        CK_NER = torch.load("models/bilstm_ner.pt", map_location="cpu", weights_only=False)
    except TypeError:
        CK_POS = torch.load("models/bilstm_pos.pt", map_location="cpu")
        CK_NER = torch.load("models/bilstm_ner.pt", map_location="cpu")
    POS_T2I = CK_POS["pos_tag_to_idx"]
    POS_LABELS = CK_POS["pos_idx_to_tag"]
    NER_T2I = CK_NER["ner_tag_to_idx"]
    NER_LABELS = CK_NER["ner_idx_to_tag"]
    HID = int(CK_POS["hid"])
    LAYERS = int(CK_POS["layers"])
    DROPOUT = float(CK_POS["dropout"])

    POS_TEST = F.LoadConllSentences("data/pos_test.conll")
    NER_TEST = F.LoadConllSentences("data/ner_test.conll")
    POS_TEST_DS = F.SentenceDataset(POS_TEST, W2NEW, NEW_UNK, POS_T2I)
    NER_TEST_DS = F.SentenceDataset(NER_TEST, W2NEW, NEW_UNK, NER_T2I)

    POS_MODEL = F.BiLstmPosModel(EMB_TENSOR, False, len(POS_LABELS), HID, LAYERS, 0, DROPOUT).to(DEVICE)
    POS_MODEL.load_state_dict(CK_POS["state_dict"], strict=True)

    ACC, MF1, CM = PosTestMetrics(POS_MODEL, POS_TEST_DS, DEVICE)
    print("POS_TEST_TOKEN_ACC", round(ACC, 5), "POS_TEST_MACRO_F1", round(MF1, 5), flush=True)

    FIG, AX = plt.subplots(figsize=(10, 9))
    IM = AX.imshow(CM, interpolation="nearest", cmap=plt.cm.Blues)
    AX.figure.colorbar(IM, ax=AX, fraction=0.046, pad=0.04)
    AX.set_xticks(np.arange(len(POS_LABELS)), labels=POS_LABELS, rotation=45, ha="right")
    AX.set_yticks(np.arange(len(POS_LABELS)), labels=POS_LABELS)
    AX.set_ylabel("TRUE")
    AX.set_xlabel("PREDICTED")
    AX.set_title("POS TEST CONFUSION (12x12)")
    FIG.tight_layout()
    os.makedirs("models", exist_ok=True)
    FIG.savefig("models/pos_confusion_test.png", dpi=150)
    plt.close(FIG)
    print("SAVED models/pos_confusion_test.png", flush=True)

    FLAT = []
    for GI in range(len(POS_LABELS)):
        for PI in range(len(POS_LABELS)):
            if GI == PI:
                continue
            FLAT.append((int(CM[GI, PI]), GI, PI))
    FLAT.sort(reverse=True)
    TOP3 = [(POS_LABELS[GI], POS_LABELS[PI], CNT) for CNT, GI, PI in FLAT[:3]]
    PAIR_KEYS = [(A, B) for A, B, _ in TOP3]
    EXMAP = PosConfusedExamples(POS_LABELS, PAIR_KEYS, POS_TEST, POS_MODEL, W2NEW, NEW_UNK, POS_T2I, DEVICE)
    print("POS_TOP3_CONFUSED_TAG_PAIRS", flush=True)
    for A, B, CNT in TOP3:
        print("PAIR", A, "AS", B, "COUNT", CNT, flush=True)
        for S in EXMAP.get((A, B), [])[:2]:
            print("  EX", S[:220], flush=True)

    print("POS_FROZEN_VS_FINETUNE_VALIDATION_F1_TABLE", flush=True)
    print(
        "MODE\tVAL_MACRO_F1",
        "FROZEN\t" + str(round(float(CK_POS["val_f1_frozen_best"]), 5)),
        "FINETUNE\t" + str(round(float(CK_POS["val_f1_finetune_best"]), 5)),
        sep="\n",
        flush=True,
    )
    print("NOTE_LOADED_CHECKPOINT_IS_FINETUNE_BEST_STATE", flush=True)

    NER_MODEL = F.BiLstmNerModel(EMB_TENSOR, False, len(NER_LABELS), HID, LAYERS, 0, DROPOUT).to(DEVICE)
    NER_MODEL.load_state_dict(CK_NER["state_dict"], strict=True)

    G_CRF, P_CRF, S_CRF = NerSentenceDecodes(NER_MODEL, NER_LABELS, W2NEW, NEW_UNK, NER_T2I, DEVICE, NER_TEST, True)
    G_ARG, P_ARG, S_ARG = NerSentenceDecodes(NER_MODEL, NER_LABELS, W2NEW, NEW_UNK, NER_T2I, DEVICE, NER_TEST, False)
    TYPES = ["PER", "LOC", "ORG", "MISC"]
    ROWS_CRF, OVL_CRF = EntityPrfPerType(G_CRF, P_CRF, TYPES)
    ROWS_ARG, OVL_ARG = EntityPrfPerType(G_ARG, P_ARG, TYPES)
    print("NER_ENTITY_METRICS_CRF_DECODE", flush=True)
    for T, TP, FP, FN, PR, RC, F1 in ROWS_CRF:
        print("TYPE", T, "P", round(PR, 4), "R", round(RC, 4), "F1", round(F1, 4), "TP", TP, "FP", FP, "FN", FN, flush=True)
    print("NER_OVERALL_MICRO_CRF", "P", round(OVL_CRF[0], 4), "R", round(OVL_CRF[1], 4), "F1", round(OVL_CRF[2], 4), flush=True)
    print("NER_ENTITY_METRICS_ARGMAX_DECODE", flush=True)
    for T, TP, FP, FN, PR, RC, F1 in ROWS_ARG:
        print("TYPE", T, "P", round(PR, 4), "R", round(RC, 4), "F1", round(F1, 4), "TP", TP, "FP", FP, "FN", FN, flush=True)
    print("NER_OVERALL_MICRO_ARGMAX", "P", round(OVL_ARG[0], 4), "R", round(OVL_ARG[1], 4), "F1", round(OVL_ARG[2], 4), flush=True)
    print("NER_CRF_VS_ARGMAX_TABLE", flush=True)
    print("DECODE\tOVERALL_F1", "CRF\t" + str(round(OVL_CRF[2], 5)), "ARGMAX\t" + str(round(OVL_ARG[2], 5)), sep="\n", flush=True)

    PSET = [set(X) for X in P_CRF]
    GSET = [set(X) for X in G_CRF]
    FPS = []
    FNS = []
    for SI in range(len(NER_TEST)):
        PS = PSET[SI]
        GS = GSET[SI]
        for E in PS - GS:
            FPS.append((SI, E))
        for E in GS - PS:
            FNS.append((SI, E))
    print("NER_FALSE_POSITIVES_5", flush=True)
    for K in range(min(5, len(FPS))):
        SI, E = FPS[K]
        S = S_CRF[SI]
        SPAN_TX = " ".join(NER_TEST[SI][0][E[0] : E[1]])
        EXPL = (
            "PRED_SPAN "
            + SPAN_TX
            + " AS "
            + E[2]
            + " BUT_NO_MATCHING_GOLD_SPAN_AT_SAME_TOKENS; "
            + "LIKELY_BOUNDARY_OR_TYPE_CONFUSION_OR_OVERSEGMENTATION."
        )
        print("FP", K + 1, "SENT", S[:200], flush=True)
        print("   ", EXPL, flush=True)
    print("NER_FALSE_NEGATIVES_5", flush=True)
    for K in range(min(5, len(FNS))):
        SI, E = FNS[K]
        S = S_CRF[SI]
        SPAN_TX = " ".join(NER_TEST[SI][0][E[0] : E[1]])
        EXPL = (
            "GOLD_SPAN "
            + SPAN_TX
            + " AS "
            + E[2]
            + " MISSING_IN_PREDICTIONS; "
            + "MODEL_LIKELY_EMITTED_O_OR_WRONG_BIO_STRUCTURE_FOR_THIS_SPAN."
        )
        print("FN", K + 1, "SENT", S[:200], flush=True)
        print("   ", EXPL, flush=True)

    print("NER_FROZEN_VS_FINETUNE_VALIDATION_F1_TABLE", flush=True)
    print(
        "MODE\tVAL_MACRO_F1",
        "FROZEN\t" + str(round(float(CK_NER["val_f1_frozen_best"]), 5)),
        "FINETUNE\t" + str(round(float(CK_NER["val_f1_finetune_best"]), 5)),
        sep="\n",
        flush=True,
    )

    POS_ALL = F.LoadConllSentences("data/pos_train.conll")
    NER_ALL = F.LoadConllSentences("data/ner_train.conll")
    POS_TR, POS_VA = F.SplitTrainVal(POS_ALL, 0.15, 42)
    NER_TR, NER_VA = F.SplitTrainVal(NER_ALL, 0.15, 42)
    POS_TR_DS = F.SentenceDataset(POS_TR, W2NEW, NEW_UNK, POS_T2I)
    POS_VA_DS = F.SentenceDataset(POS_VA, W2NEW, NEW_UNK, POS_T2I)
    NER_TR_DS = F.SentenceDataset(NER_TR, W2NEW, NEW_UNK, NER_T2I)
    NER_VA_DS = F.SentenceDataset(NER_VA, W2NEW, NEW_UNK, NER_T2I)

    LR = 1e-3
    WD = 1e-4
    ABL_MAX = 28
    ABL_PAT = 4

    print("ABLATION_A1_UNI_LSTM", flush=True)
    M1 = BiLstmPosUni(EMB_TENSOR, False, len(POS_LABELS), HID, LAYERS, 0, DROPOUT).to(DEVICE)
    TrainPosUntil(M1, POS_TR_DS, POS_VA_DS, DEVICE, ABL_MAX, ABL_PAT, LR, WD)
    ACC1, MF1_1, _ = PosTestMetrics(M1, POS_TEST_DS, DEVICE)
    print("A1_POS_TEST_MACRO_F1", round(MF1_1, 5), "A1_POS_TEST_ACC", round(ACC1, 5), flush=True)
    N1 = BiLstmNerUni(EMB_TENSOR, False, len(NER_LABELS), HID, LAYERS, 0, DROPOUT).to(DEVICE)
    TrainNerCrfUntil(N1, NER_TR_DS, NER_VA_DS, DEVICE, ABL_MAX, ABL_PAT + 1, LR, WD)
    G1, P1, _ = NerSentenceDecodes(N1, NER_LABELS, W2NEW, NEW_UNK, NER_T2I, DEVICE, NER_TEST, True)
    _, OV1 = EntityPrfPerType(G1, P1, TYPES)
    print("A1_NER_TEST_ENTITY_MICRO_F1", round(OV1[2], 5), flush=True)

    print("ABLATION_A2_NO_DROPOUT", flush=True)
    M2 = F.BiLstmPosModel(EMB_TENSOR, False, len(POS_LABELS), HID, LAYERS, 0, 0.0).to(DEVICE)
    TrainPosUntil(M2, POS_TR_DS, POS_VA_DS, DEVICE, ABL_MAX, ABL_PAT, LR, WD)
    ACC2, MF1_2, _ = PosTestMetrics(M2, POS_TEST_DS, DEVICE)
    print("A2_POS_TEST_MACRO_F1", round(MF1_2, 5), "A2_POS_TEST_ACC", round(ACC2, 5), flush=True)
    N2 = F.BiLstmNerModel(EMB_TENSOR, False, len(NER_LABELS), HID, LAYERS, 0, 0.0).to(DEVICE)
    TrainNerCrfUntil(N2, NER_TR_DS, NER_VA_DS, DEVICE, ABL_MAX, ABL_PAT + 1, LR, WD)
    G2, P2, _ = NerSentenceDecodes(N2, NER_LABELS, W2NEW, NEW_UNK, NER_T2I, DEVICE, NER_TEST, True)
    _, OV2 = EntityPrfPerType(G2, P2, TYPES)
    print("A2_NER_TEST_ENTITY_MICRO_F1", round(OV2[2], 5), flush=True)

    print("ABLATION_A3_RANDOM_EMB_INIT", flush=True)
    EMB_RAND = torch.randn_like(EMB_TENSOR) * 0.02
    EMB_RAND[0, :] = 0.0
    M3 = F.BiLstmPosModel(EMB_RAND, False, len(POS_LABELS), HID, LAYERS, 0, DROPOUT).to(DEVICE)
    TrainPosUntil(M3, POS_TR_DS, POS_VA_DS, DEVICE, ABL_MAX, ABL_PAT, LR, WD)
    ACC3, MF1_3, _ = PosTestMetrics(M3, POS_TEST_DS, DEVICE)
    print("A3_POS_TEST_MACRO_F1", round(MF1_3, 5), "A3_POS_TEST_ACC", round(ACC3, 5), flush=True)
    N3 = F.BiLstmNerModel(EMB_RAND, False, len(NER_LABELS), HID, LAYERS, 0, DROPOUT).to(DEVICE)
    TrainNerCrfUntil(N3, NER_TR_DS, NER_VA_DS, DEVICE, ABL_MAX, ABL_PAT + 1, LR, WD)
    G3, P3, _ = NerSentenceDecodes(N3, NER_LABELS, W2NEW, NEW_UNK, NER_T2I, DEVICE, NER_TEST, True)
    _, OV3 = EntityPrfPerType(G3, P3, TYPES)
    print("A3_NER_TEST_ENTITY_MICRO_F1", round(OV3[2], 5), flush=True)

    print("ABLATION_A4_SOFTMAX_NO_CRF", flush=True)
    N4 = BiLstmNerCe(EMB_TENSOR, False, len(NER_LABELS), HID, LAYERS, 0, DROPOUT).to(DEVICE)
    TrainNerCeUntil(N4, NER_TR_DS, NER_VA_DS, DEVICE, ABL_MAX, ABL_PAT + 1, LR, WD)
    G4, P4, _ = NerSentenceDecodes(N4, NER_LABELS, W2NEW, NEW_UNK, NER_T2I, DEVICE, NER_TEST, False)
    _, OV4 = EntityPrfPerType(G4, P4, TYPES)
    print("A4_NER_TEST_ENTITY_MICRO_F1", round(OV4[2], 5), flush=True)

    print("ABLATION_DISCUSSION", flush=True)
    print(
        "A1_REMOVES_BACKWARD_CONTEXT_SO_EXPECT_LOWER_F1_ON_AMBIGUOUS_URDU_MORPHOLOGY.",
        "A2_RAISES_OVERFIT_RISK_WITHOUT_DROPOUT_REGULARIZATION.",
        "A3_SHOWS_WORD2VEC_PRIOR_IS_STRONG_WHEN_RANDOM_INIT_STARTS_FROM_SCRATCH.",
        "A4_DROPS_STRUCTURED_DECODING_SO_INCONSISTENT_BIO_TAGS_HURT_ENTITY_F1.",
        sep="\n",
        flush=True,
    )

    SUM = {
        "pos_test_token_acc": ACC,
        "pos_test_macro_f1": MF1,
        "ner_overall_micro_p_crf": OVL_CRF[0],
        "ner_overall_micro_r_crf": OVL_CRF[1],
        "ner_overall_micro_f1_crf": OVL_CRF[2],
        "ner_overall_micro_f1_argmax": OVL_ARG[2],
        "ablation_a1_pos_test_macro_f1": float(MF1_1),
        "ablation_a1_ner_entity_micro_f1": float(OV1[2]),
        "ablation_a2_pos_test_macro_f1": float(MF1_2),
        "ablation_a2_ner_entity_micro_f1": float(OV2[2]),
        "ablation_a3_pos_test_macro_f1": float(MF1_3),
        "ablation_a3_ner_entity_micro_f1": float(OV3[2]),
        "ablation_a4_ner_entity_micro_f1": float(OV4[2]),
    }
    with open("models/metrics_summary.json", "w", encoding="utf-8") as HANDLE:
        json.dump(SUM, HANDLE, indent=2)
    print("SAVED models/metrics_summary.json", flush=True)


if __name__ == "__main__":
    Main()
