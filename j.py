"""
MODULE J — TRAIN TransformerCls ON CLS NUMPY SPLITS, ADAMW + COSINE WARMUP, PLOTS, TEST METRICS, HEATMAPS.
SAVES models/transformer_cls.pt AND FIGURES UNDER models/.
"""

import os
import sys
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

import h as TRANSFORMER_LIB


def ConfigureStdoutUtf8():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def MacroF1Multiclass(Y_TRUE, Y_PRED, NUM_C):
    CM = np.zeros((NUM_C, NUM_C), dtype=np.float64)
    for T, P in zip(Y_TRUE, Y_PRED):
        CM[int(T), int(P)] += 1.0
    F1S = []
    for K in range(NUM_C):
        TP = CM[K, K]
        FP = CM[:, K].sum() - TP
        FN = CM[K, :].sum() - TP
        if TP + FP + FN <= 0:
            continue
        PR = TP / (TP + FP + 1e-12)
        RC = TP / (TP + FN + 1e-12)
        F1S.append(2 * PR * RC / (PR + RC + 1e-12))
    return float(np.mean(F1S)) if F1S else 0.0


def BuildWarmupCosineLambda(WARMUP_STEPS, TOTAL_STEPS):
    def FN(STEP):
        S = float(STEP)
        if S < float(WARMUP_STEPS):
            return max(S / float(max(1, WARMUP_STEPS)), 1e-8)
        P = (S - float(WARMUP_STEPS)) / float(max(1, TOTAL_STEPS - WARMUP_STEPS))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, P)))

    return FN


def Main():
    ConfigureStdoutUtf8()
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    META = json.load(open("data/cls_meta.json", encoding="utf-8"))
    VOC_PLUS = int(META["vocab_plus_one"])
    NAMES = META["class_names"]
    NUM_C = len(NAMES)

    X_TR = np.load("data/cls_train_x.npy")
    Y_TR = np.load("data/cls_train_y.npy")
    X_VA = np.load("data/cls_val_x.npy")
    Y_VA = np.load("data/cls_val_y.npy")
    X_TE = np.load("data/cls_test_x.npy")
    Y_TE = np.load("data/cls_test_y.npy")

    TR_DS = TensorDataset(torch.from_numpy(X_TR).long(), torch.from_numpy(Y_TR).long())
    VA_DS = TensorDataset(torch.from_numpy(X_VA).long(), torch.from_numpy(Y_VA).long())
    TE_DS = TensorDataset(torch.from_numpy(X_TE).long(), torch.from_numpy(Y_TE).long())
    TR_LD = DataLoader(TR_DS, batch_size=16, shuffle=True)
    VA_LD = DataLoader(VA_DS, batch_size=32, shuffle=False)
    TE_LD = DataLoader(TE_DS, batch_size=32, shuffle=False)

    MODEL = TRANSFORMER_LIB.TransformerCls(VOC_PLUS, 0, NUM_C).to(DEVICE)
    OPT = torch.optim.AdamW(MODEL.parameters(), lr=5e-4, weight_decay=0.01)
    EPOCHS = 20
    STEPS_PER_EP = max(1, len(TR_LD))
    TOTAL_STEPS = EPOCHS * STEPS_PER_EP
    WARM = 50
    SCHED = torch.optim.lr_scheduler.LambdaLR(OPT, lr_lambda=BuildWarmupCosineLambda(WARM, TOTAL_STEPS))
    CE = nn.CrossEntropyLoss()

    TR_LOSS = []
    VA_LOSS = []
    TR_ACC = []
    VA_ACC = []

    for EP in range(EPOCHS):
        MODEL.train()
        RUN = 0.0
        COR = 0
        TOT = 0
        for XB, YB in TR_LD:
            XB = XB.to(DEVICE)
            YB = YB.to(DEVICE)
            OPT.zero_grad()
            LOGITS, _ = MODEL(XB, False)
            LOSS = CE(LOGITS, YB)
            LOSS.backward()
            OPT.step()
            SCHED.step()
            RUN += float(LOSS.item())
            PRED = LOGITS.argmax(dim=-1)
            COR += int((PRED == YB).sum().item())
            TOT += int(YB.numel())
        TR_LOSS.append(RUN / len(TR_LD))
        TR_ACC.append(COR / max(1, TOT))

        MODEL.eval()
        VR = 0.0
        VC = 0
        VT = 0
        with torch.no_grad():
            for XB, YB in VA_LD:
                XB = XB.to(DEVICE)
                YB = YB.to(DEVICE)
                LOGITS, _ = MODEL(XB, False)
                VR += float(CE(LOGITS, YB).item())
                PRED = LOGITS.argmax(dim=-1)
                VC += int((PRED == YB).sum().item())
                VT += int(YB.numel())
        VA_LOSS.append(VR / len(VA_LD))
        VA_ACC.append(VC / max(1, VT))
        print(
            "epoch",
            EP + 1,
            "train_loss",
            round(TR_LOSS[-1], 5),
            "val_loss",
            round(VA_LOSS[-1], 5),
            "train_acc",
            round(TR_ACC[-1], 4),
            "val_acc",
            round(VA_ACC[-1], 4),
            flush=True,
        )

    FIG, AX = plt.subplots(1, 2, figsize=(11, 4))
    AX[0].plot(range(1, EPOCHS + 1), TR_LOSS, label="train")
    AX[0].plot(range(1, EPOCHS + 1), VA_LOSS, label="val")
    AX[0].set_title("loss")
    AX[0].legend()
    AX[1].plot(range(1, EPOCHS + 1), TR_ACC, label="train")
    AX[1].plot(range(1, EPOCHS + 1), VA_ACC, label="val")
    AX[1].set_title("accuracy")
    AX[1].legend()
    plt.tight_layout()
    plt.savefig("models/transformer_loss_acc.png", dpi=150)
    plt.close(FIG)

    MODEL.eval()
    YT = []
    YP = []
    with torch.no_grad():
        for XB, YB in TE_LD:
            XB = XB.to(DEVICE)
            LOGITS, _ = MODEL(XB, False)
            PRED = LOGITS.argmax(dim=-1).cpu().numpy().tolist()
            YP.extend(PRED)
            YT.extend(YB.numpy().tolist())
    TE_ACC = float(np.mean(np.array(YT) == np.array(YP)))
    TE_F1 = MacroF1Multiclass(YT, YP, NUM_C)
    print("TEST_ACC", round(TE_ACC, 5), "TEST_MACRO_F1", round(TE_F1, 5), flush=True)

    CM = np.zeros((NUM_C, NUM_C), dtype=np.float64)
    for T, P in zip(YT, YP):
        CM[int(T), int(P)] += 1.0
    FIG2, AX2 = plt.subplots(figsize=(6, 5))
    IM = AX2.imshow(CM, interpolation="nearest")
    AX2.set_xticks(range(NUM_C))
    AX2.set_yticks(range(NUM_C))
    AX2.set_xticklabels(NAMES, rotation=45, ha="right")
    AX2.set_yticklabels(NAMES)
    AX2.set_xlabel("predicted")
    AX2.set_ylabel("true")
    AX2.set_title("transformer test confusion")
    plt.colorbar(IM, ax=AX2)
    plt.tight_layout()
    plt.savefig("models/transformer_confusion.png", dpi=150)
    plt.close(FIG2)

    W2I = json.load(open("embeddings/word2idx.json", encoding="utf-8"))
    INV = {0: "<PAD>"}
    for W, OID in W2I.items():
        INV[int(OID) + 1] = W

    def RowToLabels(ROW):
        OUT = []
        AR = ROW.numpy() if hasattr(ROW, "numpy") else ROW
        for J in range(AR.shape[0]):
            TID = int(AR[J])
            OUT.append(INV.get(TID, "<UNK>"))
        return OUT

    FOUND = 0
    for IDX in range(len(X_TE)):
        if FOUND >= 3:
            break
        XB = torch.from_numpy(X_TE[IDX : IDX + 1]).long().to(DEVICE)
        YB = int(Y_TE[IDX])
        with torch.no_grad():
            LOGITS, ATTN = MODEL(XB, True)
            PRED = int(LOGITS.argmax(dim=-1).item())
        if PRED != YB:
            continue
        FOUND += 1
        TOKS = RowToLabels(X_TE[IDX])
        TOKS_WITH_CLS = ["[CLS]"] + TOKS
        AH0 = ATTN[0, 0, 0, :].detach().cpu().numpy()
        AH1 = ATTN[0, 2, 0, :].detach().cpu().numpy()
        FIG3, AX3 = plt.subplots(2, 1, figsize=(14, 5))
        AX3[0].imshow(AH0.reshape(1, -1), aspect="auto")
        AX3[0].set_title("head_0 CLS row attention article " + str(IDX))
        AX3[0].set_xticks(range(len(TOKS_WITH_CLS)))
        AX3[0].set_xticklabels(TOKS_WITH_CLS, rotation=90, fontsize=6)
        AX3[1].imshow(AH1.reshape(1, -1), aspect="auto")
        AX3[1].set_title("head_2 CLS row attention article " + str(IDX))
        AX3[1].set_xticks(range(len(TOKS_WITH_CLS)))
        AX3[1].set_xticklabels(TOKS_WITH_CLS, rotation=90, fontsize=6)
        plt.tight_layout()
        plt.savefig("models/transformer_attn_article_" + str(IDX) + ".png", dpi=150)
        plt.close(FIG3)

    torch.save(
        {
            "state_dict": MODEL.state_dict(),
            "class_names": NAMES,
            "vocab_plus_one": VOC_PLUS,
            "d_model": 128,
            "test_acc": TE_ACC,
            "test_macro_f1": TE_F1,
        },
        "models/transformer_cls.pt",
    )
    print("saved models/transformer_cls.pt and plots", flush=True)


if __name__ == "__main__":
    Main()
