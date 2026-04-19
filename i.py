"""
MODULE I — FIVE-WAY TOPIC LABELS FROM KEYWORD MATCHING + 256-TOKEN ID SEQUENCES + STRATIFIED SPLITS.
OUTPUTS NUMPY ARRAYS UNDER data/cls_* FOR j.py TRAINING.
"""

import os
import re
import sys
import json
import random
import collections

import numpy as np

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None


def ConfigureStdoutUtf8():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def SplitDocsFiltered(CLEANED_PATH):
    with open(CLEANED_PATH, encoding="utf-8") as HANDLE:
        LINE_LIST = HANDLE.read().splitlines()
    DOCS = {}
    CURRENT_ID = None
    for RAW in LINE_LIST:
        STRIP = RAW.strip()
        MARK = re.match(r"^\[(\d+)\]\s*$", STRIP)
        if MARK:
            CURRENT_ID = int(MARK.group(1))
            DOCS[CURRENT_ID] = []
        elif CURRENT_ID is not None:
            if not STRIP or re.match(r"^=+$", STRIP):
                continue
            DOCS[CURRENT_ID].append(RAW)
    MERGED = {}
    for DOC_KEY, CHUNKS in DOCS.items():
        MERGED[DOC_KEY] = " ".join(CHUNKS).split()
    return MERGED


def BuildWordToNewIndex(WORD_TO_INDEX):
    W2NEW = {}
    for W, OLD in WORD_TO_INDEX.items():
        W2NEW[W] = int(OLD) + 1
    NEW_UNK = int(WORD_TO_INDEX["<UNK>"]) + 1
    VOC_PLUS = len(WORD_TO_INDEX) + 1
    return W2NEW, NEW_UNK, VOC_PLUS


def ScoreCategories(TEXT_LOWER, RULES):
    SCORES = []
    for LAB, KWS in enumerate(RULES):
        C = 0
        for KW in KWS:
            if KW.lower() in TEXT_LOWER:
                C += 1
        SCORES.append(C)
    return SCORES


def Main():
    ConfigureStdoutUtf8()
    random.seed(42)
    np.random.seed(42)

    META = json.load(open("Metadata.json", encoding="utf-8"))
    W2I = json.load(open("embeddings/word2idx.json", encoding="utf-8"))
    W2NEW, NEW_UNK, VOC_PLUS = BuildWordToNewIndex(W2I)
    DOCS = SplitDocsFiltered("cleaned.txt")

    RULES = [
        [
            "election",
            "government",
            "minister",
            "parliament",
            "انتخاب",
            "حکومت",
            "وزیر",
            "پارلیمنٹ",
            "وزیراعظم",
            "الیکشن",
            "جماعت",
            "سیاست",
            "کابینہ",
        ],
        [
            "cricket",
            "match",
            "team",
            "player",
            "score",
            "کرکٹ",
            "میچ",
            "ٹیم",
            "کھلاڑی",
            "رنز",
            "وکٹ",
            "سیریز",
            "اسٹیڈیم",
            "بولر",
            "بیٹسمین",
        ],
        [
            "inflation",
            "trade",
            "bank",
            "gdp",
            "budget",
            "مہنگائی",
            "تجارت",
            "بینک",
            "بجٹ",
            "معیشت",
            "روپیہ",
            "ڈالر",
            "قرض",
            "برآمد",
            "درآمد",
            "ٹیکس",
        ],
        [
            "un",
            "treaty",
            "foreign",
            "bilateral",
            "conflict",
            "اقوام",
            "متحدہ",
            "معاہدہ",
            "خارجہ",
            "دوطرفہ",
            "تنازع",
            "سرحد",
            "عالمی",
            "امریکہ",
            "چین",
            "روس",
            "یوکرین",
        ],
        [
            "hospital",
            "disease",
            "vaccine",
            "flood",
            "education",
            "ہسپتال",
            "بیماری",
            "ویکسین",
            "سیلاب",
            "تعلیم",
            "صحت",
            "ڈاکٹر",
            "دوائی",
            "طلبہ",
            "سکول",
            "یونیورسٹی",
        ],
    ]
    NAMES = ["politics", "sports", "economy", "international", "health_society"]

    IDS = []
    YS = []

    for DK, TOKS in sorted(DOCS.items()):
        KEY = str(DK)
        if KEY not in META:
            continue
        TITLE = META[KEY].get("title", "")
        BODY = " ".join(TOKS[:400])
        COMB = (TITLE + " " + BODY).lower()
        SC = ScoreCategories(COMB, RULES)
        if max(SC) == 0:
            LAB = 0
        else:
            LAB = int(np.argmax(np.array(SC, dtype=np.float64)))
        SEQ = [W2NEW.get(W, NEW_UNK) for W in TOKS[:256]]
        while len(SEQ) < 256:
            SEQ.append(0)
        MAT = np.array(SEQ[:256], dtype=np.int64)
        IDS.append(MAT.copy())
        YS.append(LAB)

    X = np.stack(IDS, axis=0)
    Y = np.array(YS, dtype=np.int64)

    if train_test_split is None:
        raise SystemExit("sklearn REQUIRED FOR STRATIFIED SPLITS — pip install scikit-learn")
    X_TR, X_TEMP, Y_TR, Y_TEMP = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
    X_VA, X_TE, Y_VA, Y_TE = train_test_split(X_TEMP, Y_TEMP, test_size=0.5, random_state=42, stratify=Y_TEMP)

    os.makedirs("data", exist_ok=True)
    np.save("data/cls_train_x.npy", X_TR)
    np.save("data/cls_train_y.npy", Y_TR)
    np.save("data/cls_val_x.npy", X_VA)
    np.save("data/cls_val_y.npy", Y_VA)
    np.save("data/cls_test_x.npy", X_TE)
    np.save("data/cls_test_y.npy", Y_TE)
    with open("data/cls_meta.json", "w", encoding="utf-8") as OUT:
        json.dump({"class_names": NAMES, "vocab_plus_one": VOC_PLUS, "pad_idx": 0}, OUT, indent=2)

    CNT = collections.Counter(Y.tolist())
    print("FULL CORPUS CLASS DISTRIBUTION (COUNT PER LABEL INDEX):", dict(CNT), flush=True)
    print("TRAIN", dict(collections.Counter(Y_TR.tolist())), flush=True)
    print("VAL", dict(collections.Counter(Y_VA.tolist())), flush=True)
    print("TEST", dict(collections.Counter(Y_TE.tolist())), flush=True)
    print("saved data/cls_* arrays and data/cls_meta.json", flush=True)


if __name__ == "__main__":
    Main()
