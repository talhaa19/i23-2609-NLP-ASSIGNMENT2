"""
MODULE D — WORD2VEC EVALUATION: NEIGHBOURS, ANALOGIES, FOUR-WAY CONDITION BENCH (C1–C4).
ASSUMES b.py ALREADY BUILT word2idx + ppmi; ASSUMES c.py ALREADY TRAINED DEFAULT W2V (C3).
THIS SCRIPT ALSO FITS C2 (RAW) AND C4 (DIM=200) SO ALL CONDITIONS ARE REPORTED IN ONE RUN.
"""

import os
import sys
import json
import numpy as np

import c as WORD2VEC_TRAINER


def ConfigureStdoutUtf8():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def LoadWord2Idx(PATH):
    with open(PATH, encoding="utf-8") as HANDLE:
        return json.load(HANDLE)


def BuildIndexToWord(WORD_TO_INDEX):
    return {INDEX: WORD for WORD, INDEX in WORD_TO_INDEX.items()}


def RowL2Normalize(MATRIX):
    NORMS = np.linalg.norm(MATRIX, axis=1, keepdims=True) + 1e-12
    return MATRIX / NORMS


def TopKNeighbours(NORMALIZED_EMB, WORD_TO_INDEX, INDEX_TO_WORD, QUERY_SURFACE, K):
    if QUERY_SURFACE not in WORD_TO_INDEX:
        return []
    QI = WORD_TO_INDEX[QUERY_SURFACE]
    SIMS = NORMALIZED_EMB @ NORMALIZED_EMB[QI]
    SIMS[QI] = -1e9
    ORDER = np.argsort(-SIMS)[:K]
    return [(INDEX_TO_WORD[J], float(SIMS[J])) for J in ORDER]


def AnalogyTopCandidates(NORMALIZED_EMB, WORD_TO_INDEX, A, B, C, TOP_K, BAN_SELF=True):
    """
    VECTOR OFFSET TEST: v(b) - v(a) + v(c) COSINE-CLOSEST TO OTHER VOCAB VECTORS.
    """
    for W in (A, B, C):
        if W not in WORD_TO_INDEX:
            return []
    IA, IB, IC = WORD_TO_INDEX[A], WORD_TO_INDEX[B], WORD_TO_INDEX[C]
    VEC = NORMALIZED_EMB[IB] - NORMALIZED_EMB[IA] + NORMALIZED_EMB[IC]
    VEC = VEC / (np.linalg.norm(VEC) + 1e-12)
    SIMS = NORMALIZED_EMB @ VEC
    if BAN_SELF:
        SIMS[IA] = -1e9
        SIMS[IB] = -1e9
        SIMS[IC] = -1e9
    ORDER = np.argsort(-SIMS)[:TOP_K]
    INDEX_TO_WORD = BuildIndexToWord(WORD_TO_INDEX)
    return [INDEX_TO_WORD[J] for J in ORDER]


def MeanReciprocalRank(NORMALIZED_EMB, WORD_TO_INDEX, LABELED_PAIRS):
    """
    EACH PAIR IS (QUERY_WORD, GOLD_TARGET_WORD): RANK BY COSINE TO QUERY EMBEDDING.
    """
    RECIPROCALS = []
    for QUERY_WORD, GOLD_WORD in LABELED_PAIRS:
        if QUERY_WORD not in WORD_TO_INDEX or GOLD_WORD not in WORD_TO_INDEX:
            RECIPROCALS.append(0.0)
            continue
        QI = WORD_TO_INDEX[QUERY_WORD]
        GI = WORD_TO_INDEX[GOLD_WORD]
        SIMS = NORMALIZED_EMB @ NORMALIZED_EMB[QI]
        SIMS[QI] = -1e9
        ORDER = np.argsort(-SIMS)
        RANK = int(np.where(ORDER == GI)[0][0]) + 1
        RECIPROCALS.append(1.0 / float(RANK))
    return float(np.mean(RECIPROCALS)) if RECIPROCALS else 0.0


def PrintNeighbourBlock(TITLE, NORMALIZED_EMB, WORD_TO_INDEX, QUERY_LIST, K):
    print(TITLE)
    INDEX_TO_WORD = BuildIndexToWord(WORD_TO_INDEX)
    for Q in QUERY_LIST:
        NB = TopKNeighbours(NORMALIZED_EMB, WORD_TO_INDEX, INDEX_TO_WORD, Q, K)
        if not NB:
            print("  ", Q, "-> not in vocab")
            continue
        STRINGS = "  ".join([W for W, _SC in NB])
        print("  ", Q, ":", STRINGS)


def RunFourConditionSuite(WORD_TO_INDEX):
    """
    C1 PPMI ROWS, C2 RAW W2V, C3 CLEANED W2V (FROM DISK), C4 CLEANED W2V D=200 (RETRAINED HERE).
    """
    VOCAB_SIZE = len(WORD_TO_INDEX)
    FIVE_QUERIES = ["پاکستان", "کرکٹ", "حکومت", "فلم", "انڈیا"]

    MRR_PAIRS = [
        ("کرکٹ", "میچ"),
        ("میچ", "کرکٹ"),
        ("فلم", "اداکار"),
        ("اداکار", "فلم"),
        ("ٹیم", "کھلاڑی"),
        ("کھلاڑی", "ٹیم"),
        ("پاکستان", "انڈیا"),
        ("انڈیا", "پاکستان"),
        ("حکومت", "وزیر"),
        ("وزیر", "حکومت"),
        ("فوج", "جنرل"),
        ("جنرل", "فوج"),
        ("صحت", "بیماری"),
        ("تعلیم", "سکول"),
        ("رنز", "وکٹ"),
        ("بیٹسمین", "بولر"),
        ("سعودی", "عرب"),
        ("ٹرمپ", "امریکی"),
        ("کپتان", "ٹیم"),
        ("سیریز", "میچ"),
    ]

    RESULT_ROWS = []

    # --- C1: PPMI BASELINE VECTORS (ROWS OF PPMI MATRIX, L2-NORMALIZED) ---
    PPMI_PATH = "embeddings/ppmi_matrix.npy"
    if not os.path.isfile(PPMI_PATH):
        raise SystemExit("MISSING " + PPMI_PATH + " — RUN b.py FIRST.")
    PPMI_MAT = np.load(PPMI_PATH).astype(np.float64)
    if PPMI_MAT.shape[0] != VOCAB_SIZE:
        raise SystemExit("PPMI ROW COUNT DOES NOT MATCH word2idx LENGTH — CHECK PIPELINE.")
    C1_EMB = RowL2Normalize(PPMI_MAT.astype(np.float32))
    PrintNeighbourBlock("C1 PPMI baseline — top-5 neighbours", C1_EMB, WORD_TO_INDEX, FIVE_QUERIES, 5)
    C1_MRR = MeanReciprocalRank(C1_EMB, WORD_TO_INDEX, MRR_PAIRS)
    print("C1 MRR (20 labelled pairs):", round(C1_MRR, 5))
    RESULT_ROWS.append(("C1", "PPMI baseline rows", C1_MRR))

    # --- C2: SKIP-GRAM ON raw.txt (SAME VOCAB / UNK MAPPING AS CLEANED PIPELINE) ---
    print("\nTRAINING C2 (raw.txt, d=100) — THIS CAN TAKE SEVERAL MINUTES …")
    WORD2VEC_TRAINER.TrainSkipGram(
        CORPUS_PATH="raw.txt",
        WORD_TO_INDEX=WORD_TO_INDEX,
        EMBEDDING_DIM=100,
        WINDOW_RADIUS=5,
        NEGATIVE_SAMPLES=10,
        BATCH_SIZE=512,
        LEARNING_RATE=0.001,
        NUM_EPOCHS=5,
        EMB_SAVE_PATH="embeddings/embeddings_w2v_raw.npy",
        PLOT_SAVE_PATH="embeddings/loss_w2v_raw.png",
        PRINT_EVERY_BATCHES=800,
    )
    C2_EMB = RowL2Normalize(np.load("embeddings/embeddings_w2v_raw.npy").astype(np.float32))
    PrintNeighbourBlock("C2 Word2Vec raw.txt — top-5 neighbours", C2_EMB, WORD_TO_INDEX, FIVE_QUERIES, 5)
    C2_MRR = MeanReciprocalRank(C2_EMB, WORD_TO_INDEX, MRR_PAIRS)
    print("C2 MRR (20 labelled pairs):", round(C2_MRR, 5))
    RESULT_ROWS.append(("C2", "skip-gram raw.txt d=100", C2_MRR))

    # --- C3: MAIN CLEANED MODEL WEIGHTS PRODUCED BY c.py ---
    MAIN_PATH = "embeddings/embeddings_w2v.npy"
    if not os.path.isfile(MAIN_PATH):
        raise SystemExit("MISSING " + MAIN_PATH + " — RUN c.py FIRST FOR THE PRIMARY MODEL.")
    C3_EMB = RowL2Normalize(np.load(MAIN_PATH).astype(np.float32))
    PrintNeighbourBlock("C3 Word2Vec cleaned.txt — top-5 neighbours", C3_EMB, WORD_TO_INDEX, FIVE_QUERIES, 5)
    C3_MRR = MeanReciprocalRank(C3_EMB, WORD_TO_INDEX, MRR_PAIRS)
    print("C3 MRR (20 labelled pairs):", round(C3_MRR, 5))
    RESULT_ROWS.append(("C3", "skip-gram cleaned d=100", C3_MRR))

    # --- C4: SAME AS C3 BUT EMBEDDING WIDTH 200 ---
    print("\nTRAINING C4 (cleaned.txt, d=200) — LONGER RUN …")
    WORD2VEC_TRAINER.TrainSkipGram(
        CORPUS_PATH="cleaned.txt",
        WORD_TO_INDEX=WORD_TO_INDEX,
        EMBEDDING_DIM=200,
        WINDOW_RADIUS=5,
        NEGATIVE_SAMPLES=10,
        BATCH_SIZE=512,
        LEARNING_RATE=0.001,
        NUM_EPOCHS=5,
        EMB_SAVE_PATH="embeddings/embeddings_w2v_d200.npy",
        PLOT_SAVE_PATH="embeddings/loss_w2v_d200.png",
        PRINT_EVERY_BATCHES=800,
    )
    C4_EMB = RowL2Normalize(np.load("embeddings/embeddings_w2v_d200.npy").astype(np.float32))
    PrintNeighbourBlock("C4 Word2Vec cleaned d=200 — top-5 neighbours", C4_EMB, WORD_TO_INDEX, FIVE_QUERIES, 5)
    C4_MRR = MeanReciprocalRank(C4_EMB, WORD_TO_INDEX, MRR_PAIRS)
    print("C4 MRR (20 labelled pairs):", round(C4_MRR, 5))
    RESULT_ROWS.append(("C4", "skip-gram cleaned d=200", C4_MRR))

    print("\nSUMMARY TABLE (CONDITION ID, DESCRIPTION, MRR@20 PAIRS)")
    for ROW in RESULT_ROWS:
        print("  ", ROW[0], " | ", ROW[1], " | MRR=", round(ROW[2], 5))

    print(
        "\nDISCUSSION (BRIEF): C1 PPMI CAPTURES RAW CO-OCCURRENCE STRUCTURE BUT IS SPARSE AND "
        "LESS SMOOTH THAN LEARNED EMBEDDINGS. C2 ON raw.txt INJECTS NOISY ENGLISH / LAYOUT TOKENS, "
        "WHICH OFTEN HURTS SEMANTIC CLUSTERING VERSUS C3 ON cleaned.txt. C4 WITH d=200 USUALLY "
        "IMPROVES MARGINALLY ON RARE RELATIONS AT THE COST OF MORE PARAMETERS AND TRAINING TIME. "
        "MRR NUMBERS SHOULD BE READ AS A COARSE SANITY CHECK BECAUSE THE 20 PAIRS ARE HAND-PICKED "
        "NEWS-DOMAIN ASSOCIATIONS, NOT A GOLD STANDARD BENCHMARK."
    )


def Main():
    ConfigureStdoutUtf8()
    WORD_TO_INDEX = LoadWord2Idx("embeddings/word2idx.json")
    INDEX_TO_WORD = BuildIndexToWord(WORD_TO_INDEX)

    # --- EIGHT QUERY WORDS: ROMAN LABELS IN PRINTOUT, URDU SCRIPT FOR LOOKUP ---
    QUERY_SPECS = [
        ("Pakistan", "پاکستان"),
        ("Hukumat", "حکومت"),
        ("Adalat", "عدالت"),
        ("Maeeshat", "معیشت"),
        ("Fauj", "فوج"),
        ("Sehat", "صحت"),
        ("Taleem", "تعلیم"),
        ("Aabadi", "آبادی"),
    ]

    MAIN_EMB_PATH = "embeddings/embeddings_w2v.npy"
    if not os.path.isfile(MAIN_EMB_PATH):
        raise SystemExit("MISSING " + MAIN_EMB_PATH + " — RUN c.py FIRST.")
    NORMALIZED_MAIN = RowL2Normalize(np.load(MAIN_EMB_PATH).astype(np.float32))

    print("top-10 cosine neighbours (main cleaned Word2Vec, C3 weights on disk)")
    for ROMAN_LABEL, URDU_SURFACE in QUERY_SPECS:
        NB = TopKNeighbours(NORMALIZED_MAIN, WORD_TO_INDEX, INDEX_TO_WORD, URDU_SURFACE, 10)
        if not NB:
            print(ROMAN_LABEL, "(", URDU_SURFACE, ") -> not in vocab")
            continue
        print(ROMAN_LABEL, "(", URDU_SURFACE, "):", "  ".join([W for W, _ in NB]))

    # TEN ANALOGY TASKS — GOLD ANSWER IS THE FIRST ENTRY IN ACCEPT_LIST FOR AUTO SCORING PRINT.
    ANALOGY_SPECS = [
        ("کرکٹ", "میچ", "فلم", ["ریلیز", "فلموں", "اداکار"]),
        ("حکومت", "وزیر", "فوج", ["جنرل", "افواج", "سربراہ"]),
        ("ٹیم", "کھلاڑی", "فلم", ["اداکار", "کردار", "ریلیز"]),
        ("پاکستان", "انڈیا", "میچ", ["کرکٹ", "سیریز", "ٹی"]),
        ("بیٹسمین", "رنز", "بولر", ["وکٹ", "گیند", "شکار"]),
        ("سعودی", "عرب", "امریکی", ["ٹرمپ", "صدر", "ریاست"]),
        ("کپتان", "ٹیم", "وزیر", ["حکومت", "کابینہ", "ملک"]),
        ("صحت", "بیماری", "تعلیم", ["طلبہ", "سکول", "یونیورسٹی"]),
        ("فوج", "جنرل", "عدالت", ["جج", "فیصلہ", "سزا"]),
        ("کرکٹر", "کھلاڑی", "اداکار", ["فلم", "کردار", "ریلیز"]),
    ]

    print("\n10 analogy tests — v(b) - v(a) + v(c); showing top-3 candidates (Urdu)")
    CORRECT_COUNT = 0
    for A, B, C, ACCEPT_LIST in ANALOGY_SPECS:
        TOP3 = AnalogyTopCandidates(NORMALIZED_MAIN, WORD_TO_INDEX, A, B, C, 3, BAN_SELF=True)
        HIT = any(X in TOP3 for X in ACCEPT_LIST)
        if HIT:
            CORRECT_COUNT += 1
        print("  ", A, ":", B, "::", C, ":?  => top3:", " ".join(TOP3), "  hit_accept_list=", HIT)
    print("analogy hits in top-3 (lenient accept list):", CORRECT_COUNT, "/ 10")

    print("\n===== FOUR-CONDITION BLOCK (C1–C4) =====\n")
    RunFourConditionSuite(WORD_TO_INDEX)


if __name__ == "__main__":
    Main()
