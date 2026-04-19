"""
MODULE C — SKIP-GRAM WORD2VEC (NEGATIVE SAMPLING + BCE) IN PYTORCH, NO HUGGINGFACE.
OUTPUTS: embeddings/embeddings_w2v.npy AND embeddings/loss_w2v.png (MAIN / C3 RUN).
OTHER CONDITIONS (C2 RAW, C4 DIM=200) ARE TRAINED FROM d.py BY IMPORTING TrainSkipGram.
"""

import os
import re
import sys
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


def ConfigureStdoutUtf8():
    """KEEP URDU LOGS READABLE ON WINDOWS TERMINALS."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def SplitDocsByArticleMarkers(FILE_PATH):
    """SAME [N] ARTICLE MARKER CONVENTION AS b.py — ONE LOGICAL DOC PER BRACKET LINE."""
    with open(FILE_PATH, encoding="utf-8") as INPUT_HANDLE:
        LINE_LIST = INPUT_HANDLE.read().splitlines()

    DOC_TOKEN_LINES = {}
    CURRENT_DOC_ID = None
    for RAW_LINE in LINE_LIST:
        STRIPPED = RAW_LINE.strip()
        MARKER_MATCH = re.match(r"^\[(\d+)\]\s*$", STRIPPED)
        if MARKER_MATCH:
            CURRENT_DOC_ID = int(MARKER_MATCH.group(1))
            DOC_TOKEN_LINES[CURRENT_DOC_ID] = []
        elif CURRENT_DOC_ID is not None:
            DOC_TOKEN_LINES[CURRENT_DOC_ID].append(RAW_LINE)

    DOCS_AS_TOKEN_LISTS = {}
    for DOC_KEY, LINE_CHUNK_LIST in DOC_TOKEN_LINES.items():
        MERGED_BODY = " ".join(LINE_CHUNK_LIST)
        DOCS_AS_TOKEN_LISTS[DOC_KEY] = MERGED_BODY.split()
    return DOCS_AS_TOKEN_LISTS


def MapTokensToIndices(TOKEN_LIST, WORD_TO_INDEX):
    """OOV TOKENS FALL BACK TO THE RESERVED <UNK> INDEX SO SHAPES STAY FIXED."""
    UNK_INDEX = WORD_TO_INDEX["<UNK>"]
    return [WORD_TO_INDEX[TOK] if TOK in WORD_TO_INDEX else UNK_INDEX for TOK in TOKEN_LIST]


def BuildCenterContextIndexArrays(DOCS_AS_TOKEN_LISTS, WORD_TO_INDEX, WINDOW_RADIUS):
    """
    MATERIALIZE EVERY (CENTER, POSITIVE_CONTEXT) INDEX PAIR FOR SKIP-GRAM.
    WINDOW IS SYMMETRIC: ALL OFFSETS IN [-K, K] EXCLUDING ZERO.
    """
    CENTER_LIST = []
    CONTEXT_LIST = []
    for TOKEN_LIST in DOCS_AS_TOKEN_LISTS.values():
        INDEX_SEQ = MapTokensToIndices(TOKEN_LIST, WORD_TO_INDEX)
        SEQ_LEN = len(INDEX_SEQ)
        for CENTER_POS in range(SEQ_LEN):
            CENTER_ID = INDEX_SEQ[CENTER_POS]
            LOW = max(0, CENTER_POS - WINDOW_RADIUS)
            HIGH = min(SEQ_LEN, CENTER_POS + WINDOW_RADIUS + 1)
            for CONTEXT_POS in range(LOW, HIGH):
                if CONTEXT_POS == CENTER_POS:
                    continue
                CONTEXT_LIST.append(INDEX_SEQ[CONTEXT_POS])
                CENTER_LIST.append(CENTER_ID)
    CENTER_ARRAY = np.asarray(CENTER_LIST, dtype=np.int64)
    CONTEXT_ARRAY = np.asarray(CONTEXT_LIST, dtype=np.int64)
    return CENTER_ARRAY, CONTEXT_ARRAY


def BuildNoiseSamplingProbabilities(VOCAB_SIZE, INDEX_STREAM_FOR_COUNTS):
    """
    NOISE DISTRIBUTION P_n(w) ∝ COUNT(w)^0.75 — CLASSIC WORD2VEC SUBSAMPLE POWER.
    RETURN NORMALIZED FLOAT32 TENSOR ON CPU FOR torch.multinomial.
    """
    COUNTS = np.zeros(VOCAB_SIZE, dtype=np.float64)
    for IDX in INDEX_STREAM_FOR_COUNTS:
        COUNTS[IDX] += 1.0
    POWERED = np.power(COUNTS + 1e-12, 0.75)
    PROBS = POWERED / POWERED.sum()
    return torch.from_numpy(PROBS.astype(np.float32))


class SkipGramBceModule(nn.Module):
    """
    TWO INDEPENDENT EMBEDDING TABLES: V FOR CENTERS, U FOR CONTEXTS (POS + NEG).
    INNER PRODUCTS FEED LOG-SIGMOID BCE AS IN THE ASSIGNMENT FORMULA SHEET.
    """

    def __init__(self, VOCAB_SIZE, EMBEDDING_DIM):
        super().__init__()
        self.V_EMB = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.U_EMB = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        nn.init.uniform_(self.V_EMB.weight, -0.5 / EMBEDDING_DIM, 0.5 / EMBEDDING_DIM)
        nn.init.uniform_(self.U_EMB.weight, -0.5 / EMBEDDING_DIM, 0.5 / EMBEDDING_DIM)

    def forward(self, CENTER_IDS, POS_CONTEXT_IDS, NEG_CONTEXT_IDS):
        """
        CENTER_IDS [B], POS_CONTEXT_IDS [B], NEG_CONTEXT_IDS [B, K_NEG].
        RETURNS SCALAR LOSS FOR THIS MINI-BATCH.
        """
        VEC_C = self.V_EMB(CENTER_IDS)
        VEC_U_POS = self.U_EMB(POS_CONTEXT_IDS)
        POS_SCORES = (VEC_C * VEC_U_POS).sum(dim=1)
        VEC_U_NEG = self.U_EMB(NEG_CONTEXT_IDS)
        NEG_SCORES = (VEC_C.unsqueeze(1) * VEC_U_NEG).sum(dim=2)
        POS_TERM = -F.logsigmoid(POS_SCORES)
        NEG_TERM = -F.logsigmoid(-NEG_SCORES).sum(dim=1)
        return (POS_TERM + NEG_TERM).mean()


def TrainSkipGram(
    CORPUS_PATH,
    WORD_TO_INDEX,
    EMBEDDING_DIM,
    WINDOW_RADIUS,
    NEGATIVE_SAMPLES,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    EMB_SAVE_PATH,
    PLOT_SAVE_PATH,
    PRINT_EVERY_BATCHES,
):
    """
    FULL TRAINING LOOP: PAIR PRECOMPUTE → ADAM → LOSS LOGGING → (V+U)/2 EXPORT.
    RETURNS A LIST OF MEAN LOSS VALUES (ONE FLOAT PER EPOCH) FOR DOWNSTREAM PLOTTING.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    DOCS = SplitDocsByArticleMarkers(CORPUS_PATH)
    VOCAB_SIZE = len(WORD_TO_INDEX)
    FLAT_STREAM = []
    for TOKEN_LIST in DOCS.values():
        FLAT_STREAM.extend(MapTokensToIndices(TOKEN_LIST, WORD_TO_INDEX))

    CENTER_ARRAY, CONTEXT_ARRAY = BuildCenterContextIndexArrays(DOCS, WORD_TO_INDEX, WINDOW_RADIUS)
    PAIR_COUNT = CENTER_ARRAY.shape[0]
    NOISE_PROBS = BuildNoiseSamplingProbabilities(VOCAB_SIZE, FLAT_STREAM)

    MODEL = SkipGramBceModule(VOCAB_SIZE, EMBEDDING_DIM).to(DEVICE)
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)

    EPOCH_MEAN_LOSSES = []
    BATCH_LOSS_TRACE = []

    for EPOCH_INDEX in range(NUM_EPOCHS):
        SHUFFLE_ORDER = np.random.permutation(PAIR_COUNT)
        RUNNING_LOSS_SUM = 0.0
        RUNNING_BATCH_COUNT = 0
        BATCH_START_TIME = time.time()

        for START in range(0, PAIR_COUNT, BATCH_SIZE):
            SLICE = SHUFFLE_ORDER[START : START + BATCH_SIZE]
            if SLICE.size == 0:
                continue

            BATCH_CENTERS = torch.from_numpy(CENTER_ARRAY[SLICE]).long().to(DEVICE)
            BATCH_POS_CTX = torch.from_numpy(CONTEXT_ARRAY[SLICE]).long().to(DEVICE)
            ACTUAL_BATCH = BATCH_CENTERS.shape[0]

            NEG_SAMPLES = torch.multinomial(
                NOISE_PROBS,
                num_samples=ACTUAL_BATCH * NEGATIVE_SAMPLES,
                replacement=True,
            ).view(ACTUAL_BATCH, NEGATIVE_SAMPLES).to(DEVICE)

            OPTIMIZER.zero_grad()
            LOSS_SCALAR = MODEL(BATCH_CENTERS, BATCH_POS_CTX, NEG_SAMPLES)
            LOSS_SCALAR.backward()
            OPTIMIZER.step()

            LV = float(LOSS_SCALAR.detach().cpu().item())
            RUNNING_LOSS_SUM += LV
            RUNNING_BATCH_COUNT += 1
            BATCH_LOSS_TRACE.append(LV)

            if PRINT_EVERY_BATCHES > 0 and RUNNING_BATCH_COUNT % PRINT_EVERY_BATCHES == 0:
                ELAPSED = time.time() - BATCH_START_TIME
                print(
                    "epoch",
                    EPOCH_INDEX + 1,
                    "batch",
                    RUNNING_BATCH_COUNT,
                    "loss",
                    round(LV, 5),
                    "elapsed_s",
                    round(ELAPSED, 2),
                )

        MEAN_EPOCH_LOSS = RUNNING_LOSS_SUM / max(1, RUNNING_BATCH_COUNT)
        EPOCH_MEAN_LOSSES.append(MEAN_EPOCH_LOSS)
        print("epoch", EPOCH_INDEX + 1, "mean_loss", round(MEAN_EPOCH_LOSS, 5))

    os.makedirs(os.path.dirname(EMB_SAVE_PATH), exist_ok=True)
    V_WEIGHT = MODEL.V_EMB.weight.detach().cpu().numpy().astype(np.float32)
    U_WEIGHT = MODEL.U_EMB.weight.detach().cpu().numpy().astype(np.float32)
    MERGED = (V_WEIGHT + U_WEIGHT) / 2.0
    np.save(EMB_SAVE_PATH, MERGED)

    FIG, AX = plt.subplots(figsize=(8, 5))
    AX.plot(range(1, len(BATCH_LOSS_TRACE) + 1), BATCH_LOSS_TRACE, linewidth=0.8, alpha=0.85)
    AX.set_xlabel("minibatch index (global across all epochs)")
    AX.set_ylabel("training loss (BCE skip-gram)")
    AX.set_title("skip-gram word2vec loss curve")
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=150)
    plt.close(FIG)

    return EPOCH_MEAN_LOSSES, BATCH_LOSS_TRACE


def Main():
    """
    DEFAULT COMMIT-2 RUN: CLEANED CORPUS, DIM=100, HYPERPARAMS EXACTLY AS SPEC TABLE.
    """
    ConfigureStdoutUtf8()
    with open("embeddings/word2idx.json", encoding="utf-8") as JSON_HANDLE:
        WORD_TO_INDEX = json.load(JSON_HANDLE)

    EMBEDDING_DIM = 100
    WINDOW_RADIUS = 5
    NEGATIVE_SAMPLES = 10
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    PRINT_EVERY_BATCHES = 400

    TrainSkipGram(
        CORPUS_PATH="cleaned.txt",
        WORD_TO_INDEX=WORD_TO_INDEX,
        EMBEDDING_DIM=EMBEDDING_DIM,
        WINDOW_RADIUS=WINDOW_RADIUS,
        NEGATIVE_SAMPLES=NEGATIVE_SAMPLES,
        BATCH_SIZE=BATCH_SIZE,
        LEARNING_RATE=LEARNING_RATE,
        NUM_EPOCHS=NUM_EPOCHS,
        EMB_SAVE_PATH="embeddings/embeddings_w2v.npy",
        PLOT_SAVE_PATH="embeddings/loss_w2v.png",
        PRINT_EVERY_BATCHES=PRINT_EVERY_BATCHES,
    )
    print("saved embeddings/embeddings_w2v.npy and embeddings/loss_w2v.png")


if __name__ == "__main__":
    Main()
