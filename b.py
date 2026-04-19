"""
ENTRY SCRIPT B — TF-IDF + PPMI + t-SNE + NEIGHBOUR DUMP FOR ASSIGNMENT PART 1.
NAMING: SINGLE-LETTER FILE (b.py); IDENTIFIERS MOSTLY SCREAMING_SNAKE / PASCALCASE.
"""

import os
import re
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict


def ConfigureStdoutUtf8():
    """SAME UTF-8 FIX AS a.py SO URDU QUERY / WORD PRINTS SURVIVE WINDOWS CP1252."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def SplitDocsByArticleMarkers(FILE_PATH):
    """
    PARSE cleaned.txt INTO A DICT: INTEGER ARTICLE ID -> LIST OF WHITESPACE TOKENS.
    LINES THAT LOOK LIKE [42] START A NEW DOCUMENT BUCKET.
    """
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


def BuildCappedVocabularyWithUnk(DOCS_AS_TOKEN_LISTS, MAX_REAL_WORDS):
    """
    TAKE TOP MAX_REAL_WORDS TYPES BY CORPUS FREQUENCY, THEN APPEND <UNK> SLOT.
    RETURNS (WORD_TO_INDEX, VOCAB_LIST_IN_ORDER, TOP_WORD_SET_FOR OOV CHECKS).
    """
    GLOBAL_FREQ = Counter()
    for TOKEN_LIST in DOCS_AS_TOKEN_LISTS.values():
        GLOBAL_FREQ.update(TOKEN_LIST)

    UNK_LITERAL = "<UNK>"
    TOP_WORDS_IN_ORDER = [WORD for WORD, _COUNT in GLOBAL_FREQ.most_common(MAX_REAL_WORDS)]
    TOP_WORD_SET = set(TOP_WORDS_IN_ORDER)

    WORD_TO_INDEX = {WORD: INDEX for INDEX, WORD in enumerate(TOP_WORDS_IN_ORDER)}
    WORD_TO_INDEX[UNK_LITERAL] = len(TOP_WORDS_IN_ORDER)
    VOCAB_LIST = TOP_WORDS_IN_ORDER + [UNK_LITERAL]
    return WORD_TO_INDEX, VOCAB_LIST, TOP_WORD_SET


def MapTokensToIndices(TOKEN_LIST, WORD_TO_INDEX, TOP_WORD_SET):
    """MAP EACH SURFACE TOKEN TO AN INT INDEX; OOV ROUTES TO THE RESERVED UNK INDEX."""
    UNK_INDEX = WORD_TO_INDEX["<UNK>"]
    return [WORD_TO_INDEX[TOK] if TOK in TOP_WORD_SET else UNK_INDEX for TOK in TOKEN_LIST]


def Main():
    """
    FULL PART-1 MATRIX PIPELINE: TF-IDF SAVE, PPMI SAVE, PLOT, NEIGHBOUR PRINTS.
    """
    ConfigureStdoutUtf8()
    os.makedirs("embeddings", exist_ok=True)

    if not os.path.isfile("cleaned.txt") or not os.path.isfile("Metadata.json"):
        raise SystemExit("MISSING cleaned.txt OR Metadata.json — STOPPING EARLY.")

    with open("Metadata.json", encoding="utf-8") as META_HANDLE:
        METADATA_OBJECT = json.load(META_HANDLE)

    DOCS = SplitDocsByArticleMarkers("cleaned.txt")
    NUM_DOCS = len(DOCS)
    MAX_VOCAB_WITHOUT_UNK = 10000

    WORD_TO_INDEX, VOCAB_LIST, TOP_WORD_SET = BuildCappedVocabularyWithUnk(DOCS, MAX_VOCAB_WITHOUT_UNK)
    VOCAB_SIZE = len(VOCAB_LIST)

    # RAW COUNT TERM-DOCUMENT MATRIX — WE APPLY log(N/(1+df)) WEIGHTING NEXT.
    TFIDF_WORK = np.zeros((NUM_DOCS, VOCAB_SIZE), dtype=np.float64)
    DOCUMENT_FREQUENCY_PER_TYPE = np.zeros(VOCAB_SIZE, dtype=np.float64)
    SORTED_DOC_IDS = sorted(DOCS.keys())

    for ROW_INDEX, DOC_KEY in enumerate(SORTED_DOC_IDS):
        TOKEN_LIST = DOCS[DOC_KEY]
        INDEX_SEQUENCE = MapTokensToIndices(TOKEN_LIST, WORD_TO_INDEX, TOP_WORD_SET)
        ROW_COUNTER = Counter(INDEX_SEQUENCE)
        for TYPE_INDEX, RAW_COUNT in ROW_COUNTER.items():
            TFIDF_WORK[ROW_INDEX, TYPE_INDEX] = float(RAW_COUNT)
        for TYPE_INDEX in ROW_COUNTER.keys():
            DOCUMENT_FREQUENCY_PER_TYPE[TYPE_INDEX] += 1.0

    # SPEC: TF-IDF(w,d) = TF(w,d) * log( N / (1 + df(w)) ) — HERE TF IS RAW COUNT.
    IDF_VECTOR = np.log(NUM_DOCS / (1.0 + DOCUMENT_FREQUENCY_PER_TYPE))
    TFIDF_WORK = TFIDF_WORK * IDF_VECTOR
    np.save("embeddings/tfidf_matrix.npy", TFIDF_WORK.astype(np.float32))

    with open("embeddings/word2idx.json", "w", encoding="utf-8") as JSON_OUT:
        json.dump(WORD_TO_INDEX, JSON_OUT, ensure_ascii=False)

    # GROUP ROW INDICES BY CATEGORY STRING PULLED FROM METADATA (PER-DOC TOPIC).
    CATEGORY_TO_ROW_INDICES = defaultdict(list)
    for DOC_KEY in SORTED_DOC_IDS:
        META_KEY = str(DOC_KEY)
        if META_KEY not in METADATA_OBJECT:
            continue
        CATEGORY_LABEL = METADATA_OBJECT[META_KEY].get("category", "general")
        CATEGORY_TO_ROW_INDICES[CATEGORY_LABEL].append(SORTED_DOC_IDS.index(DOC_KEY))

    print("top-10 discriminative words per category (mean tf-idf in category docs):")
    for CATEGORY_LABEL in sorted(CATEGORY_TO_ROW_INDICES.keys()):
        ROW_INDEX_LIST = CATEGORY_TO_ROW_INDICES[CATEGORY_LABEL]
        if not ROW_INDEX_LIST:
            continue
        MEAN_SCORE_PER_TYPE = TFIDF_WORK[ROW_INDEX_LIST].mean(axis=0)
        TOP_TEN_INDICES = np.argsort(-MEAN_SCORE_PER_TYPE)[:10]
        TOP_TEN_SURFACE_FORMS = [VOCAB_LIST[INDEX] for INDEX in TOP_TEN_INDICES]
        print(CATEGORY_LABEL + ":", " ".join(TOP_TEN_SURFACE_FORMS))

    # SYMMETRIC WINDOW CO-OCCURRENCE — WE ACCUMULATE ORDERED PAIRS THEN SYMMETRISE.
    CO_OCCURRENCE = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.float32)
    WINDOW_RADIUS = 5
    for DOC_KEY in SORTED_DOC_IDS:
        TOKEN_LIST = DOCS[DOC_KEY]
        INDEX_SEQUENCE = MapTokensToIndices(TOKEN_LIST, WORD_TO_INDEX, TOP_WORD_SET)
        SEQ_LEN = len(INDEX_SEQUENCE)
        for CENTER_POS in range(SEQ_LEN):
            WINDOW_LOW = max(0, CENTER_POS - WINDOW_RADIUS)
            WINDOW_HIGH = min(SEQ_LEN, CENTER_POS + WINDOW_RADIUS + 1)
            for CONTEXT_POS in range(WINDOW_LOW, WINDOW_HIGH):
                if CONTEXT_POS == CENTER_POS:
                    continue
                LEFT_TYPE = INDEX_SEQUENCE[CENTER_POS]
                RIGHT_TYPE = INDEX_SEQUENCE[CONTEXT_POS]
                CO_OCCURRENCE[LEFT_TYPE, RIGHT_TYPE] += 1.0

    CO_OCCURRENCE = (CO_OCCURRENCE + CO_OCCURRENCE.T) * np.float32(0.5)
    np.fill_diagonal(CO_OCCURRENCE, 0.0)

    TOTAL_CO_MASS = float(CO_OCCURRENCE.sum())
    if TOTAL_CO_MASS <= 0:
        TOTAL_CO_MASS = 1.0

    ROW_SUM_VECTOR = CO_OCCURRENCE.sum(axis=1)
    COL_SUM_VECTOR = CO_OCCURRENCE.sum(axis=0)
    EPSILON = np.float32(1e-12)
    DENOMINATOR_GRID = np.maximum(np.outer(ROW_SUM_VECTOR, COL_SUM_VECTOR), EPSILON)

    # IN-PLACE PMI THEN CLIP TO NON-NEGATIVE LOG2 — SAVES A FULL EXTRA 400MB ARRAY.
    np.multiply(CO_OCCURRENCE, np.float32(TOTAL_CO_MASS), out=CO_OCCURRENCE)
    np.divide(CO_OCCURRENCE, DENOMINATOR_GRID, out=CO_OCCURRENCE)
    del DENOMINATOR_GRID
    np.maximum(CO_OCCURRENCE, EPSILON, out=CO_OCCURRENCE)
    np.log2(CO_OCCURRENCE, out=CO_OCCURRENCE)
    np.maximum(CO_OCCURRENCE, np.float32(0.0), out=CO_OCCURRENCE)
    PPMI_MATRIX = CO_OCCURRENCE
    np.save("embeddings/ppmi_matrix.npy", PPMI_MATRIX)

    # FOR t-SNE WE TAKE THE 200 MOST FREQUENT TYPES (EXCLUDING RARE NOISE IF ANY).
    GLOBAL_TYPE_FREQ = Counter()
    for DOC_KEY in SORTED_DOC_IDS:
        GLOBAL_TYPE_FREQ.update(MapTokensToIndices(DOCS[DOC_KEY], WORD_TO_INDEX, TOP_WORD_SET))
    TOP_TWO_HUNDRED_INDICES = [TYPE_INDEX for TYPE_INDEX, _FREQ in GLOBAL_TYPE_FREQ.most_common(200)]
    SUBMATRIX_FOR_EMBED = np.ascontiguousarray(PPMI_MATRIX[TOP_TWO_HUNDRED_INDICES], dtype=np.float32)

    try:
        from sklearn.manifold import TSNE

        TSNE_MODEL = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
        EMBED_2D = TSNE_MODEL.fit_transform(SUBMATRIX_FOR_EMBED)
    except Exception:
        # FALLBACK IF sklearn IS MISSING OR TSNE FAILS: FIRST TWO SVD COMPONENTS.
        CENTERED = SUBMATRIX_FOR_EMBED - SUBMATRIX_FOR_EMBED.mean(axis=0)
        U_MATRIX, SINGULAR_VALUES, VT_MATRIX = np.linalg.svd(CENTERED, full_matrices=False)
        EMBED_2D = U_MATRIX[:, :2] * SINGULAR_VALUES[:2]

    # MAJORITY-VOTE CATEGORY PER TYPE AMONG ALL DOCS WHERE THAT TYPE FIRES.
    TYPE_TO_CATEGORY_COUNTS = {}
    for DOC_KEY in SORTED_DOC_IDS:
        CATEGORY_LABEL = METADATA_OBJECT.get(str(DOC_KEY), {}).get("category", "general")
        INDEX_SEQUENCE = MapTokensToIndices(DOCS[DOC_KEY], WORD_TO_INDEX, TOP_WORD_SET)
        for TYPE_INDEX in INDEX_SEQUENCE:
            TYPE_TO_CATEGORY_COUNTS.setdefault(TYPE_INDEX, Counter())[CATEGORY_LABEL] += 1

    PLOT_LABEL_PER_POINT = []
    for TYPE_INDEX in TOP_TWO_HUNDRED_INDICES:
        CAT_COUNTER = TYPE_TO_CATEGORY_COUNTS.get(TYPE_INDEX, Counter())
        if CAT_COUNTER:
            WINNING_CATEGORY = CAT_COUNTER.most_common(1)[0][0]
        else:
            WINNING_CATEGORY = "general"
        PLOT_LABEL_PER_POINT.append(WINNING_CATEGORY)

    UNIQUE_LEGEND_CATEGORIES = sorted(set(PLOT_LABEL_PER_POINT))
    FIGURE_HANDLE, AXES_HANDLE = plt.subplots(figsize=(10, 8))
    for CATEGORY_LABEL in UNIQUE_LEGEND_CATEGORIES:
        BOOLEAN_MASK = np.array([PLOT_LABEL_PER_POINT[POINT_I] == CATEGORY_LABEL for POINT_I in range(len(TOP_TWO_HUNDRED_INDICES))])
        AXES_HANDLE.scatter(EMBED_2D[BOOLEAN_MASK, 0], EMBED_2D[BOOLEAN_MASK, 1], s=14, alpha=0.75, label=CATEGORY_LABEL)
    AXES_HANDLE.legend(title="category")
    AXES_HANDLE.set_xlabel("t-SNE dimension 1")
    AXES_HANDLE.set_ylabel("t-SNE dimension 2")
    AXES_HANDLE.set_title("t-SNE of top-200 frequent tokens (PPMI rows)")
    plt.tight_layout()
    plt.savefig("embeddings/tsne_ppmi.png", dpi=150)
    plt.close(FIGURE_HANDLE)

    QUERY_WORD_LIST = ["پاکستان", "کرکٹ", "فلم", "دنیا", "میچ", "کھلاڑی", "وزیر", "سائنس", "اداکار", "حکومت"]
    print("top-5 cosine neighbours (PPMI rows, cosine to full vocab):")
    ROW_L2_NORMS = np.linalg.norm(PPMI_MATRIX, axis=1)
    INDEX_TO_SURFACE = {INDEX: WORD for WORD, INDEX in WORD_TO_INDEX.items()}

    for QUERY_SURFACE in QUERY_WORD_LIST:
        if QUERY_SURFACE not in WORD_TO_INDEX:
            print(QUERY_SURFACE + ": not in vocab")
            continue
        QUERY_INDEX = WORD_TO_INDEX[QUERY_SURFACE]
        DOT_PRODUCTS = PPMI_MATRIX @ PPMI_MATRIX[QUERY_INDEX]
        COSINE_SIMS = DOT_PRODUCTS / (ROW_L2_NORMS * ROW_L2_NORMS[QUERY_INDEX] + 1e-12)
        COSINE_SIMS[QUERY_INDEX] = -1.0
        NEIGHBOUR_INDICES = np.argsort(-COSINE_SIMS)[:5]
        NEIGHBOUR_STRINGS = [INDEX_TO_SURFACE[J] for J in NEIGHBOUR_INDICES]
        print(QUERY_SURFACE + ":", " ".join(NEIGHBOUR_STRINGS))


if __name__ == "__main__":
    Main()
