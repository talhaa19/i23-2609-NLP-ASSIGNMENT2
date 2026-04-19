"""
ENTRY SCRIPT A — CORPUS SANITY CHECKS BEFORE ANY HEAVY PIPELINE STUFF RUNS.
RENAME POLICY: SINGLE-LETTER MODULE NAMES (a.py, b.py, c.py, …) FROM HERE ON.
"""

import os
import re
import sys


def ConfigureStdoutUtf8():
    """
    WINDOWS CONSOLES OFTEN DEFAULT TO A LIMITED CODE PAGE; FORCE UTF-8 SO URDU
    PRINTS DO NOT EXPLODE WHEN STUDENTS RUN THIS FROM POWERSHELL OR CMD.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def CountBracketDocuments(CORPUS_TEXT):
    """
    EACH ARTICLE IN cleaned.txt STARTS WITH A LINE LIKE [17] ON ITS OWN ROW.
    WE COUNT THOSE MARKERS SO "DOC COUNT" MATCHES THE NUMBER OF LOGICAL DOCS,
    NOT THE RAW PHYSICAL LINE COUNT (WHICH WOULD BE MISLEADING).
    """
    PATTERN = r"^\[(\d+)\]\s*$"
    MATCHES = re.findall(PATTERN, CORPUS_TEXT, flags=re.MULTILINE)
    return len(MATCHES)


def Main():
    """
    TOP-LEVEL DRIVER: VERIFY INPUT FILES EXIST, THEN PRINT THREE CORPUS STATS.
    """
    ConfigureStdoutUtf8()

    # LIST OF FILENAMES THE REST OF THE ASSIGNMENT EXPECTS IN THE REPO ROOT.
    REQUIRED_FILES = ["cleaned.txt", "raw.txt", "Metadata.json"]
    for FILE_NAME in REQUIRED_FILES:
        if not os.path.isfile(FILE_NAME):
            raise SystemExit("MISSING REQUIRED INPUT FILE: " + FILE_NAME)

    # SLURP THE WHOLE CLEANED CORPUS INTO MEMORY — CORPUS IS ONLY A FEW MB.
    with open("cleaned.txt", encoding="utf-8") as CORPUS_FILE_HANDLE:
        FULL_CORPUS_STRING = CORPUS_FILE_HANDLE.read()

    # TOKENIZATION STRATEGY HERE IS WHITESPACE SPLIT TO STAY CONSISTENT WITH B.
    ALL_TOKENS_LIST = FULL_CORPUS_STRING.split()
    UNIQUE_TOKEN_SET = set(ALL_TOKENS_LIST)

    DOCUMENT_COUNT = CountBracketDocuments(FULL_CORPUS_STRING)
    TOTAL_TOKEN_COUNT = len(ALL_TOKENS_LIST)
    VOCABULARY_SIZE = len(UNIQUE_TOKEN_SET)

    print("doc count:", DOCUMENT_COUNT)
    print("token count:", TOTAL_TOKEN_COUNT)
    print("vocab size:", VOCABULARY_SIZE)


if __name__ == "__main__":
    Main()
