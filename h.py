"""
MODULE H — TRANSFORMER ENCODER BUILDING BLOCKS FROM SCRATCH (NO nn.Transformer / nn.MultiheadAttention).
USED BY j.py FOR TOPIC CLASSIFICATION: d_model=128, 4 HEADS, PRE-LN ENCODER x4, CLS + MLP HEAD.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, D_K):
        super().__init__()
        self.SCALE = 1.0 / math.sqrt(float(D_K))

    def forward(self, Q, K, V, ATTN_BLOCK_MASK=None, TOKEN_PAD_MASK=None):
        SCORES = torch.matmul(Q, K.transpose(-2, -1)) * self.SCALE
        if ATTN_BLOCK_MASK is not None:
            SCORES = SCORES.masked_fill(ATTN_BLOCK_MASK, float("-inf"))
        ATTN = F.softmax(SCORES, dim=-1)
        ATTN = torch.nan_to_num(ATTN, nan=0.0, posinf=0.0, neginf=0.0)
        OUT = torch.matmul(ATTN, V)
        if TOKEN_PAD_MASK is not None:
            QM = (~TOKEN_PAD_MASK).unsqueeze(-1).to(OUT.dtype)
            OUT = OUT * QM
        return OUT, ATTN


class MultiHeadAttention(nn.Module):
    def __init__(self, D_MODEL, NUM_HEADS, D_K, D_V, DROPOUT=0.1):
        super().__init__()
        assert D_MODEL == NUM_HEADS * D_K
        self.H = NUM_HEADS
        self.DK = D_K
        self.DV = D_V
        self.WQ = nn.ModuleList([nn.Linear(D_MODEL, D_K, bias=False) for _ in range(NUM_HEADS)])
        self.WK = nn.ModuleList([nn.Linear(D_MODEL, D_K, bias=False) for _ in range(NUM_HEADS)])
        self.WV = nn.ModuleList([nn.Linear(D_MODEL, D_V, bias=False) for _ in range(NUM_HEADS)])
        self.WO = nn.Linear(NUM_HEADS * D_V, D_MODEL, bias=False)
        self.ATTN_CORE = ScaledDotProductAttention(D_K)
        self.DROP = nn.Dropout(DROPOUT)

    def forward(self, X, PAD_MASK=None):
        B, L, _ = X.shape
        HEAD_OUTS = []
        HEAD_ATTNS = []
        ATTN_BLOCK_MASK = None
        if PAD_MASK is not None:
            MQ = PAD_MASK.unsqueeze(2)
            MK = PAD_MASK.unsqueeze(1)
            ATTN_BLOCK_MASK = MQ | MK
        for HI in range(self.H):
            QH = self.WQ[HI](X)
            KH = self.WK[HI](X)
            VH = self.WV[HI](X)
            OH, AH = self.ATTN_CORE(QH, KH, VH, ATTN_BLOCK_MASK, PAD_MASK)
            HEAD_OUTS.append(OH)
            HEAD_ATTNS.append(AH)
        CAT = torch.cat(HEAD_OUTS, dim=-1)
        OUT = self.WO(CAT)
        OUT = self.DROP(OUT)
        ATTN_STACK = torch.stack(HEAD_ATTNS, dim=1)
        return OUT, ATTN_STACK


class FeedForward(nn.Module):
    def __init__(self, D_MODEL, D_FF, DROPOUT=0.1):
        super().__init__()
        self.L1 = nn.Linear(D_MODEL, D_FF)
        self.L2 = nn.Linear(D_FF, D_MODEL)
        self.DROP = nn.Dropout(DROPOUT)

    def forward(self, X):
        return self.L2(self.DROP(F.relu(self.L1(X))))


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, D_MODEL, MAX_LEN=512):
        super().__init__()
        PE = torch.zeros(MAX_LEN, D_MODEL)
        POS = torch.arange(0, MAX_LEN, dtype=torch.float).unsqueeze(1)
        DIV = torch.exp(torch.arange(0, D_MODEL, 2).float() * (-math.log(10000.0) / float(D_MODEL)))
        PE[:, 0::2] = torch.sin(POS * DIV)
        PE[:, 1::2] = torch.cos(POS * DIV)
        self.register_buffer("PE_BUF", PE.unsqueeze(0))

    def forward(self, X):
        L = X.size(1)
        return X + self.PE_BUF[:, :L, :].to(X.dtype)


class EncoderBlock(nn.Module):
    def __init__(self, D_MODEL, NUM_HEADS, D_K, D_V, D_FF, DROPOUT=0.1):
        super().__init__()
        self.LN1 = nn.LayerNorm(D_MODEL)
        self.MHA = MultiHeadAttention(D_MODEL, NUM_HEADS, D_K, D_V, DROPOUT)
        self.LN2 = nn.LayerNorm(D_MODEL)
        self.FFN = FeedForward(D_MODEL, D_FF, DROPOUT)
        self.DROP = nn.Dropout(DROPOUT)

    def forward(self, X, PAD_MASK=None, RETURN_ATTN=False):
        H = self.LN1(X)
        MH_OUT, ATTN = self.MHA(H, PAD_MASK)
        X = X + self.DROP(MH_OUT)
        H2 = self.LN2(X)
        X = X + self.DROP(self.FFN(H2))
        if RETURN_ATTN:
            return X, ATTN
        return X, None


class TransformerCls(nn.Module):
    """
    STACKS FOUR PRE-LN ENCODER BLOCKS; PREPENDS LEARNED CLS; FINAL CLS -> MLP -> NUM_CLASSES.
    """

    def __init__(self, VOC_SIZE, PAD_IDX, NUM_CLASSES, D_MODEL=128, NUM_HEADS=4, D_K=32, D_V=32, D_FF=512, MAX_LEN=256, DROPOUT=0.1):
        super().__init__()
        self.D_MODEL = D_MODEL
        self.PAD_IDX = PAD_IDX
        self.EMB = nn.Embedding(VOC_SIZE, D_MODEL, padding_idx=PAD_IDX)
        self.CLS = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        self.SINPE = SinusoidalPositionalEncoding(D_MODEL, MAX_LEN + 4)
        self.BLOCKS = nn.ModuleList(
            [EncoderBlock(D_MODEL, NUM_HEADS, D_K, D_V, D_FF, DROPOUT) for _ in range(4)]
        )
        self.MLP = nn.Sequential(nn.Linear(D_MODEL, 64), nn.ReLU(), nn.Linear(64, NUM_CLASSES))

    def forward(self, TOKEN_IDS, RETURN_LAST_BLOCK_ATTN=False):
        B, L = TOKEN_IDS.shape
        TOK_PAD = TOKEN_IDS == self.PAD_IDX
        X = self.EMB(TOKEN_IDS) * math.sqrt(float(self.D_MODEL))
        CLS_TILE = self.CLS.expand(B, -1, -1)
        X = torch.cat([CLS_TILE, X], dim=1)
        CLS_PAD = torch.zeros((B, L + 1), dtype=torch.bool, device=TOKEN_IDS.device)
        CLS_PAD[:, 1:] = TOK_PAD
        X = self.SINPE(X)
        LAST_ATTN = None
        for BI, BLK in enumerate(self.BLOCKS):
            RA = RETURN_LAST_BLOCK_ATTN and (BI == len(self.BLOCKS) - 1)
            X, AT = BLK(X, CLS_PAD, RETURN_ATTN=RA)
            if RA:
                LAST_ATTN = AT
        CLS_OUT = X[:, 0, :]
        LOGITS = self.MLP(CLS_OUT)
        if RETURN_LAST_BLOCK_ATTN:
            return LOGITS, LAST_ATTN
        return LOGITS, None
