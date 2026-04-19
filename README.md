# CS-4063 NLP — Assignment 2 (reproduction)

Run from the repository root in order. Python 3.10+ and PyTorch with `matplotlib` and `numpy` are required.

## Data and embeddings

1. **`python a.py`** — corpus prep (writes cleaned text and related artifacts as configured in the script).
2. **`python b.py`** — TF-IDF / PPMI matrices under `embeddings/`.
3. **`python c.py`** — Word2Vec training (or embedding pipeline stage 1 per your project).
4. **`python d.py`** — Word2Vec evaluation and `embeddings/embeddings_w2v.npy`, `embeddings/word2idx.json`, and plots.

## Sequence labeling (POS / NER)

5. **`python e.py`** — sample sentences and CoNLL files under `data/*.conll`.
6. **`python f.py`** — trains BiLSTM POS and NER (frozen + finetuned runs), loss plots, and saves **`models/bilstm_pos.pt`**, **`models/bilstm_ner.pt`**.

## BiLSTM evaluation and ablations

7. **`python g.py`** — loads the two checkpoints, evaluates POS on `data/pos_test.conll` (token accuracy, macro-F1, **`models/pos_confusion_test.png`**), NER entity metrics (CRF vs argmax), FP/FN examples, retrains A1–A4 on the same 85/15 train/val split as `f.py`, reports test metrics, and writes **`models/metrics_summary.json`**.

## Transformer topic classifier

8. **`python i.py`** — builds stratified 70/15/15 topic splits from `Metadata.json` and `embeddings/word2idx.json`, saves `data/cls_* .npy` and `data/cls_meta.json`.
9. **`h.py`** — library only (encoder blocks); imported by `j.py`.
10. **`python j.py`** — trains `TransformerCls` for 20 epochs, saves **`models/transformer_cls.pt`**, **`models/transformer_loss_acc.png`**, **`models/transformer_confusion.png`**, and attention figures **`models/transformer_attn_article_*.png`**.

## Notebook

Open **`assignment2.ipynb`** and run all cells top-to-bottom after the scripts above (or use the notebook’s staged `subprocess` cells if you prefer a single place to drive the pipeline). The last markdown cell states the BiLSTM vs Transformer comparison using metrics loaded from disk.

## Checklist artifacts

| Path | Produced by |
|------|-------------|
| `embeddings/tfidf_matrix.npy`, `embeddings/ppmi_matrix.npy`, `embeddings/embeddings_w2v.npy`, `embeddings/word2idx.json` | `b.py` / `d.py` (per your pipeline) |
| `data/*.conll` | `e.py` |
| `models/bilstm_pos.pt`, `models/bilstm_ner.pt` | `f.py` |
| `models/transformer_cls.pt`, `data/cls_*.npy` | `j.py`, `i.py` |
| `models/pos_confusion_test.png`, `models/metrics_summary.json` | `g.py` |

If `python j.py` previously showed `nan` losses, pull the latest `h.py` (padding-safe attention) and re-run `j.py`.
