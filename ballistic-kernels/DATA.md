# Data

All experiments use the public **minimind** pretraining corpus (Chinese text).

## Option A — raw text (statistics side, `autoballistic` / `diag5` / `exp5` / `decay5` / `pair6`)

Download `pretrain_t2t_mini.jsonl` (~1.2 GB, one `{"text": ...}` document per line):

- HuggingFace: <https://huggingface.co/datasets/jingyaogong/minimind_dataset>
- ModelScope mirror also works.

Place it as `data/pretrain_t2t_mini.jsonl` (or anywhere) and point `--data` /
`DATA_JSONL` at it. Each JSONL line is one document; cross-document token pairs are
excluded from all statistics.

## Option B — tokenized cache (attention side, `train6.py --src sub`)

`make_sub.py` / `make_sub_big.py` parse a `torch.save`d token cache **without torch**
(`core5.load_pt`) and remap to the top-2047 token ids (UNK for the rest):

```bash
python src/make_sub_big.py        # expects the .pt path inside; edit if needed
# -> sub_stream_big.npy (int32 token stream), train/val split 95/5 inside data6.SubStream
```

The character-level path (`--src char`) builds its stream directly from the JSONL.

## Scale reference

| stream | tokens | used for |
|---|---|---|
| `sub_stream.npy` (30k samples) | 23.2M | 11.4M / 26.3M model runs |
| `sub_stream_big.npy` (400k samples) | 57.5M | 86.6M overnight validation |
| char stream (16M chars) | 16M | all statistics-side experiments |
