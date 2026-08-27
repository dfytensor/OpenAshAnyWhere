"""Streaming data module for the MiniMind Chinese corpus.

The cached ``.pt`` files contain MiniMind-tokenized ids but the matching
tokenizer is not available offline, so we cannot decode them.  Instead we
stream the raw ``pretrain_t2t_mini.jsonl`` (``{"text": "..."}`` per line) and
build a **character-level** tokenizer on the fly: Chinese is naturally
one-character-per-token, decoding is trivial, and the vocabulary stays small
enough (~4k) for an equilibrium-propagation model to be feasible.

Build the tokenizer once from a prefix of the file (character counts are an
excellent estimate of the full distribution), then stream training batches.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np
import torch


PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL = [PAD, BOS, EOS, UNK]


@dataclass
class CharTokenizerMM:
    itos: List[str] = field(default_factory=list)
    stoi: dict = field(default_factory=dict)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    def encode(self, text: str) -> List[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        pad = self.stoi[PAD]
        return "".join(self.itos[i] for i in ids if i != pad and i < len(self.itos))


def build_tokenizer(path: str, max_chars: int = 4500, read_mb: int = 60) -> CharTokenizerMM:
    """Count characters over the first ``read_mb`` MB and keep the top
    ``max_chars``."""
    from collections import Counter
    counts = Counter()
    target_bytes = read_mb * 1024 * 1024
    read = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            read += len(line.encode("utf-8"))
            try:
                obj = json.loads(line)
            except Exception:
                continue
            txt = obj.get("text", "")
            counts.update(txt)
            if read >= target_bytes:
                break
    # most frequent first, but always include special tokens
    top = [ch for ch, _ in counts.most_common(max_chars - len(SPECIAL))]
    itos = list(SPECIAL) + top
    stoi = {ch: i for i, ch in enumerate(itos)}
    tok = CharTokenizerMM(itos=itos, stoi=stoi)
    print(f"[tokenizer] read {read/1e6:.1f} MB, {len(counts)} unique chars, "
          f"vocab={tok.vocab_size}")
    return tok


def _read_texts(path: str) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get("text", "")
            if t:
                yield t


class StreamingBatcher:
    """Endless stream of fixed-length ``(x, y)`` next-token batches.

    Each raw text is encoded, appended with EOS, then chopped into non-overlapping
    ``seq_len + 1`` windows; consecutive windows are packed across documents to
    avoid wasting context.  ``x`` = tokens[:-1], ``y`` = tokens[1:].
    """

    def __init__(
        self,
        path: str,
        tokenizer: CharTokenizerMM,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        seed: int = 0,
    ):
        self.path = path
        self.tok = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.rng = np.random.default_rng(seed)
        # prime a buffer of token ids packed from documents
        self._buf: List[int] = []
        self._it = _read_texts(path)
        self._refill(target=seq_len * (batch_size + 4))

    def _refill(self, target: int):
        eos = self.tok.stoi[EOS]
        while len(self._buf) < target:
            try:
                txt = next(self._it)
            except StopIteration:
                # rewind file
                self._it = _read_texts(self.path)
                txt = next(self._it)
            self._buf.extend(self.tok.encode(txt))
            self._buf.append(eos)

    def __iter__(self) -> Iterator:
        win = self.seq_len + 1
        while True:
            while len(self._buf) < win * self.batch_size:
                self._refill(target=win * (self.batch_size + 4))
            # sample batch_size windows from the front region with small jitter
            xs = torch.empty(self.batch_size, self.seq_len, dtype=torch.long)
            ys = torch.empty(self.batch_size, self.seq_len, dtype=torch.long)
            for b in range(self.batch_size):
                start = self.rng.integers(0, max(1, len(self._buf) - win - 1))
                chunk = self._buf[start : start + win]
                xs[b] = torch.tensor(chunk[:self.seq_len], dtype=torch.long)
                ys[b] = torch.tensor(chunk[1 : self.seq_len + 1], dtype=torch.long)
            # trim consumed head occasionally to bound memory
            if len(self._buf) > win * 256:
                self._buf = self._buf[win * 64 :]
            yield xs.to(self.device), ys.to(self.device)
