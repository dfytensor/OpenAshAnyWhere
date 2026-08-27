"""Character-level tokenizer and tiny text corpus for EnergyLM.

EnergyLM is trained as a character-level language model. We keep the corpus
small and self-contained so the equilibrium-propagation dynamics can be studied
without a heavy data pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# A small, public-domain-flavored toy corpus.  The text is intentionally short
# and highly repetitive so that a character-level model can show clear learning
# within a few minutes of training.  Replace with any .txt file via
# ``CharTokenizer.from_text_file``.
# ---------------------------------------------------------------------------
DEFAULT_CORPUS = """\
the sun rises in the east and sets in the west.
the river flows from the mountain to the sea.
the wind blows across the open field at dawn.
the stars shine above the quiet town at night.
the cat sat on the warm mat by the window.
the dog ran through the green park in the rain.
the child laughed as the kite climbed the sky.
the old man told a story of the distant shore.
the baker baked fresh bread in the early morning.
the sailor sailed his ship across the deep ocean.
time flows like a river under the old stone bridge.
the garden grew bright flowers every passing spring.
the clock on the wall struck the hour of the night.
the teacher wrote a lesson on the black board.
the farmer planted seeds in the rich dark soil.
music filled the hall as the choir began to sing.
the train rolled slowly out of the misty station.
the moon rose silver over the calm still lake.
the forest stood silent beneath the falling snow.
the city woke to the sound of the morning bells.
she opened the book and read the first line again.
the waves crashed against the rocks of the shore.
a soft light fell across the pages of the old book.
the path wound up the hill to the small stone house.
the river bent where the old willow tree once stood.
every morning the birds sang the same gentle song.
the children played a game of hide and seek outside.
the captain steered the vessel through the narrow channel.
the artist painted the scene with careful, steady strokes.
the clock ticked softly while the fire burned low.
the answer was hidden in the last chapter of the tale.
"""


@dataclass
class CharTokenizer:
    """Minimal character-level tokenizer.

    ``stoi`` maps a character to an integer id, ``itos`` the reverse.  The
    vocabulary is derived from a corpus so it only contains the characters
    that actually appear.
    """

    stoi: dict = field(default_factory=dict)
    itos: list = field(default_factory=list)

    # ----- construction -------------------------------------------------
    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = chars
        return cls(stoi=stoi, itos=itos)

    @classmethod
    def from_text_file(cls, path: str) -> Tuple["CharTokenizer", str]:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls.from_text(text), text

    # ----- properties ---------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    # ----- encode / decode ---------------------------------------------
    def encode(self, text: str) -> List[int]:
        unk = self.stoi.get(" ", 0)
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)


def make_sequences(
    text: str,
    tokenizer: CharTokenizer,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode the whole corpus once into a 1-D LongTensor of token ids."""
    ids = tokenizer.encode(text)
    return torch.tensor(ids, dtype=torch.long, device=device)


def batch_iter(
    data: torch.Tensor,
    seq_len: int,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
):
    """Yield ``(x, y)`` batches for next-token prediction.

    ``x`` are the input ids (tokens ``t`` .. ``t+seq_len-1``) and ``y`` are the
    targets (tokens ``t+1`` .. ``t+seq_len``).  Start indices are sampled
    uniformly at random from the valid range.
    """
    n = data.numel()
    max_start = n - seq_len - 1
    if max_start <= 0:
        raise ValueError("corpus too short for the requested sequence length")

    while True:
        starts = rng.integers(0, max_start, size=batch_size)
        x = torch.stack([data[s : s + seq_len] for s in starts])
        y = torch.stack([data[s + 1 : s + 1 + seq_len] for s in starts])
        yield x.to(device), y.to(device)


def save_corpus(path: str, text: str = DEFAULT_CORPUS) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
