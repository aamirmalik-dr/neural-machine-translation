"""A controlled date-translation dataset and character vocabularies.

The source language is human-written dates in a variety of formats; the target
language is the ISO 8601 form ``YYYY-MM-DD``. Both sides are tokenized at the
character level. Everything is generated deterministically from a seed, so the
demo is fully reproducible with no download.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


class Vocab:
    """A character vocabulary with the four special tokens reserved first."""

    def __init__(self, chars: list[str]) -> None:
        specials = [PAD, SOS, EOS, UNK]
        self.itos = specials + [c for c in chars if c not in specials]
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def sos_id(self) -> int:
        return self.stoi[SOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids = [self.stoi.get(c, self.stoi[UNK]) for c in text]
        if add_special:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            tok = self.itos[i] if 0 <= i < len(self.itos) else UNK
            if tok in {PAD, SOS}:
                continue
            if tok == EOS:
                break
            out.append(tok)
        return "".join(out)


def _ordinal(day: int) -> str:
    if 10 <= day % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def _human_date(year: int, month: int, day: int, rng: np.random.Generator) -> str:
    """Render a date in one of several human formats."""
    m_name = MONTHS[month - 1]
    fmt = int(rng.integers(0, 6))
    if fmt == 0:
        return f"{day} {m_name} {year}"
    if fmt == 1:
        return f"{m_name} {day}, {year}"
    if fmt == 2:
        return f"{_ordinal(day)} of {m_name} {year}"
    if fmt == 3:
        return f"{month:02d}/{day:02d}/{year}"
    if fmt == 4:
        return f"{m_name[:3]} {day}, {year}"
    return f"{day:02d}.{month:02d}.{year}"


@dataclass
class DateTranslationData:
    """Parallel source/target date strings with shared-scope vocabularies."""

    sources: list[str]
    targets: list[str]
    src_vocab: Vocab
    tgt_vocab: Vocab

    def __len__(self) -> int:
        return len(self.sources)


def make_date_dataset(n: int = 8000, seed: int = 0) -> DateTranslationData:
    """Generate ``n`` (human date, ISO date) pairs and build vocabularies."""
    rng = np.random.default_rng(seed)
    sources: list[str] = []
    targets: list[str] = []
    for _ in range(n):
        year = int(rng.integers(1970, 2035))
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 29))  # keep every month valid
        sources.append(_human_date(year, month, day, rng))
        targets.append(f"{year:04d}-{month:02d}-{day:02d}")

    src_chars = sorted({c for s in sources for c in s})
    tgt_chars = sorted({c for s in targets for c in s})
    return DateTranslationData(sources, targets, Vocab(src_chars), Vocab(tgt_chars))


def tensorize_batch(
    sources: list[str],
    targets: list[str],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode and pad a batch, returning source and target id tensors and lengths.

    Targets are wrapped with start and end tokens. Returns
    ``(src_ids, src_lengths, tgt_ids, tgt_lengths)``.
    """
    src_enc = [src_vocab.encode(s) for s in sources]
    tgt_enc = [tgt_vocab.encode(t, add_special=True) for t in targets]
    src_len = [len(s) for s in src_enc]
    tgt_len = [len(t) for t in tgt_enc]
    max_s, max_t = max(src_len), max(tgt_len)

    src_ids = np.full((len(sources), max_s), src_vocab.pad_id, dtype=np.int64)
    tgt_ids = np.full((len(targets), max_t), tgt_vocab.pad_id, dtype=np.int64)
    for i, (s, t) in enumerate(zip(src_enc, tgt_enc, strict=True)):
        src_ids[i, : len(s)] = s
        tgt_ids[i, : len(t)] = t
    return (
        torch.from_numpy(src_ids),
        torch.tensor(src_len, dtype=torch.long),
        torch.from_numpy(tgt_ids),
        torch.tensor(tgt_len, dtype=torch.long),
    )
