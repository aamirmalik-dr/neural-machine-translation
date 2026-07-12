"""Training loop and greedy decoding for the attention seq2seq model."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from nmt.data import DateTranslationData, Vocab, tensorize_batch
from nmt.model import Seq2Seq


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def greedy_decode(
    model: Seq2Seq,
    sources: list[str],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    max_len: int = 12,
    device: str = "cpu",
) -> tuple[list[str], torch.Tensor]:
    """Greedily decode target strings and return them with attention weights.

    Returns:
        A tuple ``(predictions, attention)`` where ``attention`` has shape
        ``(B, max_len, S)``.
    """
    model.eval()
    src_ids, _, _, _ = tensorize_batch(sources, sources, src_vocab, tgt_vocab)
    src_ids = src_ids.to(device)
    enc_outputs, state = model.encoder(src_ids)
    mask = src_ids != model.src_pad

    b = src_ids.shape[0]
    tokens = torch.full((b,), tgt_vocab.sos_id, dtype=torch.long, device=device)
    finished = torch.zeros(b, dtype=torch.bool, device=device)
    seqs: list[list[int]] = [[] for _ in range(b)]
    attns = []
    for _ in range(max_len):
        logits, state, weights = model.decoder.step(tokens, state, enc_outputs, mask)
        tokens = logits.argmax(dim=-1)
        attns.append(weights)
        for i in range(b):
            if not finished[i]:
                tok = int(tokens[i])
                if tok == tgt_vocab.eos_id:
                    finished[i] = True
                else:
                    seqs[i].append(tok)
        if bool(finished.all()):
            break
    attention = torch.stack(attns, dim=1)
    preds = [tgt_vocab.decode(s) for s in seqs]
    return preds, attention


@dataclass
class Trainer:
    """Trains the seq2seq model with teacher forcing and Adam."""

    model: Seq2Seq
    tgt_pad: int
    lr: float = 1e-3
    device: str = "cpu"
    history: dict[str, list[float]] = field(default_factory=lambda: {"loss": []})

    def fit(
        self,
        data: DateTranslationData,
        epochs: int = 8,
        batch_size: int = 128,
        verbose: bool = True,
    ) -> Trainer:
        self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        n = len(data)
        for epoch in range(epochs):
            self.model.train()
            perm = np.random.permutation(n)
            running, seen = 0.0, 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                src = [data.sources[i] for i in idx]
                tgt = [data.targets[i] for i in idx]
                src_ids, _, tgt_ids, _ = tensorize_batch(
                    src, tgt, data.src_vocab, data.tgt_vocab
                )
                src_ids, tgt_ids = src_ids.to(self.device), tgt_ids.to(self.device)
                logits, _ = self.model(src_ids, tgt_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_ids[:, 1:].reshape(-1),
                    ignore_index=self.tgt_pad,
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                running += loss.item() * len(idx)
                seen += len(idx)
            self.history["loss"].append(running / seen)
            if verbose:
                print(f"epoch {epoch + 1:3d}  loss={running / seen:.4f}")
        return self
