"""Save and load a trained model together with its vocabularies.

A checkpoint is a single ``.pt`` file that stores the model weights, the model
hyperparameters needed to rebuild the architecture, and the source and target
character vocabularies. Loading a checkpoint reconstructs a ready-to-run model
with no training and no dataset generation, so inference and the attention
heatmap render instantly and offline.
"""

from __future__ import annotations

from pathlib import Path

import torch

from nmt.data import Vocab
from nmt.model import Seq2Seq


def save_checkpoint(
    model: Seq2Seq,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    path: str | Path,
    embed_dim: int,
    hidden_dim: int,
) -> Path:
    """Write the model weights, hyperparameters, and vocabularies to ``path``.

    Args:
        model: The trained sequence-to-sequence model.
        src_vocab: Source character vocabulary.
        tgt_vocab: Target character vocabulary.
        path: Destination ``.pt`` file.
        embed_dim: Embedding dimension the model was built with.
        hidden_dim: Hidden dimension the model was built with.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "src_itos": src_vocab.itos,
        "tgt_itos": tgt_vocab.itos,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
    }
    torch.save(payload, path)
    return path


def _vocab_from_itos(itos: list[str]) -> Vocab:
    """Rebuild a ``Vocab`` from a stored index-to-string list."""
    vocab = Vocab([])
    vocab.itos = list(itos)
    vocab.stoi = {c: i for i, c in enumerate(vocab.itos)}
    return vocab


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[Seq2Seq, Vocab, Vocab]:
    """Rebuild a model and its vocabularies from a checkpoint file.

    Args:
        path: A ``.pt`` file written by :func:`save_checkpoint`.
        device: Device to map the weights onto.

    Returns:
        A tuple ``(model, src_vocab, tgt_vocab)`` with the model in eval mode.
    """
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    src_vocab = _vocab_from_itos(payload["src_itos"])
    tgt_vocab = _vocab_from_itos(payload["tgt_itos"])
    model = Seq2Seq(
        len(src_vocab),
        len(tgt_vocab),
        embed_dim=payload["embed_dim"],
        hidden_dim=payload["hidden_dim"],
        src_pad=src_vocab.pad_id,
        tgt_pad=tgt_vocab.pad_id,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, src_vocab, tgt_vocab
