"""Plotting helpers for attention alignment, shared by scripts and the notebook."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nmt.data import Vocab
from nmt.model import Seq2Seq
from nmt.train import greedy_decode


def translate(
    model: Seq2Seq,
    sources: list[str],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: str = "cpu",
) -> list[str]:
    """Translate a list of source date strings into ISO 8601 form.

    Args:
        model: A trained sequence-to-sequence model.
        sources: Human-written date strings.
        src_vocab: Source character vocabulary.
        tgt_vocab: Target character vocabulary.
        device: Device to run on.

    Returns:
        One predicted ISO date string per source.
    """
    preds, _ = greedy_decode(model, sources, src_vocab, tgt_vocab, device=device)
    return preds


def alignment_matrix(
    model: Seq2Seq,
    source: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: str = "cpu",
) -> tuple[str, np.ndarray]:
    """Decode one source and return its prediction and attention matrix.

    Returns:
        A tuple ``(prediction, weights)`` where ``weights`` has shape
        ``(len(prediction), len(source))`` and each row sums to one over the
        source positions the model attended to.
    """
    preds, attn = greedy_decode(model, [source], src_vocab, tgt_vocab, device=device)
    pred = preds[0]
    weights = attn[0, : len(pred), : len(source)]
    return pred, weights.cpu().numpy()


def plot_attention(
    model: Seq2Seq,
    source: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    out_path: str | Path | None = None,
    device: str = "cpu",
):
    """Render the source-to-target attention alignment as a heatmap.

    The source characters run along the x axis and the generated output
    characters run down the y axis. Bright cells mark, for each output
    character, which input characters the decoder attended to when producing it.

    Args:
        model: A trained model.
        source: A single human-written date string.
        src_vocab: Source character vocabulary.
        tgt_vocab: Target character vocabulary.
        out_path: If given, the figure is saved here.
        device: Device to run on.

    Returns:
        The matplotlib ``Figure``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred, weights = alignment_matrix(model, source, src_vocab, tgt_vocab, device=device)

    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(source)), 4.2))
    im = ax.imshow(weights, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(source)))
    ax.set_xticklabels(list(source), rotation=0)
    ax.set_yticks(range(len(pred)))
    ax.set_yticklabels(list(pred))
    ax.set_xlabel("source characters")
    ax.set_ylabel("generated ISO date")
    ax.set_title(f'Attention alignment:  "{source}"  ->  "{pred}"')
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=130)
    return fig
