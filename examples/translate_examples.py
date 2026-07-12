"""Runnable examples: load the committed model and normalize dates offline.

This uses only the pretrained checkpoint. It does no training and needs no
network. Run it from the repository root:

    python examples/translate_examples.py
"""

from __future__ import annotations

from pathlib import Path

from nmt.checkpoint import load_checkpoint
from nmt.viz import alignment_matrix, plot_attention, translate

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "date_translator.pt"

EXAMPLES = [
    "March 3, 2001",
    "Jul 2, 1990",
    "06.03.1978",
    "28th of August 2012",
    "February 22, 2006",
    "12/25/2020",
    "9 September 1999",
    "01.01.2000",
]


def main() -> int:
    model, src_vocab, tgt_vocab = load_checkpoint(MODEL_PATH)

    print("Date normalization (human format -> ISO 8601)\n")
    preds = translate(model, EXAMPLES, src_vocab, tgt_vocab)
    width = max(len(s) for s in EXAMPLES)
    for src, pred in zip(EXAMPLES, preds, strict=True):
        print(f"  {src:<{width}}  ->  {pred}")

    # Show the attention alignment for one example as plain numbers.
    source = "28th of August 2012"
    pred, weights = alignment_matrix(model, source, src_vocab, tgt_vocab)
    print(f'\nAttention argmax for "{source}" -> "{pred}":')
    for i, ch in enumerate(pred):
        j = int(weights[i].argmax())
        print(f"  output '{ch}' attends most to source '{source[j]}' (position {j})")

    # Write a heatmap next to this script.
    out = Path(__file__).resolve().parent / "example_attention.png"
    plot_attention(model, source, src_vocab, tgt_vocab, out_path=out)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
