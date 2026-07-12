"""Translate dates with the committed pretrained model. No training, no network.

Loads the checkpoint at ``models/date_translator.pt``, normalizes any human date
strings passed on the command line into ISO 8601, and optionally regenerates the
attention-alignment heatmap.

Usage:
    python scripts/translate.py "March 3, 2001" "07.06.1994"
    python scripts/translate.py --heatmap "28th of August 2012"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nmt.checkpoint import load_checkpoint
from nmt.viz import plot_attention, translate

DEFAULT_EXAMPLES = [
    "March 3, 2001",
    "Jul 2, 1990",
    "06.03.1978",
    "28th of August 2012",
    "February 22, 2006",
    "12/25/2020",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dates", nargs="*", help="human date strings to normalize")
    parser.add_argument("--model", default="models/date_translator.pt")
    parser.add_argument("--heatmap", action="store_true", help="write results/attention.png")
    parser.add_argument("--out", default="results/attention.png")
    args = parser.parse_args()

    model, src_vocab, tgt_vocab = load_checkpoint(args.model)
    sources = args.dates or DEFAULT_EXAMPLES

    preds = translate(model, sources, src_vocab, tgt_vocab)
    width = max(len(s) for s in sources)
    for src, pred in zip(sources, preds, strict=True):
        print(f"  {src:<{width}}  ->  {pred}")

    if args.heatmap:
        out = Path(args.out)
        plot_attention(model, sources[0], src_vocab, tgt_vocab, out_path=out)
        print(f"\nWrote attention heatmap to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
