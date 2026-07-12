"""Generate the committed sample of synthetic date pairs.

Writes a few hundred deterministic (human date, ISO date) pairs to
``data/sample_dates.csv``. The pairs are fully synthetic, produced by
``nmt.data.make_date_dataset``, so the sample is license clean and the quickstart
runs with no download.

Usage:
    python scripts/make_sample.py --n 300
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from nmt.data import make_date_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--out", default="data/sample_dates.csv")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data = make_date_dataset(n=args.n, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source", "target"])
        for src, tgt in zip(data.sources, data.targets, strict=True):
            writer.writerow([src, tgt])
    print(f"Wrote {len(data)} synthetic date pairs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
