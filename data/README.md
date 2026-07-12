# Data

The translation task is fully synthetic. Parallel (human date, ISO date) pairs
are generated deterministically in code by `nmt.data.make_date_dataset`, which
draws a year, month, and day from fixed ranges and renders the human side in one
of six formats. No dataset is downloaded and the demo is fully reproducible.

## Committed sample

`data/sample_dates.csv` holds 300 synthetic pairs (seed 7) written by
`scripts/make_sample.py`. It is committed only so the format is visible at a
glance and the repository is browsable with no run. It is not a carved subset of
any external corpus, it is generated, and it is license clean. Regenerate it with:

```bash
python scripts/make_sample.py --n 300
```

Everything else under `data/` is gitignored.

## Scaling to a real corpus

The same encoder-attention-decoder architecture applies to a natural-language
parallel corpus. `scripts/download_data.py` documents how to swap in a corpus
such as Multi30k: load its source and target sentence pairs, build character or
subword vocabularies the same way, and pass them through the model. The training
loop and the BLEU metric are unchanged.
