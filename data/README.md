# Data

This directory is gitignored (this README aside). No datasets are committed.

The translation task is generated deterministically in code by
`nmt.data.make_date_dataset`, which produces parallel (human date, ISO date)
pairs from a seed. No download is needed and the demo is fully reproducible.

The same encoder-attention-decoder architecture applies to a natural-language
parallel corpus. `scripts/download_data.py` documents how to swap in a corpus
such as Multi30k; the model, training loop, and BLEU metric are unchanged.
