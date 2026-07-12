# Neural machine translation with attention

A sequence-to-sequence model with Bahdanau (additive) attention, implemented from
scratch in PyTorch: a bidirectional GRU encoder, an additive attention module, a
GRU decoder with teacher forcing, greedy decoding, a corpus BLEU metric, and
attention-alignment extraction.

It is demonstrated on a controlled translation task, converting human-written
dates in many formats into ISO 8601 (`YYYY-MM-DD`). This task is a real
sequence-to-sequence alignment problem (the model must copy digits, map month
names to numbers, and reorder fields), it trains to completion on a CPU, and its
attention maps are directly interpretable. The same architecture applies
unchanged to a natural-language parallel corpus such as Multi30k.

## What it does

- Generates parallel (human date, ISO date) pairs deterministically, with six
  input formats and character-level vocabularies (`data.py`).
- Encodes the source with a bidirectional GRU, attends over the encoder outputs
  with additive attention (masked over padding), and decodes with a GRU
  (`model.py`).
- Trains with teacher forcing and evaluates with exact-match accuracy and a
  character-level corpus BLEU (`train.py`, `metrics.py`).
- Extracts and plots the attention alignment for any example (`scripts/train.py`).

## What it does not do

- The headline demo is a controlled date task, chosen so the full attention
  pipeline is verifiable end to end on a CPU. It is not trained on a natural
  language corpus here, though the code is corpus-agnostic (see
  `scripts/download_data.py`).
- Decoding is greedy, not beam search.
- Tokenization is character level, not subword.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Run

```bash
python scripts/train.py --n 12000 --epochs 15
```

`notebooks/demo.ipynb` is a short executed walkthrough with an attention heatmap.

## Results

Trained on 12000 generated date pairs (1200 held out), 15 epochs, single CPU,
seed 0. Produced by `scripts/train.py` in this repository.

| Metric                    | Value |
|---------------------------|------:|
| Test exact-match accuracy | 1.0000 |
| Test BLEU (character)     | 1.0000 |

The model translates every held-out date correctly across all six input formats.
A sample of its output:

```
Jul 2, 1990          -> 1990-07-02
06.03.1978           -> 1978-03-06
28th of August 2012  -> 2012-08-28
February 22, 2006    -> 2006-02-22
```

The learned attention aligns each output digit with the corresponding characters
in the input (the year digits, the month name, the day), which is exactly the
behavior the additive attention mechanism is meant to produce. The alignment
heatmap is written to `results/attention.png`.

## Layout

```
src/nmt/        data, model (encoder, attention, decoder, seq2seq), train, metrics
scripts/        download_data.py (data note), train.py
notebooks/      demo.ipynb (executed)
tests/          pytest suite for data, model shapes, attention, metrics, training
data/           gitignored; see data/README.md
```

## Tests

```bash
pytest -q
ruff check src tests scripts
```

## License

MIT, see [LICENSE](LICENSE).

## Author

Aamir Malik. [GitHub](https://github.com/aamirmalik-dr) ·
[LinkedIn](https://linkedin.com/in/dr-aamirmalik)
