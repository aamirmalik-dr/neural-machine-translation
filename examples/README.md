# Examples

Runnable, offline examples that use only the committed pretrained checkpoint at
`models/date_translator.pt`. No training, no download.

## translate_examples.py

Normalizes a handful of human-written dates into ISO 8601, prints the
attention-argmax alignment for one example (which source character each output
character attends to most), and writes an attention heatmap to
`examples/example_attention.png`.

```bash
python examples/translate_examples.py
```

Expected output begins:

```
Date normalization (human format -> ISO 8601)

  March 3, 2001        ->  2001-03-03
  Jul 2, 1990          ->  1990-07-02
  06.03.1978           ->  1978-03-06
  ...
```

## Command-line translation

The same model is exposed as a small CLI:

```bash
python scripts/translate.py "March 3, 2001" "07.06.1994"
python scripts/translate.py --heatmap "28th of August 2012"
```
