"""Train the attention seq2seq model on the date-translation task.

Trains on generated (human date, ISO date) pairs, evaluates exact-match accuracy
and BLEU on a held-out split, prints example translations, and saves an
attention-alignment heatmap.

Usage:
    python scripts/train.py --n 8000 --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nmt.data import DateTranslationData, make_date_dataset
from nmt.metrics import bleu, exact_match_accuracy
from nmt.model import Seq2Seq
from nmt.train import Trainer, greedy_decode, set_seed


def _split(data: DateTranslationData, n_test: int) -> tuple[DateTranslationData, DateTranslationData]:
    train = DateTranslationData(
        data.sources[:-n_test], data.targets[:-n_test], data.src_vocab, data.tgt_vocab
    )
    test = DateTranslationData(
        data.sources[-n_test:], data.targets[-n_test:], data.src_vocab, data.tgt_vocab
    )
    return train, test


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(0)
    data = make_date_dataset(n=args.n, seed=0)
    train, test = _split(data, n_test=max(200, args.n // 10))
    print(f"train={len(train)}  test={len(test)}  "
          f"src_vocab={len(data.src_vocab)}  tgt_vocab={len(data.tgt_vocab)}")

    model = Seq2Seq(
        len(data.src_vocab), len(data.tgt_vocab),
        src_pad=data.src_vocab.pad_id, tgt_pad=data.tgt_vocab.pad_id,
    )
    trainer = Trainer(model, tgt_pad=data.tgt_vocab.pad_id, lr=1e-3)
    trainer.fit(train, epochs=args.epochs)

    preds, _ = greedy_decode(model, test.sources, data.src_vocab, data.tgt_vocab)
    acc = exact_match_accuracy(preds, test.targets)
    score = bleu(preds, test.targets)
    print(f"\nTest exact-match accuracy: {acc:.4f}")
    print(f"Test BLEU (character): {score:.4f}")

    print("\nExample translations:")
    for src, pred, ref in list(zip(test.sources, preds, test.targets, strict=True))[:8]:
        flag = "ok" if pred == ref else "X"
        print(f"  [{flag}] {src:<28} -> {pred:<12} (gold {ref})")

    # Attention heatmap for one example.
    example = [test.sources[0]]
    _, attn = greedy_decode(model, example, data.src_vocab, data.tgt_vocab)
    pred0 = preds[0]
    a = attn[0, : len(pred0), : len(example[0])].cpu().numpy()
    plt.figure(figsize=(6, 4))
    plt.imshow(a, aspect="auto", cmap="viridis")
    plt.xticks(range(len(example[0])), list(example[0]), rotation=90)
    plt.yticks(range(len(pred0)), list(pred0))
    plt.xlabel("source")
    plt.ylabel("prediction")
    plt.title("Attention alignment")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / "attention.png", dpi=120)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(trainer.history["loss"]) + 1), trainer.history["loss"], marker="o")
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.title("Seq2seq training")
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=120)
    plt.close()

    print(f"\nWrote figures to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
