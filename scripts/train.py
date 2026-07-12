"""Train the attention seq2seq model on the date-normalization task.

Trains on generated (human date, ISO date) pairs, evaluates exact-match accuracy
and character-level BLEU on a held-out split, prints example translations, writes
a metrics file, saves the trained checkpoint, and renders the attention-alignment
heatmap that is the hero figure of this project.

Usage:
    python scripts/train.py --n 12000 --epochs 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nmt.checkpoint import save_checkpoint
from nmt.data import DateTranslationData, make_date_dataset
from nmt.metrics import bleu, exact_match_accuracy
from nmt.model import Seq2Seq
from nmt.train import Trainer, greedy_decode, set_seed
from nmt.viz import plot_attention


def _split(
    data: DateTranslationData, n_test: int
) -> tuple[DateTranslationData, DateTranslationData]:
    train = DateTranslationData(
        data.sources[:-n_test], data.targets[:-n_test], data.src_vocab, data.tgt_vocab
    )
    test = DateTranslationData(
        data.sources[-n_test:], data.targets[-n_test:], data.src_vocab, data.tgt_vocab
    )
    return train, test


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--out", default="results")
    parser.add_argument("--model-out", default="models/date_translator.pt")
    parser.add_argument("--example", default=None, help="source date for the heatmap")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(0)
    data = make_date_dataset(n=args.n, seed=0)
    train, test = _split(data, n_test=max(200, args.n // 10))
    print(
        f"train={len(train)}  test={len(test)}  "
        f"src_vocab={len(data.src_vocab)}  tgt_vocab={len(data.tgt_vocab)}"
    )

    model = Seq2Seq(
        len(data.src_vocab),
        len(data.tgt_vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        src_pad=data.src_vocab.pad_id,
        tgt_pad=data.tgt_vocab.pad_id,
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

    # Persist metrics.
    metrics = {
        "task": "date-normalization (synthetic, controlled)",
        "train_size": len(train),
        "test_size": len(test),
        "epochs": args.epochs,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "seed": 0,
        "test_exact_match_accuracy": round(acc, 4),
        "test_bleu_character": round(score, 4),
        "final_train_loss": round(trainer.history["loss"][-1], 4),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nWrote {out_dir / 'metrics.json'}")

    # Save the trained checkpoint for instant offline inference.
    ckpt = save_checkpoint(
        model,
        data.src_vocab,
        data.tgt_vocab,
        args.model_out,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    )
    print(f"Saved model to {ckpt}")

    # Hero figure: attention alignment for one example.
    example = args.example or test.sources[0]
    plot_attention(
        model, example, data.src_vocab, data.tgt_vocab, out_path=out_dir / "attention.png"
    )

    # Secondary: training loss curve.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(trainer.history["loss"]) + 1), trainer.history["loss"], marker="o")
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.title("Seq2seq training")
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=120)
    plt.close()

    print(f"Wrote figures to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
