"""Evaluation metrics: exact-match accuracy and corpus BLEU."""

from __future__ import annotations

import math
from collections import Counter


def exact_match_accuracy(predictions: list[str], references: list[str]) -> float:
    """Fraction of predictions that exactly equal their reference string."""
    if not predictions:
        return 0.0
    correct = sum(int(p == r) for p, r in zip(predictions, references, strict=True))
    return correct / len(predictions)


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu(predictions: list[str], references: list[str], max_n: int = 4) -> float:
    """Corpus BLEU over character tokens with add-one smoothing.

    Tokens are the individual characters of each string, which suits the
    character-level date-translation task. Returns a score in ``[0, 1]``.
    """
    weights = [1.0 / max_n] * max_n
    p_log_sum = 0.0
    pred_len_total = 0
    ref_len_total = 0

    clipped = [0] * max_n
    totals = [0] * max_n
    for pred, ref in zip(predictions, references, strict=True):
        p_tokens = list(pred)
        r_tokens = list(ref)
        pred_len_total += len(p_tokens)
        ref_len_total += len(r_tokens)
        for n in range(1, max_n + 1):
            p_counts = _ngram_counts(p_tokens, n)
            r_counts = _ngram_counts(r_tokens, n)
            overlap = sum(min(c, r_counts[g]) for g, c in p_counts.items())
            clipped[n - 1] += overlap
            totals[n - 1] += max(sum(p_counts.values()), 0)

    for n in range(max_n):
        # Add-one smoothing keeps the score defined when a higher-order n-gram
        # never matches.
        precision = (clipped[n] + 1) / (totals[n] + 1)
        p_log_sum += weights[n] * math.log(precision)

    brevity = 1.0 if pred_len_total > ref_len_total else math.exp(
        1 - ref_len_total / max(pred_len_total, 1)
    )
    return brevity * math.exp(p_log_sum)
