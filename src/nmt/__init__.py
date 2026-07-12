"""Sequence-to-sequence translation with attention, from scratch in PyTorch.

The package implements a bidirectional GRU encoder, Bahdanau (additive)
attention, and a GRU decoder with teacher forcing, plus greedy decoding, a
corpus BLEU metric, and attention-alignment extraction. It is demonstrated on a
controlled date-normalization translation task (varied human date formats to
ISO 8601), which is fully reproducible on a CPU and yields interpretable
attention maps. The same architecture applies to natural-language parallel
corpora such as Multi30k.
"""

from nmt.data import DateTranslationData, Vocab, make_date_dataset, tensorize_batch
from nmt.metrics import bleu, exact_match_accuracy
from nmt.model import BahdanauAttention, Decoder, Encoder, Seq2Seq
from nmt.train import Trainer, greedy_decode, set_seed

__all__ = [
    "DateTranslationData",
    "Vocab",
    "make_date_dataset",
    "tensorize_batch",
    "bleu",
    "exact_match_accuracy",
    "BahdanauAttention",
    "Decoder",
    "Encoder",
    "Seq2Seq",
    "Trainer",
    "greedy_decode",
    "set_seed",
]

__version__ = "0.1.0"
