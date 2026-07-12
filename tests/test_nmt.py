import torch

from nmt.data import Vocab, make_date_dataset, tensorize_batch
from nmt.metrics import bleu, exact_match_accuracy
from nmt.model import Seq2Seq
from nmt.train import Trainer, greedy_decode, set_seed


def test_vocab_roundtrip():
    vocab = Vocab(list("abc123"))
    ids = vocab.encode("1a2", add_special=True)
    assert ids[0] == vocab.sos_id and ids[-1] == vocab.eos_id
    assert vocab.decode(ids) == "1a2"


def test_make_date_dataset_valid_iso():
    data = make_date_dataset(n=50, seed=0)
    assert len(data) == 50
    for tgt in data.targets:
        assert len(tgt) == 10 and tgt[4] == "-" and tgt[7] == "-"


def test_tensorize_shapes():
    data = make_date_dataset(n=8, seed=1)
    src_ids, src_len, tgt_ids, tgt_len = tensorize_batch(
        data.sources, data.targets, data.src_vocab, data.tgt_vocab
    )
    assert src_ids.shape[0] == 8
    assert tgt_ids.shape[0] == 8
    assert src_len.shape == (8,)


def test_model_forward_shapes():
    data = make_date_dataset(n=8, seed=2)
    model = Seq2Seq(len(data.src_vocab), len(data.tgt_vocab),
                    src_pad=data.src_vocab.pad_id, tgt_pad=data.tgt_vocab.pad_id)
    src_ids, _, tgt_ids, _ = tensorize_batch(
        data.sources, data.targets, data.src_vocab, data.tgt_vocab
    )
    logits, attn = model(src_ids, tgt_ids)
    assert logits.shape[0] == 8
    assert logits.shape[1] == tgt_ids.shape[1] - 1
    assert attn.shape[2] == src_ids.shape[1]


def test_attention_weights_sum_to_one():
    data = make_date_dataset(n=4, seed=3)
    model = Seq2Seq(len(data.src_vocab), len(data.tgt_vocab),
                    src_pad=data.src_vocab.pad_id, tgt_pad=data.tgt_vocab.pad_id)
    src_ids, _, tgt_ids, _ = tensorize_batch(
        data.sources, data.targets, data.src_vocab, data.tgt_vocab
    )
    _, attn = model(src_ids, tgt_ids)
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_bleu_and_accuracy_perfect():
    preds = ["2020-06-24", "1999-01-01"]
    assert exact_match_accuracy(preds, preds) == 1.0
    assert bleu(preds, preds) > 0.99


def test_trainer_learns_dates():
    set_seed(0)
    data = make_date_dataset(n=5000, seed=0)
    model = Seq2Seq(len(data.src_vocab), len(data.tgt_vocab), embed_dim=48, hidden_dim=96,
                    src_pad=data.src_vocab.pad_id, tgt_pad=data.tgt_vocab.pad_id)
    trainer = Trainer(model, tgt_pad=data.tgt_vocab.pad_id, lr=1e-3)
    trainer.fit(data, epochs=12, batch_size=128, verbose=False)
    # Loss should drop substantially as the model learns the mapping.
    assert trainer.history["loss"][-1] < 0.3 * trainer.history["loss"][0]
    # The model should translate a clear majority of held-in dates exactly. The
    # full run in scripts/train.py reaches 100 percent; this is a fast smoke test.
    preds, _ = greedy_decode(model, data.sources[:100], data.src_vocab, data.tgt_vocab)
    assert exact_match_accuracy(preds, data.targets[:100]) > 0.5
