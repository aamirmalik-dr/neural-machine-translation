"""Encoder, Bahdanau attention, decoder, and the full sequence-to-sequence model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """A bidirectional GRU encoder over the source sequence."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_id: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return encoder outputs ``(B, S, 2H)`` and an initial decoder state ``(B, H)``."""
        embedded = self.embedding(src)
        outputs, h_n = self.rnn(embedded)
        # Combine the final forward and backward hidden states.
        last = torch.cat([h_n[0], h_n[1]], dim=-1)
        dec_init = torch.tanh(self.bridge(last))
        return outputs, dec_init


class BahdanauAttention(nn.Module):
    """Additive attention scoring encoder outputs against the decoder state."""

    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self, dec_state: torch.Tensor, enc_outputs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a context vector ``(B, enc_dim)`` and attention weights ``(B, S)``."""
        # dec_state: (B, dec_dim) -> (B, 1, attn_dim); enc_outputs: (B, S, enc_dim)
        scores = self.v(torch.tanh(self.W_enc(enc_outputs) + self.W_dec(dec_state).unsqueeze(1)))
        scores = scores.squeeze(-1)  # (B, S)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights


class Decoder(nn.Module):
    """A GRU decoder that attends over the encoder outputs at each step."""

    def __init__(
        self, vocab_size: int, embed_dim: int, hidden_dim: int, enc_dim: int, pad_id: int
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.attention = BahdanauAttention(enc_dim, hidden_dim, hidden_dim)
        self.rnn = nn.GRUCell(embed_dim + enc_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim + enc_dim, vocab_size)

    def step(
        self,
        token: torch.Tensor,
        state: torch.Tensor,
        enc_outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one decoding step; return logits, new state, and attention weights."""
        embedded = self.embedding(token)  # (B, E)
        context, weights = self.attention(state, enc_outputs, mask)
        state = self.rnn(torch.cat([embedded, context], dim=-1), state)
        logits = self.out(torch.cat([state, context], dim=-1))
        return logits, state, weights


class Seq2Seq(nn.Module):
    """Full attention sequence-to-sequence model with teacher-forced training."""

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        src_pad: int = 0,
        tgt_pad: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(src_vocab, embed_dim, hidden_dim, src_pad)
        self.decoder = Decoder(tgt_vocab, embed_dim, hidden_dim, 2 * hidden_dim, tgt_pad)
        self.src_pad = src_pad

    def _mask(self, src: torch.Tensor) -> torch.Tensor:
        return src != self.src_pad

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced forward pass.

        Returns logits of shape ``(B, T-1, V)`` and attention weights of shape
        ``(B, T-1, S)``, predicting each target token from the previous one.
        """
        enc_outputs, state = self.encoder(src)
        mask = self._mask(src)
        logits_all, attn_all = [], []
        for t in range(tgt.shape[1] - 1):
            logits, state, weights = self.decoder.step(tgt[:, t], state, enc_outputs, mask)
            logits_all.append(logits)
            attn_all.append(weights)
        return torch.stack(logits_all, dim=1), torch.stack(attn_all, dim=1)
