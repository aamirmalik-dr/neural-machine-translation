# Bahdanau attention, from the ground up

This note derives the additive attention mechanism used in this project and
explains why it is the right tool for the date-normalization task. It is written
to be read alongside `src/nmt/model.py`, where every equation here has a direct
counterpart in code.

## The problem attention solves

A plain sequence-to-sequence model reads the whole source with an encoder, then
compresses it into a single fixed-length vector and hands that vector to the
decoder. Every output character has to be produced from that one summary. For a
short input this is tolerable, but the summary becomes a bottleneck: information
about the first characters has to survive all the way through the encoder and
then through every decoding step.

Attention removes the bottleneck. Instead of forcing the decoder to work from one
vector, it lets the decoder look back at the full sequence of encoder states and,
at each output step, build a fresh summary weighted toward the source positions
that matter right now. For date normalization this is exactly what we want: when
the decoder writes the month field `03`, it should look at the characters `Mar`
in the source, and when it writes the year it should look at `2001`.

## The encoder

The encoder is a bidirectional GRU over the source characters. For a source of
length `S` it produces a sequence of hidden states

```
h_1, h_2, ..., h_S,   each h_j in R^{2H}
```

where `2H` is twice the hidden size because the forward and backward directions
are concatenated. Each `h_j` is a representation of source position `j` in the
context of the whole sequence. These are the vectors attention will weigh. The
final forward and backward states are also combined through a small linear
`bridge` and a `tanh` to initialize the decoder state `s_0`.

In code this is `Encoder.forward`, which returns `outputs` of shape `(B, S, 2H)`
and the initial decoder state.

## The additive score

At decoding step `t` the decoder holds a state `s_{t-1}` in `R^{H}`. Additive
attention compares that state against every encoder state `h_j` with a small
feed-forward network and produces one scalar score per source position:

```
e_{t,j} = v^T tanh(W_enc h_j + W_dec s_{t-1})
```

Here `W_enc` maps the `2H`-dimensional encoder state into an attention space of
size `A`, `W_dec` maps the `H`-dimensional decoder state into the same space,
they are added, squashed with `tanh`, and projected to a scalar by `v`. This is
called additive attention because the encoder and decoder contributions are
summed inside the nonlinearity, in contrast to the dot-product form that
multiplies them. The three matrices are `self.W_enc`, `self.W_dec`, and `self.v`
in `BahdanauAttention`.

## Masking and normalization

Sources in a batch have different lengths, so shorter ones are padded. Padding
positions must not receive attention, so their scores are set to negative
infinity before the softmax:

```
e_{t,j} = -inf   for every padded position j
```

The scores are then turned into a probability distribution over source positions
with a softmax:

```
alpha_{t,j} = softmax_j(e_{t,j}),    sum_j alpha_{t,j} = 1
```

The vector `alpha_t` is the attention weight for output step `t`. It is exactly
what the heatmap in `results/attention.png` visualizes: row `t` is `alpha_t`, and
a bright cell at column `j` means the decoder leaned on source character `j` when
producing output character `t`.

## The context vector and the output

The context vector is the attention-weighted average of the encoder states:

```
c_t = sum_j alpha_{t,j} h_j
```

Because the weights sum to one, `c_t` is a convex combination of encoder states,
a focused re-reading of the source for this step. The decoder GRU cell then takes
the previous output embedding concatenated with `c_t`, updates its state to
`s_t`, and the output layer maps `[s_t ; c_t]` to logits over the target
vocabulary:

```
s_t   = GRUCell([emb(y_{t-1}) ; c_t], s_{t-1})
logit = W_out [s_t ; c_t]
```

This is `Decoder.step`. One call performs one attention lookup and emits one
character. `Seq2Seq.forward` runs it across all target positions with teacher
forcing during training, and `greedy_decode` runs it autoregressively at
inference time, feeding each predicted character back in as the next input.

## Why the alignment is interpretable here

Date normalization is a copy-and-reorder task with no long-range ambiguity. Every
output character is a deterministic function of a small, contiguous group of
source characters: the year digits copy across, the month name maps to two
digits, the day copies across, and the three fields are reordered into
`YYYY-MM-DD`. A model that solves the task well has no reason to spread its
attention, so the learned weights concentrate sharply on the relevant source
span. That is why the heatmap reads almost like a permutation matrix, and why it
is a clean demonstration that the attention mechanism has learned an alignment
rather than memorized outputs.

## Honest framing of the perfect score

The held-out exact-match accuracy on this task is 1.0. That is not a claim about
general machine translation. The date task is synthetic, finite in structure, and
fully learnable on a CPU, so a correctly implemented attention model is expected
to solve it completely. The value of the perfect score is as controlled
validation: it confirms the encoder, the additive attention, the decoder, the
masking, and the greedy decoder are all wired correctly, and the heatmap confirms
the model succeeds by aligning source and target rather than by some shortcut. The
same architecture, unchanged, applies to a natural-language parallel corpus, where
scores would be far lower and beam search and subword tokenization would matter.
