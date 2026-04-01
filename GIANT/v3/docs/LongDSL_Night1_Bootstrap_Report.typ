#set page(width: 210mm, height: 297mm, margin: 18mm)
#set text(size: 11pt)
#set heading(numbering: none)

#let ink = rgb(32, 38, 46)
#let soft = rgb(110, 118, 129)
#let line = rgb(210, 214, 220)
#let blue = rgb(52, 120, 246)
#let green = rgb(46, 160, 67)
#let orange = rgb(245, 158, 11)
#let red = rgb(220, 53, 69)

#let stat-card(title, value, note: none, tint: blue) = box(
  width: 100%,
  inset: 8pt,
  stroke: 0.7pt + line,
  radius: 6pt,
  fill: tint.lighten(88%),
)[
  #text(size: 9pt, fill: soft)[#title]
  #v(2pt)
  #text(size: 15pt, weight: "bold", fill: ink)[#value]
  #if note != none [
    #v(2pt)
    #text(size: 8pt, fill: soft)[#note]
  ]
]

#let pct-bar(frac, color, width: 64mm) = {
  let inner = frac * (width - 2pt)
  box(width: width, inset: 1pt, radius: 3pt, stroke: 0.6pt + line, fill: rgb(248, 249, 251))[
    #box(width: inner, height: 8pt, radius: 2pt, fill: color)[]
  ]
}

#let result-row(label, frac, text-value, color, comment: none) = [
  #grid(
    columns: (48mm, 68mm, 24mm, 1fr),
    gutter: 8pt,
    align: (left, center, right, left),
    [#text(size: 9.5pt, weight: "semibold")[#label]],
    [#pct-bar(frac, color)],
    [#text(size: 9.5pt)[#text-value]],
    [#text(size: 8.5pt, fill: soft)[#if comment != none [#comment]]],
  )
]

= LongDSL Night 1 Bootstrap Report

#text(fill: soft)[GIANT v3 synthetic long-context bootstrap experiment note.]

== Purpose

This document records the first autonomous LongDSL stress-test session for GIANT v3. The immediate goal was not to reach 64K or 1M context in one night, but to answer a narrower question first:

- Can a small GIANT v3 baseline learn the LongDSL retrieval task at all?
- Is direct-from-scratch long-context training viable, or is a short-context bootstrap required?
- Does the transfer path survive the first context jumps?

== Experiment Summary

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  stat-card("Model", "~28.5M params", note: "384 dim, 6 heads, 12 layers, FF 1536", tint: blue),
  stat-card("GPU", "1x RTX A6000", note: "single GPU RunPod secure cloud session", tint: orange),
  stat-card("Best Result", "45.31% exact match", note: "level 1, 128 tokens, transfer from 64-token checkpoint", tint: green),
)

#v(6pt)

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  stat-card("DSL Focus", "Level 1 retrieval", note: "entity -> value with aliasing and distractors", tint: blue),
  stat-card("Best Bootstrap", "64 -> 128", note: "lower-LR transfer-only answer adaptation", tint: green),
  stat-card("Failure Point", "256 tokens", note: "transfer quality collapsed sharply", tint: red),
)

== LongDSL Setup

=== DSL structure

The new synthetic pipeline lives under `GIANT/v3/Long`. Three levels were implemented:

- Level 1: `DEF`, `SET`, `ALIAS`, `ASK`
- Level 2: `INC`, `DEC`, `SWAP`
- Level 3: `DEFARR`, `SETAT`, `SWAPAT`, `INCAT`, `GET`

The overnight runs stayed on level 1 only. That was intentional: the baseline first had to prove it could learn the simplest retrieval version before pushing to more complex operators or longer contexts.

=== Tokenization and data layout

- Small custom word-level tokenizer stored at `/proj/giant-data/GIANT/Long/tokenizers/longdsl_wordlevel`
- Raw JSONL samples stored under `/proj/giant-data/GIANT/Long/raw/...`
- Arrow shards stored under `/proj/giant-data/GIANT/dataset_artifacts/...`
- Curated saved night results stored under `/proj/giant-data/GIANT/single-gpu/checkpoints/longdsl/night1`

=== Baseline model

```yaml
embedding_size: 384
num_heads: 6
num_kv_heads: 6
num_layers: 12
feed_forward_size: 1536
rope_dim: 64
dropout_rate: 0.0
param_dtype: float32
compute_dtype: bfloat16
```

This is the smallest GIANT v3 shape that looked reasonable while still being cheap enough to iterate on quickly.

== Training Variants Tried

=== 1. Direct long-context baseline

The first family of runs tried to learn the task directly at larger context lengths, including 512-token setups. Several dataset and masking variants were tried during the night:

- chat-style masked assistant target
- plain LM ending with `ANS Vxxx`
- two-stage `LM -> answer-only` supervision at the same context length

These direct 512-token baselines consistently stayed near floor on held-out exact match. This was the strongest negative result of the session.

=== 2. Bootstrap curriculum

The first useful signal appeared only after switching to a short-context curriculum:

- bootstrap at 64 tokens
- transfer from the 64-token checkpoint into 128 tokens
- lower learning rate during transfer
- answer-focused adaptation instead of re-running the whole curriculum from scratch

This transfer path was clearly better than direct 512-token training.

== Key Results

=== Held-out exact match

#box(width: 100%, inset: 8pt, stroke: 0.7pt + line, radius: 6pt)[
  #result-row(
    "Direct baseline @ 512",
    0.0312,
    "4 / 128",
    red,
    comment: "baseline_v5, LM + answer-only at 512 from scratch",
  )
  #v(5pt)
  #result-row(
    "Bootstrap @ 64",
    0.4375,
    "56 / 128",
    blue,
    comment: "bootstrap_v1, first positive result",
  )
  #v(5pt)
  #result-row(
    "64 -> 128 transfer",
    0.4531,
    "58 / 128",
    green,
    comment: "bootstrap_v2_transfer, best run of the night",
  )
  #v(5pt)
  #result-row(
    "128 -> 256 transfer",
    0.0469,
    "6 / 128",
    red,
    comment: "quality collapsed after the next context jump",
  )
]

=== Interpretation

- The task is learnable by the model family, so the idea is not dead.
- Direct-from-scratch long-context training is a poor entry point.
- A short-context bootstrap is necessary.
- The `64 -> 128` transition works.
- The `128 -> 256` transition does not yet hold, which means the curriculum is still too aggressive or too sparse.

== What Actually Changed the Outcome

The main engineering lesson from the night is that the data formulation mattered more than any architectural tweak.

The final useful setup was:

- short-context bootstrap first
- lower-LR transfer after bootstrap
- answer-only supervision at transfer time
- reuse of the smaller-context checkpoint instead of restarting from scratch

The following approaches did *not* unlock learning:

- jumping straight to 512 tokens from scratch
- sparse answer-only training without a bootstrap stage
- retraining a larger context with a fresh optimizer and no careful transfer recipe

== Log Shape Notes

The LM bootstrap stages usually converged to a narrow band around loss ~`2.3 - 2.6`, which is enough to make the DSL syntax/model state somewhat legible to the model, but not enough by itself to solve retrieval.

The answer-only stages showed a very different signature:

- individual batches sometimes hit very low loss
- neighboring batches could still spike badly
- this means the model sometimes retrieved the correct value, but the behavior was not stable enough across the whole evaluation set

That pattern is exactly what the exact-match results show:

- retrieval skill appears locally
- robustness across samples collapses too early as context grows

== Recommended Next Setup

The next experiment should *not* continue from 256 immediately. The evidence says a better curriculum is needed first.

Recommended next changes:

- Make level 1 easier during bootstrap:
  - shorter alias chains
  - lower distractor density
  - fewer active entities/values early on
- Keep the transfer recipe that worked:
  - bootstrap at a short length
  - lower-LR transfer-only answer adaptation
- Add finer ladder steps instead of big jumps:
  - 64 -> 96 -> 128 -> 192 -> 256
- Only retry 512 after 256 is clearly stable.

== Main Conclusion

The overnight result is a partial win, not a headline win.

- We did *not* prove that the current 30M GIANT v3 setup can simply scale LongDSL retrieval to 256+ tokens by brute force.
- We *did* prove that the task becomes learnable once the model is bootstrapped at short context and transferred carefully.
- The correct next research direction is curriculum design and transfer stability, not immediate long-context scaling.

That is enough signal to justify a second experiment configuration, but not enough to claim that the current setup is ready for the long-context ladder.

== Saved Artifacts

Important local artifacts:

- `/proj/giant-data/GIANT/single-gpu/checkpoints/longdsl/night1/bootstrap_v1_l1_ctx64/step_0010240.npz`
- `/proj/giant-data/GIANT/single-gpu/checkpoints/longdsl/night1/bootstrap_v2_transfer_l1_ctx128/step_0016384.npz`
- `/proj/giant-data/GIANT/single-gpu/checkpoints/longdsl/night1/bootstrap_v2_transfer_l1_ctx256/step_0019456.npz`
- `/proj/giant-data/GIANT/single-gpu/checkpoints/longdsl/night1/baseline_v5_l1_ctx512/step_0005120.npz`

Mirrored S3 location:

- `s3://giant-data/GIANT/single-gpu/checkpoints/longdsl/night1/`

== Session Status

The GPU pod used for the overnight run was terminated at the end of the session after the result checkpoints and logs were copied back and synced.
