#set page(width: 210mm, height: 297mm, margin: 18mm)
#set text(size: 11pt)

= TiDAR: Short Summary + Throughput Snapshot

== What TiDAR Does
TiDAR (Think in Diffusion, Talk in Autoregression) keeps a single decoder-only transformer
but trains it to operate under two synchronized attention regimes:

- *Talk (AR)*: standard causal self-attention for verifying the current draft tokens.
- *Think (Diffusion)*: blockwise bidirectional attention for drafting the next candidate tokens.

Training doubles each sequence (clean + masked copy) so the diffusion branch predicts aligned
targets with blockwise masks; inference uses structured masks to verify the current K tokens
and pre-draft the next K candidates in a single forward pass.

== One-Pass Inference Sketch
```text
prefix tokens ─────────► Talk (AR verify) ──► commit r tokens
verify tokens (K) ─────►
candidate blocks (K×K) ─► Think (Diffusion draft) ─► next K proposals
```

== Throughput Benchmark (GPU)
#let chart_height = 120pt
#let max_speed = 140.0

#let bar(label, speed, fill) = table(
  columns: 1,
  inset: 0pt,
  align: center,
  [#box(width: 100%, height: chart_height, stroke: 0.5pt)[
    #align(bottom)[
      #rect(width: 100%, height: (speed / max_speed) * chart_height, fill: fill, radius: 2pt)
    ]
  ]],
  [#text(label)],
  [#text(str(speed) + " tok/s")],
)

#box(width: 100%)[
  #table(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 14pt,
    inset: 0pt,
    align: center,
    [#bar("AR only", 136.25, rgb(255, 110, 165))],
    [#bar("TiDAR draft_len=1", 25.04, rgb(60, 115, 217))],
    [#bar("TiDAR draft_len=5", 66.44, rgb(60, 170, 90))],
    [#bar("TiDAR draft_len=10", 87.49, rgb(140, 140, 55))],
  )
]

== Notes
- AR-only baseline uses `draft_len=0`.
- TiDAR results use K in {1, 5, 10} with the structured one-pass mask.
