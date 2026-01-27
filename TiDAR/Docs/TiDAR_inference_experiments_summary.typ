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

== Throughput Sweep (RTX 3060)
#let chart_height = 46pt

#let bar_cell(value, max_speed, fill) = table(
  columns: 1,
  inset: 0pt,
  align: center,
  [#box(width: 100%, height: chart_height, stroke: 0.5pt)[
    #align(bottom)[
      #rect(width: 100%, height: (value / max_speed) * chart_height, fill: fill, radius: 2pt)
    ]
  ]],
  [#text(str(value))],
)

=== dense_bigmask_scan (iter/s)
#let max_speed = 95.0
#let fill = rgb(255, 110, 165)
#box(width: 100%)[
  #table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    gutter: 8pt,
    align: center,
    [*ctx/K*], [*4*], [*8*], [*16*], [*32*],
    [*512*], bar_cell(94.23, max_speed, fill), bar_cell(77.94, max_speed, fill), bar_cell(34.20, max_speed, fill), bar_cell(10.38, max_speed, fill),
    [*1024*], bar_cell(57.19, max_speed, fill), bar_cell(46.32, max_speed, fill), bar_cell(22.01, max_speed, fill), bar_cell(8.10, max_speed, fill),
    [*2048*], bar_cell(29.01, max_speed, fill), bar_cell(25.72, max_speed, fill), bar_cell(13.01, max_speed, fill), bar_cell(5.45, max_speed, fill),
    [*8192*], bar_cell(8.03, max_speed, fill), bar_cell(7.11, max_speed, fill), bar_cell(3.78, max_speed, fill), bar_cell(1.92, max_speed, fill)
  )
]

=== dense_tightkeys_scan (iter/s)
#let max_speed = 135.0
#let fill = rgb(60, 115, 217)
#box(width: 100%)[
  #table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    gutter: 8pt,
    align: center,
    [*ctx/K*], [*4*], [*8*], [*16*], [*32*],
    [*512*], bar_cell(134.46, max_speed, fill), bar_cell(105.87, max_speed, fill), bar_cell(40.25, max_speed, fill), bar_cell(10.70, max_speed, fill),
    [*1024*], bar_cell(80.52, max_speed, fill), bar_cell(64.52, max_speed, fill), bar_cell(27.59, max_speed, fill), bar_cell(8.56, max_speed, fill),
    [*2048*], bar_cell(43.13, max_speed, fill), bar_cell(42.22, max_speed, fill), bar_cell(16.72, max_speed, fill), bar_cell(6.40, max_speed, fill),
    [*8192*], bar_cell(9.82, max_speed, fill), bar_cell(9.65, max_speed, fill), bar_cell(4.09, max_speed, fill), bar_cell(1.85, max_speed, fill)
  )
]

=== split_calls_scan (iter/s)
#let max_speed = 19.0
#let fill = rgb(60, 170, 90)
#box(width: 100%)[
  #table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    gutter: 8pt,
    align: center,
    [*ctx/K*], [*4*], [*8*], [*16*], [*32*],
    [*512*], bar_cell(18.12, max_speed, fill), bar_cell(10.09, max_speed, fill), bar_cell(5.17, max_speed, fill), bar_cell(2.59, max_speed, fill),
    [*1024*], bar_cell(10.40, max_speed, fill), bar_cell(5.78, max_speed, fill), bar_cell(2.94, max_speed, fill), bar_cell(1.46, max_speed, fill),
    [*2048*], bar_cell(5.70, max_speed, fill), bar_cell(3.13, max_speed, fill), bar_cell(1.60, max_speed, fill), bar_cell(0.80, max_speed, fill),
    [*8192*], bar_cell(1.51, max_speed, fill), bar_cell(0.83, max_speed, fill), bar_cell(0.42, max_speed, fill), bar_cell(0.21, max_speed, fill)
  )
]

=== pallas_optional_scan (iter/s)
#let max_speed = 95.0
#let fill = rgb(140, 140, 55)
#box(width: 100%)[
  #table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    gutter: 8pt,
    align: center,
    [*ctx/K*], [*4*], [*8*], [*16*], [*32*],
    [*512*], bar_cell(93.65, max_speed, fill), bar_cell(78.47, max_speed, fill), bar_cell(33.90, max_speed, fill), bar_cell(10.73, max_speed, fill),
    [*1024*], bar_cell(55.46, max_speed, fill), bar_cell(46.65, max_speed, fill), bar_cell(21.91, max_speed, fill), bar_cell(8.05, max_speed, fill),
    [*2048*], bar_cell(29.42, max_speed, fill), bar_cell(25.72, max_speed, fill), bar_cell(13.02, max_speed, fill), bar_cell(5.55, max_speed, fill),
    [*8192*], bar_cell(8.04, max_speed, fill), bar_cell(7.12, max_speed, fill), bar_cell(3.77, max_speed, fill), bar_cell(1.91, max_speed, fill)
  )
]

== Notes
- Metric: iter/s (1 / seconds_per_iter).
- Prefill length: `round(0.66 * context_len)`.
- Config: heads=16, d=64, steps=128, iters=10, dtype=bf16, impl=cudnn.
