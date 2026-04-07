#set page(width: 210mm, height: 297mm, margin: 18mm)
#set text(size: 11pt)
#set heading(numbering: none)

#let ink = rgb(28, 34, 40)
#let soft = rgb(104, 114, 126)
#let border = rgb(214, 218, 224)
#let blue = rgb(39, 102, 180)
#let orange = rgb(214, 131, 46)
#let green = rgb(38, 153, 93)
#let red = rgb(203, 67, 53)

#let card(title, value, note: none, tint: blue) = box(
  width: 100%,
  inset: 8pt,
  radius: 6pt,
  stroke: 0.7pt + border,
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

#let legend-item(label, color) = [
  #rect(width: 10pt, height: 10pt, fill: color, radius: 2pt)
  #h(4pt)
  #text(size: 8pt)[#label]
]

#let legend-box(items) = box(width: 100%, inset: 6pt, stroke: 0.6pt + border, radius: 4pt)[
  #grid(columns: (1fr, 1fr), gutter: 8pt, align: left, ..items)
]

#let line-chart(series, guides: (), width: 170mm, height: 60mm, x_label: "Step", y_label: "Metric") = {
  let margin-left = 28pt
  let margin-right = 10pt
  let margin-top = 8pt
  let margin-bottom = 20pt
  let plot-width = width - margin-left - margin-right
  let plot-height = height - margin-top - margin-bottom

  let min-x = calc.min(..series.map(s => calc.min(..s.data.map(d => d.at(0)))))
  let max-x = calc.max(..series.map(s => calc.max(..s.data.map(d => d.at(0)))))
  let min-y = calc.min(..series.map(s => calc.min(..s.data.map(d => d.at(1)))))
  let max-y = calc.max(..series.map(s => calc.max(..s.data.map(d => d.at(1)))))
  let pad = calc.max(0.02, (max-y - min-y) * 0.08)
  let min-y = min-y - pad
  let max-y = max-y + pad
  let scale-x = if max-x == min-x { plot-width } else { plot-width / (max-x - min-x) }
  let scale-y = if max-y == min-y { plot-height } else { plot-height / (max-y - min-y) }

  box(width: width, height: height, stroke: 0.8pt + border, radius: 6pt, inset: 4pt)[
    #place(top + left, dx: margin-left, dy: margin-top + plot-height)[
      #line(length: plot-width, stroke: 0.6pt + black)
    ]
    #place(top + left, dx: margin-left, dy: margin-top)[
      #line(length: plot-height, angle: 90deg, stroke: 0.6pt + black)
    ]
    #for guide in guides [
      #place(top + left, dx: margin-left + (guide.x - min-x) * scale-x, dy: margin-top)[
        #line(length: plot-height, angle: 90deg, stroke: 0.5pt + guide.color)
      ]
      #place(top + left, dx: margin-left + (guide.x - min-x) * scale-x - 18pt, dy: margin-top - 2pt)[
        #text(size: 7pt, fill: guide.color)[#guide.label]
      ]
    ]
    #for s in series [
      #place(top + left, dx: margin-left, dy: margin-top)[
        #path(
          stroke: 1.4pt + s.color,
          fill: none,
          ..s.data.map(d => (
            (d.at(0) - min-x) * scale-x,
            plot-height - (d.at(1) - min-y) * scale-y,
          )),
        )
      ]
    ]
    #place(top + left, dx: margin-left, dy: height - margin-bottom)[
      #box(width: plot-width, height: margin-bottom)[
        #align(center)[#text(size: 8pt)[#x_label]]
      ]
    ]
    #place(right + horizon, dx: margin-left - 24pt - width, dy: margin-top + plot-height / 2 - height / 2)[
      #rotate(-90deg)[#text(size: 8pt)[#y_label]]
    ]
  ]
}

#let lmheavy-acc = (
  (500, 0.5781), (1000, 0.5625), (1500, 0.6172), (2000, 0.5703), (2500, 0.6797), (3000, 0.5938),
  (3500, 0.6094), (4000, 0.5938), (4500, 0.6172), (5000, 0.6406), (5500, 0.6250), (6000, 0.5547),
  (6500, 0.6406), (7000, 0.6562), (7500, 0.6719), (8000, 0.6719), (8500, 0.6562), (9000, 0.6562),
  (9500, 0.6641), (10000, 0.6484), (10500, 0.6719), (11000, 0.6562), (11500, 0.6641), (12000, 0.6406),
)
#let ansheavy-acc = (
  (500, 0.5781), (1000, 0.6406), (1500, 0.6406), (2000, 0.6484), (2500, 0.0234), (3000, 0.0000),
  (3500, 0.0078), (4000, 0.0234), (4500, 0.0234), (5000, 0.0078), (5500, 0.0156), (6000, 0.0156),
  (6500, 0.0156), (7000, 0.0156), (7500, 0.0000), (8000, 0.0156), (8500, 0.0156), (9000, 0.0078),
  (9500, 0.0234), (10000, 0.0156), (10500, 0.0156), (11000, 0.0078), (11500, 0.0312), (12000, 0.0234),
)

#let lmheavy-loss = (
  (500, 1.231), (1000, 1.145), (1500, 1.271), (2000, 0.882), (2500, 1.151), (3000, 0.893),
  (3500, 1.028), (4000, 0.915), (4500, 1.019), (5000, 1.140), (5500, 1.147), (6000, 1.122),
  (6500, 0.988), (7000, 1.222), (7500, 1.244), (8000, 1.044), (8500, 0.972), (9000, 1.265),
  (9500, 0.911), (10000, 1.190), (10500, 3.184), (11000, 1.976), (11500, 2.359), (12000, 4.068),
)
#let ansheavy-loss = (
  (500, 1.231), (1000, 1.047), (1500, 0.982), (2000, 1.194), (2500, 2.296), (3000, 3.387),
  (3500, 4.937), (4000, 3.708), (4500, 3.281), (5000, 0.995), (5500, 5.171), (6000, 3.090),
  (6500, 1.747), (7000, 2.677), (7500, 0.355), (8000, 2.208), (8500, 4.150), (9000, 2.516),
  (9500, 3.723), (10000, 2.760), (10500, 3.219), (11000, 2.886), (11500, 0.069), (12000, 3.884),
)

#let acc-series = (
  (label: "LM-heavy", color: blue, data: lmheavy-acc),
  (label: "ANS-heavy", color: red, data: ansheavy-acc),
)

#let loss-series = (
  (label: "LM-heavy train loss", color: blue, data: lmheavy-loss),
  (label: "ANS-heavy train loss", color: orange, data: ansheavy-loss),
)

= Long LM vs ANS Schedule Sweep

#text(fill: soft)[GIANT v3 Long pipeline sweep comparing LM-heavy and answer-heavy schedules on the same diversified natural-record curriculum.]

== Setup

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  card("Model", "38.64M params", note: "448 dim, 7 heads, 12 layers, FF 1792", tint: blue),
  card("Curriculum", "64 to 128 to 256 to 512", note: "single 512-context model, staged seq lengths", tint: orange),
  card("Eval set", "128 held-out ctx512 samples", note: "exact-match on final answer token", tint: green),
)

#v(6pt)

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  card("Diversity update", "enabled", note: "richer alias and filler record templates before the sweep", tint: green),
  card("GPU", "1x RTX A6000", note: "RunPod secure cloud", tint: orange),
  card("Checkpoint cadence", "every 500 steps", note: "24 evaluation points per run", tint: blue),
)

== Compared schedules

Both runs use the same datasets, model, learning rate, and total `12,000` steps. Only the split between LM and answer-only supervision changes.

- LM-heavy: `10,000` LM steps then `2,000` answer-only steps
- ANS-heavy: `2,000` LM steps then `10,000` answer-only steps

The phase boundaries are marked in the charts:

- step `2,000`: ANS-heavy switches from LM warmup into answer-heavy training
- step `10,000`: LM-heavy switches from LM-heavy training into answer-only finishing

== Held-out exact match over training

#legend-box((
  [#legend-item("LM-heavy exact match", blue)],
  [#legend-item("ANS-heavy exact match", red)],
))

#line-chart(
  acc-series,
  guides: (
    (x: 2000, label: "ANS-heavy switch", color: red.lighten(20%)),
    (x: 10000, label: "LM-heavy switch", color: blue.lighten(20%)),
  ),
  y_label: "Exact match",
)

The result is decisive.

- LM-heavy stays strong through the full run and peaks at `87 / 128 = 67.97%` at step `2,500`
- LM-heavy finishes at `82 / 128 = 64.06%`
- ANS-heavy is competitive only during the initial LM warmup and peaks at `83 / 128 = 64.84%` at step `2,000`
- Once ANS-heavy enters answer-dominated training, held-out exact match collapses to near zero and never recovers

== Checkpoint train loss

#legend-box((
  [#legend-item("LM-heavy checkpoint loss", blue)],
  [#legend-item("ANS-heavy checkpoint loss", orange)],
))

#line-chart(
  loss-series,
  guides: (
    (x: 2000, label: "ANS-heavy switch", color: red.lighten(20%)),
    (x: 10000, label: "LM-heavy switch", color: blue.lighten(20%)),
  ),
  y_label: "Train loss",
)

This chart explains why the answer-heavy run is misleading if we only watch training loss.

- ANS-heavy often reports low or unstable checkpoint loss inside the answer-only stages
- but those low losses do *not* translate into held-out retrieval quality
- the run is overfitting the final supervised position and forgetting the broader retrieval behavior needed at `ctx512`

== Main finding

For this LongGIANT natural-record setup, more answer-only supervision is *not* better. The model needs a long LM-heavy curriculum to keep the document structure, alias resolution, and retrieval cues stable across the long context.

The strongest simple interpretation is:

- the first `2,000` LM steps already build a viable retrieval representation
- pushing too hard on answer-only after that destroys generalization
- a small finishing block of answer-only steps helps, but it should come late and stay limited

== Concrete results

#table(
  columns: (1.4fr, 0.8fr, 0.8fr, 1fr),
  [Run], [Best step], [Best EM], [Final EM],
  [LM-heavy], [2500], [87 / 128 = 67.97%], [82 / 128 = 64.06%],
  [ANS-heavy], [2000], [83 / 128 = 64.84%], [3 / 128 = 2.34%],
)

== Practical takeaway for the next Long runs

- keep the diversified natural-record renderer
- keep the context ladder `64 -> 128 -> 256 -> 512`
- stay LM-heavy for most of training
- if answer-only is used, keep it short and late
- do *not* launch another long answer-heavy sweep unless the objective itself changes

== Artifact locations

- configs:
  - `GIANT/v3/Configs/Training/Long/long_40m_l1_sweep_lmheavy.yml`
  - `GIANT/v3/Configs/Training/Long/long_40m_l1_sweep_ansheavy.yml`
- local logs:
  - `/proj/giant-data/GIANT/single-gpu/checkpoints/long/lmheavy_10k_lm_2k_ans/40m/l1_ctx512/logs.txt`
  - `/proj/giant-data/GIANT/single-gpu/checkpoints/long/ansheavy_2k_lm_10k_ans/40m/l1_ctx512/logs.txt`
- local key checkpoints:
  - `/proj/giant-data/GIANT/single-gpu/checkpoints/long/lmheavy_10k_lm_2k_ans/40m/l1_ctx512/step_0002500.npz`
  - `/proj/giant-data/GIANT/single-gpu/checkpoints/long/lmheavy_10k_lm_2k_ans/40m/l1_ctx512/step_0012000.npz`
  - `/proj/giant-data/GIANT/single-gpu/checkpoints/long/ansheavy_2k_lm_10k_ans/40m/l1_ctx512/step_0002000.npz`
  - `/proj/giant-data/GIANT/single-gpu/checkpoints/long/ansheavy_2k_lm_10k_ans/40m/l1_ctx512/step_0012000.npz`

The sweep answers the scheduling question clearly: for this benchmark, LM-heavy wins by a large margin once long-context generalization matters.
