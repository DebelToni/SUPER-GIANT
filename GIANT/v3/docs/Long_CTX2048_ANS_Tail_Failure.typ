#set page(width: 210mm, height: 297mm, margin: 18mm)
#set text(size: 11pt)
#set heading(numbering: none)

#let ink = rgb(28, 34, 40)
#let soft = rgb(104, 114, 126)
#let border = rgb(214, 218, 224)
#let blue = rgb(39, 102, 180)
#let red = rgb(203, 67, 53)

#let legend-item(label, color) = [
  #rect(width: 10pt, height: 10pt, fill: color, radius: 2pt)
  #h(4pt)
  #text(size: 8pt)[#label]
]

#let legend-box(items) = box(width: 100%, inset: 6pt, stroke: 0.6pt + border, radius: 4pt)[
  #grid(columns: (1fr, 1fr), gutter: 8pt, align: left, ..items)
]

#let line-chart(series, width: 168mm, height: 58mm, x_label: "Checkpoint step", y_label: "Loss") = {
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
  let pad = calc.max(0.03, (max-y - min-y) * 0.12)
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
      #for p in s.data [
        #place(top + left, dx: margin-left + (p.at(0) - min-x) * scale-x - 2.5pt, dy: margin-top + plot-height - (p.at(1) - min-y) * scale-y - 2.5pt)[
          #circle(radius: 2.5pt, fill: s.color)
        ]
      ]
    ]
    #place(top + left, dx: margin-left, dy: height - margin-bottom)[
      #box(width: plot-width, height: margin-bottom)[
        #grid(
          columns: (1fr, 1fr),
          align: horizon,
          [#text(size: 7pt)[616523]],
          [#align(right)[#text(size: 7pt)[620307]]],
        )
        #align(center)[#text(size: 8pt)[#x_label]]
      ]
    ]
    #place(right + horizon, dx: margin-left - 24pt - width, dy: margin-top + plot-height / 2 - height / 2)[
      #rotate(-90deg)[#text(size: 8pt)[#y_label]]
    ]
  ]
}

#let train-loss = (
  (616523, 1.144163),
  (620307, 1.758575),
)

#let val-loss = (
  (616523, 0.424475),
  (620307, 1.562885),
)

= Long ctx2048: 1% ANS tail hurts

#text(fill: soft)[38.74M decoder + XSA, exact 20x token budget, 40% curriculum warmup, final comparison on a fresh 1024-example ctx2048 validation set.]

== Loss overlay

#legend-box((
  [#legend-item("checkpoint train loss", blue)],
  [#legend-item("validation loss", red)],
))

#line-chart(
  (
    (label: "train", color: blue, data: train-loss),
    (label: "val", color: red, data: val-loss),
  ),
)

The only change between the two points is the final `1%` answer-only tail.

- step `616523`: checkpoint right before ANS
- step `620307`: final checkpoint after the ANS tail

== Accuracy

- pre-ANS checkpoint: `910 / 1024 = 88.87%`
- final checkpoint after ANS: `764 / 1024 = 74.61%`
- delta: `-14.26` accuracy points

== Finding

- The final `1%` ANS phase makes both metrics worse: train loss rises, validation loss rises, and exact-match accuracy drops sharply.
- For future Long runs, do not add an ANS tail by default.
- The safer default is to stop at the LM checkpoint before ANS, or only revisit ANS with a much smaller tail and explicit validation gating.

== Artifacts

- config: `GIANT/v3/Configs/Training/Long/long_40m_l1_ctx2048_chinchilla20x_warm40_5090.yml`
- evals: `s3://giant-data/GIANT/single-gpu/logs/long/ctx2048_chinchilla20x_warm40_5090/pre_ans_val1024.json`
- evals: `s3://giant-data/GIANT/single-gpu/logs/long/ctx2048_chinchilla20x_warm40_5090/final_val1024.json`
- compare: `s3://giant-data/GIANT/single-gpu/logs/long/ctx2048_chinchilla20x_warm40_5090/pre_vs_final_val1024_compare.json`
