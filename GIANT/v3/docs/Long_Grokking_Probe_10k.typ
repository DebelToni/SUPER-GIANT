#set page(width: 210mm, height: 297mm, margin: 18mm)
#set text(size: 11pt)
#set heading(numbering: none)

#let color-train = rgb(70, 70, 70)
#let color-val = rgb(33, 150, 243)

#let legend-item(label, color) = grid(
  columns: (auto, auto),
  gutter: 2pt,
  align: left,
  rect(width: 10pt, height: 10pt, fill: color, radius: 2pt),
  text(size: 11pt)[#label],
)

#let minimal-legend = box(width: 100%, inset: 6pt, stroke: 0.6pt + rgb(200, 200, 200), radius: 4pt)[
  #grid(
    columns: (auto, auto),
    gutter: 8pt,
    align: left,
    [#legend-item("train loss", color-train)],
    [#legend-item("held-out LM val loss", color-val)],
  )
]

#let minimal-chart(series, width: 170mm, height: 52mm, border: rgb(180, 180, 180), x_label: "Step", y_label: "Loss") = {
  let margin-left = 18pt
  let margin-right = 8pt
  let margin-top = 6pt
  let margin-bottom = 18pt
  let y-label-area = 16pt
  let plot-width = width - margin-left - margin-right
  let plot-height = height - margin-top - margin-bottom

  let min-x = calc.min(..series.map(s => calc.min(..s.data.map(d => d.at(0)))))
  let max-x = calc.max(..series.map(s => calc.max(..s.data.map(d => d.at(0)))))
  let min-y = calc.min(..series.map(s => calc.min(..s.data.map(d => d.at(1)))))
  let max-y = calc.max(..series.map(s => calc.max(..s.data.map(d => d.at(1)))))
  let pad = (max-y - min-y) * 0.08
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
    #for tick in (1000, 3000, 5000, 7000, 10000) [
      #place(top + left, dx: margin-left + (tick - min-x) * scale-x, dy: margin-top + plot-height)[
        #line(length: 4pt, angle: 90deg, stroke: 0.5pt + black)
      ]
      #place(top + left, dx: margin-left + (tick - min-x) * scale-x - 10pt, dy: margin-top + plot-height + 5pt)[
        #text(size: 7pt)[#tick]
      ]
    ]
    #for s in series [
      #place(top + left, dx: margin-left, dy: margin-top)[
        #path(
          stroke: 1.2pt + s.color,
          fill: none,
          ..s.data.map(d => (
            (d.at(0) - min-x) * scale-x,
            plot-height - (d.at(1) - min-y) * scale-y,
          ))
        )
      ]
    ]
    #place(top + left, dx: margin-left, dy: height - margin-bottom)[
      #box(width: plot-width, height: margin-bottom)[
        #align(center)[#text(size: 8pt)[#x_label]]
      ]
    ]
    #place(top + left, dx: 0pt, dy: margin-top)[
      #box(width: y-label-area, height: plot-height)[
        #align(center)[
          #rotate(-90deg, origin: center, reflow: true)[
            #box(width: plot-height, height: y-label-area)[
              #align(center)[#text(size: 8pt)[#y_label]]
            ]
          ]
        ]
      ]
    ]
  ]
}

#let train-loss = (
  (1000, 1.1830),
  (2000, 0.9140),
  (3000, 0.8500),
  (4000, 0.9250),
  (5000, 1.1070),
  (6000, 1.1140),
  (7000, 1.2400),
  (8000, 1.1140),
  (9000, 1.2030),
  (10000, 1.0940),
)

#let val-loss = (
  (1000, 2.1934),
  (2000, 1.8564),
  (3000, 1.6922),
  (4000, 1.9785),
  (5000, 1.4418),
  (6000, 1.3617),
  (7000, 1.3124),
  (8000, 1.4831),
  (9000, 1.6694),
  (10000, 1.4891),
)

= Long Grokking Probe, first 10k steps

Config: `long_40m_l1_grokking_probe_100k.yml`  \
Model: `38.64M` params  \
Run: pure LM, `1x RTX 5090`

#minimal-legend

#minimal-chart(
  (
    (label: "train", color: color-train, data: train-loss),
    (label: "val", color: color-val, data: val-loss),
  ),
)

Best held-out LM val loss in the first `10k` is `1.3124` at `step 7000`.

- `step 1000`: val `2.1934`
- `step 3000`: best train loss `0.8500`
- `step 7000`: best val loss `1.3124`
- `step 10000`: val `1.4891`

This does not look like classic delayed grokking yet. The first `10k` steps look more like ordinary early generalization improvement, with the best checkpoint appearing in the first `ctx256` phase and then regressing slightly by `10k`.

#table(
  columns: (0.8fr, 0.9fr, 0.9fr),
  [step], [train], [val],
  [1000], [1.1830], [2.1934],
  [3000], [0.8500], [1.6922],
  [5000], [1.1070], [1.4418],
  [7000], [1.2400], [1.3124],
  [10000], [1.0940], [1.4891],
)
