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

#let compare-chart(series, width: 170mm, height: 56mm, border: rgb(180, 180, 180), x_label: "Step", y_label: "Loss") = {
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
    #for tick in (10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000) [
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
  (11000, 1.1100),
  (12000, 1.1480),
  (13000, 1.0060),
  (14000, 1.0190),
  (15000, 1.2150),
  (16000, 0.9340),
  (17000, 1.1590),
  (18000, 1.1230),
  (19000, 0.9420),
  (20000, 1.1370),
  (21000, 1.2330),
  (22000, 1.2890),
  (23000, 1.0670),
  (24000, 1.0540),
  (25000, 0.9330),
  (26000, 1.1840),
  (27000, 1.1730),
  (28000, 0.9520),
  (29000, 0.9900),
  (30000, 1.0880),
  (31000, 1.0210),
  (32000, 0.9300),
  (33000, 0.8100),
  (34000, 0.9270),
  (35000, 1.2180),
  (36000, 0.9260),
  (37000, 0.9980),
  (38000, 1.0360),
  (39000, 0.8040),
  (40000, 0.8330),
  (41000, 0.9700),
  (42000, 1.0900),
  (43000, 0.9940),
  (44000, 0.7720),
  (45000, 1.0610),
  (46000, 1.0660),
  (47000, 0.9550),
  (48000, 1.1780),
  (49000, 1.0710),
  (50000, 0.9470),
  (51000, 1.0730),
  (52000, 0.8930),
  (53000, 1.0530),
  (54000, 1.0510),
  (55000, 0.4860),
  (56000, 0.7790),
  (57000, 0.7280),
  (58000, 0.8500),
  (59000, 0.9650),
  (60000, 1.1190),
  (61000, 1.0660),
  (62000, 0.7570),
  (63000, 0.6240),
  (64000, 0.7450),
  (65000, 0.7950),
  (66000, 1.0520),
  (67000, 0.8500),
  (68000, 0.8790),
  (69000, 0.9930),
  (70000, 0.7740),
  (71000, 0.9280),
  (72000, 0.6840),
  (73000, 0.5060),
  (74000, 0.7890),
  (75000, 0.7010),
  (76000, 0.9110),
  (77000, 0.7510),
  (78000, 0.7920),
  (79000, 0.8890),
  (80000, 0.8540),
  (81000, 0.8550),
  (82000, 0.8630),
  (83000, 0.9110),
  (84000, 0.9080),
  (85000, 0.7600),
  (86000, 0.6250),
  (87000, 0.6280),
  (88000, 0.7760),
  (89000, 0.6960),
  (90000, 0.7310),
  (91000, 1.0660),
  (92000, 0.9130),
  (93000, 0.6290),
  (94000, 0.6650),
  (95000, 0.4940),
  (96000, 0.4860),
  (97000, 0.7210),
  (98000, 0.7240),
  (99000, 0.6560),
  (100000, 0.6940)
)

#let val-loss = (
  (10000, 1.4891),
  (20000, 1.3408),
  (30000, 1.3103),
  (40000, 1.4527),
  (50000, 1.4692),
  (60000, 1.2602),
  (70000, 1.4302),
  (80000, 1.5527),
  (90000, 1.4791),
  (91000, 1.3391),
  (92000, 1.3139),
  (93000, 1.5541),
  (94000, 1.7025),
  (95000, 1.6713),
  (96000, 1.7130),
  (97000, 1.5151),
  (98000, 1.5747),
  (99000, 1.3992),
  (100000, 1.5074)
)

= Long Grokking Probe, full 100k

Config: `long_40m_l1_grokking_probe_100k.yml`  Model: `38.64M` params  Run: pure LM, `1x RTX 5090`  Train curve from `logs.txt` at every `1k`; held-out LM val curve from evaluated checkpoints.

#box(width: 100%, inset: 6pt, stroke: 0.6pt + rgb(200, 200, 200), radius: 4pt)[
  #grid(
    columns: (auto, auto),
    gutter: 8pt,
    align: left,
    [#legend-item("train loss", color-train)],
    [#legend-item("held-out LM val loss", color-val)],
  )
]

#compare-chart(
  (
    (label: "train", color: color-train, data: train-loss),
    (label: "val", color: color-val, data: val-loss),
  ),
)

- best held-out LM val loss in the sampled full run is `1.2602` at `step 60000`
- best train loss from the `1k` samples is `0.4860` at `step 55000`
- val loss goes from `1.4891` at `step 10000` to `1.5074` at `step 100000`

This still does not read like classic late grokking. Generalization improves well past the early `7k` result and reaches its best sampled point around `60k`, but it does not show a late monotonic jump near the end; the last `10k` is worse than that `60k` minimum even while train loss stays low.
