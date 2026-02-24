#set page(width: 210mm, height: 297mm, margin: 18mm)
#set text(size: 11pt)
#set heading(numbering: none)

#let c-h6 = rgb(33, 150, 243)      // h6 MHA baseline
#let c-h18 = rgb(244, 67, 54)      // h18 MHA
#let c-gqa = rgb(76, 175, 80)      // h18 / kv6 GQA
#let c-h6hd32 = rgb(255, 152, 0)   // h6 MHA, head_dim=32 control
#let border-color = rgb(180, 180, 180)

#let legend-item(label, color) = grid(
  columns: (auto, auto),
  gutter: 2pt,
  align: left,
  rect(width: 10pt, height: 10pt, fill: color, radius: 2pt),
  text(size: 10pt)[#(label)],
)

#let multi-line-chart(
  series,
  width: 165mm,
  height: 60mm,
  x_label: "Step",
  y_label: "Metric",
  x_min: none,
  x_max: none,
  y_min: none,
  y_max: none,
  y_ticks: 0,
  y_tick_decimals: 3,
  baseline_y: none,
) = {
  let margin-left = 24pt
  let margin-right = 8pt
  let margin-top = 6pt
  let margin-bottom = 18pt
  let y-label-gap = -2pt
  let plot-width = width - margin-left - margin-right
  let plot-height = height - margin-top - margin-bottom

  let min-x = if x_min != none { x_min } else { calc.min(..series.map(s => calc.min(..s.data.map(d => d.at(0))))) }
  let max-x = if x_max != none { x_max } else { calc.max(..series.map(s => calc.max(..s.data.map(d => d.at(0))))) }
  let min-y = if y_min != none { y_min } else { calc.min(..series.map(s => calc.min(..s.data.map(d => d.at(1))))) }
  let max-y = if y_max != none { y_max } else { calc.max(..series.map(s => calc.max(..s.data.map(d => d.at(1))))) }

  let pad = (max-y - min-y) * 0.08
  let min-y = min-y - pad
  let max-y = max-y + pad

  let scale-x = if max-x == min-x { plot-width } else { plot-width / (max-x - min-x) }
  let scale-y = if max-y == min-y { plot-height } else { plot-height / (max-y - min-y) }

  box(width: width, height: height, stroke: 0.8pt + border-color, radius: 6pt, inset: 4pt)[
    #place(top + left, dx: margin-left, dy: margin-top + plot-height)[
      #line(length: plot-width, stroke: 0.6pt + black)
    ]
    #place(top + left, dx: margin-left, dy: margin-top)[
      #line(length: plot-height, angle: 90deg, stroke: 0.6pt + black)
    ]

    #if baseline_y != none [
      #let base-pos = plot-height - (baseline_y - min-y) * scale-y
      #place(top + left, dx: margin-left, dy: margin-top + base-pos)[
        #line(length: plot-width, stroke: (paint: rgb(170, 170, 170), thickness: 0.7pt, dash: (2pt, 2pt)))
      ]
    ]

    #if y_ticks > 0 [
      #for i in range(0, y_ticks + 1) [
        #let yv = min-y + (max-y - min-y) * i / y_ticks
        #let py = plot-height - (yv - min-y) * scale-y
        #let ytxt = str(calc.round(yv, digits: y_tick_decimals))
        #place(top + left, dx: margin-left - 2pt, dy: margin-top + py)[
          #line(length: 2pt, stroke: 0.5pt + rgb(120, 120, 120))
        ]
        #place(top + left, dx: 0pt, dy: margin-top + py - 4pt)[
          #box(width: margin-left - 4pt)[
            #align(right)[#text(size: 6.6pt, fill: rgb(90, 90, 90))[#ytxt]]
          ]
        ]
      ]
    ]

    #for s in series [
      #place(top + left, dx: margin-left, dy: margin-top)[
        #path(
          stroke: 1.2pt + s.color,
          fill: none,
          ..s.data.map(d => (
            (d.at(0) - min-x) * scale-x,
            plot-height - (d.at(1) - min-y) * scale-y
          ))
        )
      ]
    ]

    #place(top + left, dx: margin-left, dy: height - margin-bottom)[
      #box(width: plot-width, height: margin-bottom)[
        #align(center)[#text(size: 8pt)[#(x_label)]]
      ]
    ]

    #let y-label-x = margin-left - y-label-gap
    #let y-label-y = margin-top + plot-height / 2
    #place(right + horizon, dx: y-label-x - width, dy: y-label-y - height / 2)[
      #rotate(-90deg)[#text(size: 8pt)[#(y_label)]]
    ]
  ]
}

#let parse-metric(path, key) = {
  let text = read(path)
  let lines = text.split("\n")
  let points = lines.map(line => {
    if line.contains("step") and line.contains("/") and line.contains(key) {
      let step-part = line.split("step").at(1).trim()
      let step-str = step-part.split("/").at(0).trim()
      let metric-part = line.split(key).at(1).trim()
      let metric-str = metric-part.split(" ").at(0)
      (int(step-str), float(metric-str))
    } else { none }
  })
  points.filter(p => p != none)
}

#let ratio-series(base, other) = {
  let vals = other.map(p => {
    let x = p.at(0)
    let match = base.filter(q => q.at(0) == x)
    if match.len() > 0 and match.at(0).at(1) != 0.0 {
      (x, p.at(1) / match.at(0).at(1))
    } else { none }
  })
  vals.filter(v => v != none)
}

#let delta-series(base, other) = {
  let vals = other.map(p => {
    let x = p.at(0)
    let match = base.filter(q => q.at(0) == x)
    if match.len() > 0 {
      (x, p.at(1) - match.at(0).at(1))
    } else { none }
  })
  vals.filter(v => v != none)
}

#let end-step(data) = calc.max(..data.map(d => d.at(0)))

#let giant-h6-loss = parse-metric("Training_logs/head_sweep/giant_h6_mha_logs.txt", "loss ")
#let giant-h18-loss = parse-metric("Training_logs/head_sweep/giant_h18_mha_logs.txt", "loss ")
#let giant-gqa-loss = parse-metric("Training_logs/head_sweep/giant_h18_kv6_gqa_logs.txt", "loss ")
#let giant-h6hd32-loss = parse-metric("Training_logs/head_sweep/giant_h6_hd32_mha_logs.txt", "loss ")
#let giant-loss-delta-h6 = giant-h6-loss.map(p => (p.at(0), 0.0))
#let giant-loss-delta-h18 = delta-series(giant-h6-loss, giant-h18-loss)
#let giant-loss-delta-gqa = delta-series(giant-h6-loss, giant-gqa-loss)
#let giant-loss-delta-h6hd32 = delta-series(giant-h6-loss, giant-h6hd32-loss)

#let tidar-h6-ar = parse-metric("Training_logs/head_sweep/tidar_h6_mha_logs.txt", " ar ")
#let tidar-h18-ar = parse-metric("Training_logs/head_sweep/tidar_h18_mha_logs.txt", " ar ")
#let tidar-gqa-ar = parse-metric("Training_logs/head_sweep/tidar_h18_kv6_gqa_logs.txt", " ar ")
#let tidar-h6hd32-ar = parse-metric("Training_logs/head_sweep/tidar_h6_hd32_mha_logs.txt", " ar ")

#let tidar-h6-diff = parse-metric("Training_logs/head_sweep/tidar_h6_mha_logs.txt", " diff ")
#let tidar-h18-diff = parse-metric("Training_logs/head_sweep/tidar_h18_mha_logs.txt", " diff ")
#let tidar-gqa-diff = parse-metric("Training_logs/head_sweep/tidar_h18_kv6_gqa_logs.txt", " diff ")
#let tidar-h6hd32-diff = parse-metric("Training_logs/head_sweep/tidar_h6_hd32_mha_logs.txt", " diff ")

#let tidar-h6-greedy = parse-metric("Training_logs/head_sweep/tidar_h6_mha_logs.txt", "greedy_acc ")
#let tidar-h18-greedy = parse-metric("Training_logs/head_sweep/tidar_h18_mha_logs.txt", "greedy_acc ")
#let tidar-gqa-greedy = parse-metric("Training_logs/head_sweep/tidar_h18_kv6_gqa_logs.txt", "greedy_acc ")
#let tidar-h6hd32-greedy = parse-metric("Training_logs/head_sweep/tidar_h6_hd32_mha_logs.txt", "greedy_acc ")

#let diff-ratio-h6 = tidar-h6-diff.map(p => (p.at(0), 1.0))
#let diff-ratio-h18 = ratio-series(tidar-h6-diff, tidar-h18-diff)
#let diff-ratio-gqa = ratio-series(tidar-h6-diff, tidar-gqa-diff)
#let diff-ratio-h6hd32 = ratio-series(tidar-h6-diff, tidar-h6hd32-diff)

#align(center)[
  #text(size: 16pt, weight: "bold")[TinyStories Head Sweep: GIANT vs TiDAR]
]

#v(6pt)

#text(size: 9pt, fill: rgb(80, 80, 80))[
  Runs stopped manually after about 1.5h. Each curve is plotted up to its last logged step.
]

#v(6pt)

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  legend-item("h6 MHA (baseline)", c-h6),
  legend-item("h18 MHA", c-h18),
  legend-item("h18 with kv6 GQA", c-gqa),
  legend-item("h6 MHA (head_dim=32 control)", c-h6hd32),
)

#v(8pt)

== GIANT training loss

#text(size: 9pt, fill: rgb(60, 60, 60))[Interpretation: lower is better.]

#multi-line-chart(
  (
    (name: "h6 MHA", color: c-h6, data: giant-h6-loss),
    (name: "h18 MHA", color: c-h18, data: giant-h18-loss),
    (name: "h18 kv6 GQA", color: c-gqa, data: giant-gqa-loss),
    (name: "h6 MHA hd32", color: c-h6hd32, data: giant-h6hd32-loss),
  ),
  y_label: "Loss",
  y_ticks: 6,
  y_tick_decimals: 2,
)

#v(4pt)
#text(size: 9pt)[
  Logged points: h6 #giant-h6-loss.len(), h18 #giant-h18-loss.len(), gqa #giant-gqa-loss.len(), h6hd32 #giant-h6hd32-loss.len().
  End steps: h6 #end-step(giant-h6-loss), h18 #end-step(giant-h18-loss), gqa #end-step(giant-gqa-loss), h6hd32 #end-step(giant-h6hd32-loss).
]

#v(10pt)

== GIANT loss delta vs baseline

#text(size: 9pt, fill: rgb(60, 60, 60))[
  Interpretation: baseline is 0.00. Below 0.00 is better than baseline, above 0.00 is worse.
]

#multi-line-chart(
  (
    (name: "h6 MHA baseline", color: c-h6, data: giant-loss-delta-h6),
    (name: "h18 MHA - h6", color: c-h18, data: giant-loss-delta-h18),
    (name: "h18 kv6 GQA - h6", color: c-gqa, data: giant-loss-delta-gqa),
    (name: "h6 hd32 MHA - h6", color: c-h6hd32, data: giant-loss-delta-h6hd32),
  ),
  y_label: "Loss delta vs h6",
  y_ticks: 6,
  y_tick_decimals: 3,
  baseline_y: 0.0,
)

#v(10pt)

== TiDAR AR loss

#text(size: 9pt, fill: rgb(60, 60, 60))[Interpretation: lower is better.]

#multi-line-chart(
  (
    (name: "h6 MHA", color: c-h6, data: tidar-h6-ar),
    (name: "h18 MHA", color: c-h18, data: tidar-h18-ar),
    (name: "h18 kv6 GQA", color: c-gqa, data: tidar-gqa-ar),
    (name: "h6 MHA hd32", color: c-h6hd32, data: tidar-h6hd32-ar),
  ),
  y_label: "AR loss",
  y_ticks: 6,
  y_tick_decimals: 3,
)

#v(8pt)

== TiDAR Diff loss

#text(size: 9pt, fill: rgb(60, 60, 60))[Interpretation: lower is better.]

#multi-line-chart(
  (
    (name: "h6 MHA", color: c-h6, data: tidar-h6-diff),
    (name: "h18 MHA", color: c-h18, data: tidar-h18-diff),
    (name: "h18 kv6 GQA", color: c-gqa, data: tidar-gqa-diff),
    (name: "h6 MHA hd32", color: c-h6hd32, data: tidar-h6hd32-diff),
  ),
  y_label: "Diff loss",
  y_ticks: 6,
  y_tick_decimals: 3,
)

#pagebreak()

== TiDAR Greedy acceptance

#text(size: 9pt)[Greedy acceptance = fraction of token positions where argmax(AR) equals argmax(Diff).]
#text(size: 9pt, fill: rgb(60, 60, 60))[Interpretation: higher is better.]

#multi-line-chart(
  (
    (name: "h6 MHA", color: c-h6, data: tidar-h6-greedy),
    (name: "h18 MHA", color: c-h18, data: tidar-h18-greedy),
    (name: "h18 kv6 GQA", color: c-gqa, data: tidar-gqa-greedy),
    (name: "h6 MHA hd32", color: c-h6hd32, data: tidar-h6hd32-greedy),
  ),
  y_label: "Greedy acceptance",
  y_ticks: 6,
  y_tick_decimals: 3,
)

#v(10pt)

== TiDAR Diff delta vs baseline (normalized)

#text(size: 9pt)[
  Baseline is h6 MHA fixed at 1.00. Other lines are ratio = Diff(loss_run) / Diff(loss_baseline) at matching steps.
]
#text(size: 9pt, fill: rgb(60, 60, 60))[
  Interpretation: 1.00 matches baseline; below 1.00 is better than baseline; above 1.00 is worse than baseline.
]

#multi-line-chart(
  (
    (name: "h6 MHA = 1.00", color: c-h6, data: diff-ratio-h6),
    (name: "h18 MHA / baseline", color: c-h18, data: diff-ratio-h18),
    (name: "h18 kv6 GQA / baseline", color: c-gqa, data: diff-ratio-gqa),
    (name: "h6 hd32 / baseline", color: c-h6hd32, data: diff-ratio-h6hd32),
  ),
  y_label: "Diff ratio vs h6 baseline",
  y_ticks: 6,
  y_tick_decimals: 3,
  baseline_y: 1.0,
)

#v(6pt)
#text(size: 9pt)[
  Ratio points: h18 #diff-ratio-h18.len(), gqa #diff-ratio-gqa.len(), h6hd32 #diff-ratio-h6hd32.len().
]

#v(10pt)

== Appendix: Updated sweep interpretation

#text(size: 10pt, weight: "bold")[GIANT summary (full 54k-step epoch)]
#text(size: 9pt)[
  Runtime vs h6 baseline (1797.2s): h18 MHA = +3.66% (1863.0s), h18 kv6 GQA = -3.10% (1741.4s), h6 hd32 = -63.0% (665.0s).
]
#text(size: 9pt)[
  Final GIANT loss at step 54k: h6 = 1.1174, h18 = 1.1297, h18 kv6 GQA = 1.1423, h6 hd32 = 1.4661.
]

#v(4pt)
#text(size: 10pt, weight: "bold")[TiDAR summary (timed runs)]
#text(size: 9pt)[
  Reached steps: h6 = 32.4k, h18 = 29.6k, h18 kv6 GQA = 30.0k, h6 hd32 = 42.8k.
]
#text(size: 9pt)[
  Diff ratio vs h6 baseline (mean over matched steps): h18 = 0.9966, h18 kv6 GQA = 1.0039, h6 hd32 = 1.1425.
]

#v(4pt)
#text(size: 10pt, weight: "bold")[Short interpretation]
#text(size: 9pt)[
  Within the comparable-width models (h6/h18/GQA at d_model=576), more heads gave little quality gain for the extra cost:
  h18 is slightly slower, and GQA is near-parity on quality with modest speed benefit. The h6 hd32 control buys much higher throughput,
  but quality drops clearly; it is a compute-for-quality trade, not a strict heads-only comparison.
]
