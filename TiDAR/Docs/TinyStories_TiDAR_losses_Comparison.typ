#set page(width: 210mm, height: 297mm, margin: 18mm)
#set text(size: 11pt)
#set heading(numbering: none)

// Colors for the runs
#let stable-color = rgb(33, 150, 243)      // Blue
#let kl-only-color = rgb(244, 67, 54)      // Red
#let kl-keep-color = rgb(76, 175, 80)      // Green
#let distill-color = rgb(156, 39, 176)     // Purple
#let smallar-color = rgb(255, 152, 0)      // Orange
#let biggerbeta-color = rgb(0, 150, 136)   // Teal
#let topk-color = rgb(121, 85, 72)         // Brown
#let biggamma-color = rgb(63, 81, 181)     // Indigo
#let maskedlater-color = rgb(233, 30, 99)  // Magenta
#let deltamasked-color = rgb(0, 188, 212)  // Cyan
#let border-color = rgb(180, 180, 180)

// Branch step
#let cut-step = 90000

// Legend item
#let legend-item(label, color) = grid(
  columns: (auto, auto),
  gutter: 2pt,
  align: left,
  rect(width: 10pt, height: 10pt, fill: color, radius: 2pt),
  text(size: 10pt)[#(label)],
)

// Multi-series line chart with fixed x bounds
#let multi-line-chart(
  series,
  width: 80mm,
  height: 40mm,
  x_label: "Step",
  y_label: "Metric",
  x_min: none,
  x_max: none,
  y_min: none,
  y_max: none,
  y_ticks: 0,
  y_tick_decimals: 2,
  y_tick_percent: false,
  cut_x: none,
  cut_label: none,
  baseline_y: none,
) = {
  let margin-left = 22pt
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
    #if cut_x != none [
      #let cut-pos = (cut_x - min-x) * scale-x
      #place(top + left, dx: margin-left + cut-pos, dy: margin-top)[
        #line(length: plot-height, angle: 90deg, stroke: (paint: rgb(160, 160, 160), thickness: 0.8pt, dash: (2pt, 2pt)))
      ]
      #if cut_label != none [
        #place(top + left, dx: margin-left + cut-pos + 2pt, dy: margin-top + 2pt)[
          #text(size: 7pt, fill: rgb(90, 90, 90))[#(cut_label)]
        ]
      ]
    ]
    #if baseline_y != none [
      #let base-pos = plot-height - (baseline_y - min-y) * scale-y
      #place(top + left, dx: margin-left, dy: margin-top + base-pos)[
        #line(length: plot-width, stroke: (paint: rgb(170, 170, 170), thickness: 0.6pt, dash: (2pt, 2pt)))
      ]
    ]
    #if y_ticks > 0 [
      #for i in range(0, y_ticks + 1) [
        #let yv = min-y + (max-y - min-y) * i / y_ticks
        #let py = plot-height - (yv - min-y) * scale-y
        #let yshow = if y_tick_percent { yv * 100 } else { yv }
        #let ytxt = str(calc.round(yshow, digits: y_tick_decimals))
        #place(top + left, dx: margin-left - 2pt, dy: margin-top + py)[
          #line(length: 2pt, stroke: 0.5pt + rgb(120, 120, 120))
        ]
        #place(top + left, dx: 0pt, dy: margin-top + py - 4pt)[
          #box(width: margin-left - 4pt)[
            #align(right)[
              #text(size: 6.6pt, fill: rgb(90, 90, 90))[
                #if y_tick_percent [#ytxt%] else [#ytxt]
              ]
            ]
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

// Log parsing (generic metric)
#let parse-metric(path, key) = {
  let text = read(path)
  let lines = text.split("\n")
  let points = lines.map(line => {
    if line.contains(key) and line.contains("step") {
      let step_part = line.split("step").at(1).trim()
      let step_str = step_part.split("/").at(0).trim()
      let metric_part = line.split(key).at(1).trim()
      let metric_str = metric_part.split(" ").at(0)
      (int(step_str), float(metric_str))
    } else { none }
  })
  points.filter(p => p != none)
}

#let filter-range(data, min: none, max: none) = data.filter(p =>
  ((min == none) or (p.at(0) >= min)) and ((max == none) or (p.at(0) <= max))
)

#let delta-series(base, other) = {
  let deltas = other.map(p => {
    let x = p.at(0)
    let match = base.filter(q => q.at(0) == x)
    if match.len() > 0 { (x, p.at(1) - match.at(0).at(1)) } else { none }
  })
  deltas.filter(d => d != none)
}

// Load logs from Training_logs
// AR loss
#let stable-ar = parse-metric("Training_logs/tidar_stable_logs.txt", " ar ")
#let kl-only-ar = parse-metric("Training_logs/tidar_0p2_0_0p5_0_0p3_logs.txt", " ar ")
#let kl-keep-ar = parse-metric("Training_logs/tidar_1_1_0p1_0_0p3_logs.txt", " ar ")
#let distill-ar = parse-metric("Training_logs/tidar_eta0p04_t2_logs.txt", " ar ")

// Diff loss
#let stable-diff = parse-metric("Training_logs/tidar_stable_logs.txt", " diff ")
#let kl-only-diff = parse-metric("Training_logs/tidar_0p2_0_0p5_0_0p3_logs.txt", " diff ")
#let kl-keep-diff = parse-metric("Training_logs/tidar_1_1_0p1_0_0p3_logs.txt", " diff ")
#let distill-diff = parse-metric("Training_logs/tidar_eta0p04_t2_logs.txt", " diff ")

// Greedy acceptance
#let stable-greedy = parse-metric("Training_logs/tidar_stable_logs.txt", "greedy_acc ")
#let kl-only-greedy = parse-metric("Training_logs/tidar_0p2_0_0p5_0_0p3_logs.txt", "greedy_acc ")
#let kl-keep-greedy = parse-metric("Training_logs/tidar_1_1_0p1_0_0p3_logs.txt", "greedy_acc ")
#let distill-greedy = parse-metric("Training_logs/tidar_eta0p04_t2_logs.txt", "greedy_acc ")
#let smallar-greedy = parse-metric("Training_logs/tidar_smallAR_big_greedy_eta_logs.txt", "greedy_acc ")
#let biggerbeta-greedy = parse-metric("Training_logs/tidar_stable_bigger_beta_logs.txt", "greedy_acc ")
#let topk-greedy = parse-metric("Training_logs/tidar_later_stage_topk_logs.txt", "greedy_acc ")
#let biggamma-greedy = parse-metric("Training_logs/tidar_bigGamma_andDelta_logs.txt", "greedy_acc ")
#let maskedlater-greedy = parse-metric("Training_logs/tidar_masked_delta_later_logs.txt", "greedy_acc ")

// Early masked-delta run (short run, <30k)
#let deltamasked-diff = parse-metric("Training_logs/tidar_delta_masked_logs.txt", " diff ")
#let deltamasked-greedy = parse-metric("Training_logs/tidar_delta_masked_logs.txt", "greedy_acc ")

// 90k+ ranges
#let stable-ar-90 = filter-range(stable-ar, min: cut-step)
#let kl-only-ar-90 = filter-range(kl-only-ar, min: cut-step)
#let kl-keep-ar-90 = filter-range(kl-keep-ar, min: cut-step)
#let distill-ar-90 = filter-range(distill-ar, min: cut-step)

#let stable-diff-90 = filter-range(stable-diff, min: cut-step)
#let kl-only-diff-90 = filter-range(kl-only-diff, min: cut-step)
#let kl-keep-diff-90 = filter-range(kl-keep-diff, min: cut-step)
#let distill-diff-90 = filter-range(distill-diff, min: cut-step)

#let stable-greedy-90 = filter-range(stable-greedy, min: cut-step)
#let kl-only-greedy-90 = filter-range(kl-only-greedy, min: cut-step)
#let kl-keep-greedy-90 = filter-range(kl-keep-greedy, min: cut-step)
#let distill-greedy-90 = filter-range(distill-greedy, min: cut-step)
#let smallar-greedy-90 = filter-range(smallar-greedy, min: cut-step)
#let biggerbeta-greedy-90 = filter-range(biggerbeta-greedy, min: cut-step)
#let topk-greedy-90 = filter-range(topk-greedy, min: cut-step)
#let biggamma-greedy-90 = filter-range(biggamma-greedy, min: cut-step)
#let maskedlater-greedy-90 = filter-range(maskedlater-greedy, min: cut-step)

#let stable-x-min = calc.min(..stable-greedy.map(d => d.at(0)))
#let stable-x-max = calc.max(..stable-greedy.map(d => d.at(0)))
#let branch-x-min = cut-step
#let branch-x-max = stable-x-max

// AR y-range (zoomed 90k+)
#let ar-zoom-all = stable-ar-90 + kl-only-ar-90 + kl-keep-ar-90 + distill-ar-90
#let ar-zoom-min = calc.min(..ar-zoom-all.map(d => d.at(1)))
#let ar-zoom-max = calc.max(..ar-zoom-all.map(d => d.at(1)))
#let ar-zoom-pad = (ar-zoom-max - ar-zoom-min) * 0.1
#let ar-zoom-y-min = ar-zoom-min - ar-zoom-pad
#let ar-zoom-y-max = ar-zoom-max + ar-zoom-pad

// Diff y-range (zoomed 90k+, exclude KL-only zeros)
#let diff-zoom-all = stable-diff-90 + kl-keep-diff-90 + distill-diff-90
#let diff-zoom-min = calc.min(..diff-zoom-all.map(d => d.at(1)))
#let diff-zoom-max = calc.max(..diff-zoom-all.map(d => d.at(1)))
#let diff-zoom-pad = (diff-zoom-max - diff-zoom-min) * 0.1
#let diff-zoom-y-min = diff-zoom-min - diff-zoom-pad
#let diff-zoom-y-max = diff-zoom-max + diff-zoom-pad

// Greedy y-range
#let greedy-all = stable-greedy + kl-only-greedy + kl-keep-greedy + distill-greedy
#let greedy-min = calc.min(..greedy-all.map(d => d.at(1)))
#let greedy-max = calc.max(..greedy-all.map(d => d.at(1)))
#let greedy-pad = (greedy-max - greedy-min) * 0.1
#let greedy-y-min = greedy-min - greedy-pad
#let greedy-y-max = greedy-max + greedy-pad

// Greedy y-range (extended with smallAR big greedy eta + stable bigger beta, 90k+ zoom)
#let greedy-all-ext-90 = stable-greedy-90 + kl-only-greedy-90 + kl-keep-greedy-90 + distill-greedy-90 + smallar-greedy-90 + biggerbeta-greedy-90 + topk-greedy-90 + biggamma-greedy-90 + maskedlater-greedy-90
#let greedy-ext-min = calc.min(..greedy-all-ext-90.map(d => d.at(1)))
#let greedy-ext-max = calc.max(..greedy-all-ext-90.map(d => d.at(1)))
#let greedy-ext-pad = (greedy-ext-max - greedy-ext-min) * 0.01
#let greedy-ext-y-min = greedy-ext-min - greedy-ext-pad
#let greedy-ext-y-max = greedy-ext-max + greedy-ext-pad

#let delta-greedy-kl-only = delta-series(stable-greedy-90, kl-only-greedy-90)
#let delta-greedy-kl-keep = delta-series(stable-greedy-90, kl-keep-greedy-90)
#let delta-greedy-distill = delta-series(stable-greedy-90, distill-greedy-90)
#let delta-greedy-smallar = delta-series(stable-greedy-90, smallar-greedy-90)
#let delta-greedy-biggerbeta = delta-series(stable-greedy-90, biggerbeta-greedy-90)
#let delta-greedy-topk = delta-series(stable-greedy-90, topk-greedy-90)
#let delta-greedy-biggamma = delta-series(stable-greedy-90, biggamma-greedy-90)
#let delta-greedy-maskedlater = delta-series(stable-greedy-90, maskedlater-greedy-90)
#let delta-greedy-all = delta-greedy-kl-only + delta-greedy-kl-keep + delta-greedy-distill
#let delta-greedy-min = calc.min(..delta-greedy-all.map(d => d.at(1)))
#let delta-greedy-max = calc.max(..delta-greedy-all.map(d => d.at(1)))
#let delta-greedy-pad = (delta-greedy-max - delta-greedy-min) * 0.2
#let delta-greedy-y-min = delta-greedy-min - delta-greedy-pad
#let delta-greedy-y-max = delta-greedy-max + delta-greedy-pad

#let delta-greedy-all-ext = delta-greedy-all + delta-greedy-smallar + delta-greedy-biggerbeta + delta-greedy-topk + delta-greedy-biggamma + delta-greedy-maskedlater
#let delta-greedy-ext-min = calc.min(..delta-greedy-all-ext.map(d => d.at(1)))
#let delta-greedy-ext-max = calc.max(..delta-greedy-all-ext.map(d => d.at(1)))
#let delta-greedy-ext-pad = (delta-greedy-ext-max - delta-greedy-ext-min) * 0.01
#let delta-greedy-ext-y-min = delta-greedy-ext-min - delta-greedy-ext-pad
#let delta-greedy-ext-y-max = delta-greedy-ext-max + delta-greedy-ext-pad

// ============================================================================
// DOCUMENT
// ============================================================================

= TinyStories TiDAR Greedy Acceptance Comparison

90k+ branch runs from stable (plus one short early run overlay):

- *Stable (blue)*: `alpha=1, beta=1` -- baseline AR+Diff training, continued to 121k
- *KL-only (red)*: `alpha=0.2, beta=0, rho=0.5, delta=0.3` -- forward KL + greedy, no diff loss
- *KL-keep (green)*: `alpha=1, beta=1, rho=0.1, delta=0.3` -- forward KL + greedy, keeps AR+Diff
- *Distill (purple)*: `alpha=1, beta=1, eta=0.04, T=2` -- soft distillation AR->Diff
- *Greedy Eta (orange)*: `alpha,beta=0.1, delta=3.0, eta=1.0, T=0.7` -- aggressive agreement + distill
- *Stable bigger beta (teal)*: `alpha=1, beta=5` -- baseline with 5x diffusion weight
- *Top-K set (brown)*: `gamma=0.01, gamma_topk=8` -- top-k set distillation (later-stage)
- *Big Gamma+Delta (indigo)*: `gamma=5, gamma_topk=4, delta=1` -- strong top-k + hard agreement
- *Masked Delta later (magenta)*: partial 90k+ run, plotted up to available checkpoints
#text(size: 12pt, weight: "bold", fill: rgb("#DC2626"))[
> Това е точно идеята, която ми предложи - тествах я вчера.
]
- *Delta masked early (cyan)*: short run (under 30k), overlaid on stable full-range Diff/Greedy charts

== Legend

#box(width: 100%, inset: 8pt, stroke: 0.6pt + rgb(200, 200, 200), radius: 4pt)[
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr),
    gutter: 12pt,
    [
      #legend-item("Stable (1, 1, 0, 0, 0, 0)", stable-color)
      #v(4pt)
      #legend-item("KL-only (0.2, 0, 0.5, 0, 0.3, 0)", kl-only-color)
    ],
    [
      #legend-item("KL-keep (1, 1, 0.1, 0, 0.3, 0)", kl-keep-color)
      #v(4pt)
      #legend-item("Distill (1, 1, 0, 0, 0, eta=0.04 T=2)", distill-color)
    ],
    [
      #legend-item("Greedy eta (0.1, 0.1, 0, 0, 3.0, eta=1.0 T=0.7)", smallar-color)
      #v(4pt)
      #legend-item("Stable bigger beta (1, 5, 0, 0, 0, 0)", biggerbeta-color)
    ],
    [
      #legend-item("Top-K set (gamma=0.01 K=8)", topk-color)
      #v(4pt)
      #legend-item("Big Gamma+Delta (gamma=5 K=4 delta=1)", biggamma-color)
    ],
    [
      #legend-item("Masked Delta later (90k+ partial)", maskedlater-color)
      #v(4pt)
      #legend-item("Delta masked early (under 30k)", deltamasked-color)
    ],
  )
]

=== Total cost of the experiment so far: 30.25\$

== AR loss (raw)

#text(size: 9pt, weight: "bold")[Stable 0-121k]
#multi-line-chart(
  ((color: stable-color, data: stable-ar),),
  width: 170mm,
  height: 65mm,
  y_label: "AR loss",
  cut_x: cut-step,
  cut_label: "90k",
  x_min: stable-x-min,
  x_max: branch-x-max,
)

#v(4pt)

#text(size: 9pt, weight: "bold")[All runs 90k-121k]
#multi-line-chart(
  (
    (color: stable-color, data: stable-ar-90),
    (color: kl-only-color, data: kl-only-ar-90),
    (color: kl-keep-color, data: kl-keep-ar-90),
    (color: distill-color, data: distill-ar-90),
  ),
  width: 170mm,
  height: 50mm,
  y_label: "AR loss",
  x_min: branch-x-min,
  x_max: branch-x-max,
  y_min: ar-zoom-y-min,
  y_max: ar-zoom-y-max,
)

=== Diff loss (raw)

#text(size: 9pt, weight: "bold")[Stable 0-121k + Delta masked early]
#multi-line-chart(
  (
    (color: stable-color, data: stable-diff),
    (color: deltamasked-color, data: deltamasked-diff),
  ),
  width: 170mm,
  height: 45mm,
  y_label: "Diff loss",
  cut_x: cut-step,
  cut_label: "90k",
  x_min: stable-x-min,
  x_max: branch-x-max,
)

#v(4pt)

#text(size: 9pt, weight: "bold")[All runs 90k-121k]
#multi-line-chart(
  (
    (color: stable-color, data: stable-diff-90),
    (color: kl-keep-color, data: kl-keep-diff-90),
    (color: distill-color, data: distill-diff-90),
  ),
  width: 170mm,
  height: 45mm,
  y_label: "Diff loss",
  x_min: branch-x-min,
  x_max: branch-x-max,
  y_min: diff-zoom-y-min,
  y_max: diff-zoom-y-max,
)

#text(size: 9pt)[KL-only diff=0 after 90k, excluded from the zoomed panel.]

== Greedy acceptance

#text(size: 9pt)[Greedy acceptance = fraction of positions where argmax(AR) == argmax(Diff).]

#text(size: 9pt, weight: "bold")[Stable 0-121k + Delta masked early]
#multi-line-chart(
  (
    (color: stable-color, data: stable-greedy),
    (color: deltamasked-color, data: deltamasked-greedy),
  ),
  width: 170mm,
  height: 45mm,
  y_label: "Greedy acc",
  cut_x: cut-step,
  cut_label: "90k",
  x_min: stable-x-min,
  x_max: branch-x-max,
  y_min: greedy-y-min,
  y_max: greedy-y-max,
)

#v(4pt)

#text(size: 9pt, weight: "bold")[All runs 90k-121k]
#multi-line-chart(
  (
    (color: stable-color, data: stable-greedy-90),
    (color: kl-only-color, data: kl-only-greedy-90),
    (color: kl-keep-color, data: kl-keep-greedy-90),
    (color: distill-color, data: distill-greedy-90),
  ),
  width: 170mm,
  height: 45mm,
  y_label: "Greedy acc",
  x_min: branch-x-min,
  x_max: branch-x-max,
  y_min: greedy-y-min,
  y_max: greedy-y-max,
)

#v(4pt)

#text(size: 9pt, weight: "bold")[Delta vs stable (90k-121k)]
#multi-line-chart(
  (
    (color: kl-only-color, data: delta-greedy-kl-only),
    (color: kl-keep-color, data: delta-greedy-kl-keep),
    (color: distill-color, data: delta-greedy-distill),
  ),
  width: 170mm,
  height: 45mm,
  y_label: "Delta",
  x_min: branch-x-min,
  x_max: branch-x-max,
  y_min: delta-greedy-y-min,
  y_max: delta-greedy-y-max,
  baseline_y: 0.0,
)

#v(8pt)

#text(size: 9pt, weight: "bold")[All runs 90k-121k (incl. smallAR + bigger beta + top-k + big gamma+delta + masked delta later)]
#multi-line-chart(
  (
    (color: stable-color, data: stable-greedy-90),
    (color: kl-only-color, data: kl-only-greedy-90),
    (color: kl-keep-color, data: kl-keep-greedy-90),
    (color: distill-color, data: distill-greedy-90),
    (color: smallar-color, data: smallar-greedy-90),
    (color: biggerbeta-color, data: biggerbeta-greedy-90),
    (color: topk-color, data: topk-greedy-90),
    (color: biggamma-color, data: biggamma-greedy-90),
    (color: maskedlater-color, data: maskedlater-greedy-90),
  ),
  width: 170mm,
  height: 45mm,
  y_label: "Greedy acc",
  x_min: branch-x-min,
  x_max: branch-x-max,
  y_min: greedy-ext-y-min,
  y_max: greedy-ext-y-max,
)

#v(4pt)

#text(size: 9pt, weight: "bold")[Delta vs stable (90k-121k) (incl. smallAR + bigger beta + top-k + big gamma+delta + masked delta later)]
#multi-line-chart(
  (
    (color: kl-only-color, data: delta-greedy-kl-only),
    (color: kl-keep-color, data: delta-greedy-kl-keep),
    (color: distill-color, data: delta-greedy-distill),
    (color: smallar-color, data: delta-greedy-smallar),
    (color: biggerbeta-color, data: delta-greedy-biggerbeta),
    (color: topk-color, data: delta-greedy-topk),
    (color: biggamma-color, data: delta-greedy-biggamma),
    (color: maskedlater-color, data: delta-greedy-maskedlater),
  ),
  width: 170mm,
  height: 60mm,
  y_label: "Delta",
  x_min: branch-x-min,
  x_max: branch-x-max,
  y_min: delta-greedy-ext-y-min,
  y_max: delta-greedy-ext-y-max,
  baseline_y: 0.0,
)

#v(4pt)

#text(size: 9pt, weight: "bold")[Masked Delta later focus vs Stable (90k-121k, partial run)]
#multi-line-chart(
  (
    (color: stable-color, data: stable-greedy-90),
    (color: maskedlater-color, data: maskedlater-greedy-90),
  ),
  width: 170mm,
  height: 45mm,
  y_label: "Greedy acc",
  x_min: branch-x-min,
  x_max: branch-x-max,
  y_min: greedy-ext-y-min,
  y_max: greedy-ext-y-max,
)

#pagebreak()

== Inference Results

Greedy decoding (`temperature=0`) with draft_len=6.

Prompts (10):
- P0: "There was a small village"
- P1: "Once upon a time, a little girl"
- P2: "The boy was very happy because"
- P3: "In the forest, a tiny fox"
- P4: "The cat and the dog were"
- P5: "One day, the teacher said"
- P6: "After school, the kids went"
- P7: "The robot wanted to learn"
- P8: "On a sunny morning, Emma"
- P9: "The brave knight looked at"

=== Accept/Iter summary (mean / best / worst across prompts)

#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, center, center),
  [Checkpoint], [Steps=50], [Steps=100], [Steps=300],
  [Stable \@90k], [2.41 / 3.50 / 1.63], [2.26 / 2.75 / 1.71], [2.10 / 2.45 / 1.88],
  [Stable \@121k], [2.22 / 3.27 / 1.53], [2.09 / 2.68 / 1.60], [2.10 / 2.69 / 1.57],
  [KL-only \@121k], [2.42 / 3.27 / 1.81], [2.30 / 2.68 / 1.80], [2.08 / 2.49 / 1.61],
  [KL-keep \@121k], [2.40 / 3.27 / 1.75], [2.29 / 2.83 / 1.80], [2.18 / 2.90 / 1.75],
  [Distill \@121k], [2.31 / 3.50 / 1.53], [2.26 / 3.09 / 1.87], [2.19 / 2.74 / 1.80],
  [SmallAR big greedy eta \@105k], [2.15 / 2.88 / 1.75], [2.15 / 2.75 / 1.90], [2.00 / 2.43 / 1.74],
  [Stable bigger beta \@121.5k], [2.37 / 3.27 / 1.63], [2.33 / 3.09 / 1.62], [1.96 / 2.27 / 1.74],
  [Top-K set \@121k], [2.28 / 3.12 / 1.52], [2.18 / 2.56 / 1.72], [2.19 / 2.91 / 1.56],
  [Big Gamma+Delta \@105k], [2.32 / 3.33 / 1.67], [2.26 / 2.86 / 1.79], [2.09 / 2.59 / 1.71],
  [Masked Delta later \@105k], [2.30 / 3.12 / 1.72], [2.34 / 2.86 / 1.72], [2.01 / 2.40 / 1.67],
)

#pagebreak()

== Inference Logits Histogram

#let meta1 = (
  prompt: "Once upon",
  checkpoint_name: "step_0121566.npz",
  checkpoint_path: "/proj/giant-data/TiDAR/checkpoints/TinyStories_exp/tidar_stable/params/step_0121566.npz",
  target_position: 30,
  block_start: 28,
  block_end: 33,
  draft_len: 6,
  temperature: 0.0,
  top_k: 0,
  context_text: "One day, she saw a big, scary dog.",
  context_start: 23,
  context_end: 33,
)

#let data1 = (
  (title: "pos 28", labels: (" a", " an", " her", " some", " some~"), ar: (0.939, 0.032, 0.012, 0.008, 0.004), diff: (0.831, 0.021, 0.013, 0.006, 0.002), diff_top: " a", accept: true),
  (title: "pos 29", labels: (" big", " boy", " bird", " butt~", " shiny"), ar: (0.481, 0.041, 0.039, 0.035, 0.027), diff: (0.261, 0.019, 0.018, 0.031, 0.067), diff_top: " big", accept: true),
  (title: "pos 30", labels: (",", " tree", " slide", " dog", " red"), ar: (0.144, 0.135, 0.089, 0.066, 0.020), diff: (0.048, 0.045, 0.023, 0.021, 0.014), diff_top: ",", accept: true),
  (title: "pos 31", labels: (" scary", " red", " green", " brown", " white"), ar: (0.084, 0.074, 0.058, 0.046, 0.046), diff: (0.002, 0.005, 0.003, 0.001, 0.000), diff_top: " on", accept: false),
  (title: "pos 32", labels: (" the", " a", " top", " her", " an"), ar: (0.567, 0.395, 0.012, 0.009, 0.003), diff: (0.208, 0.080, 0.000, 0.087, 0.002), diff_top: " the", accept: false),
  (title: "pos 33", labels: (" grou~", " other", " slide", " swin~", " side"), ar: (0.577, 0.138, 0.037, 0.033, 0.028), diff: (0.565, 0.186, 0.023, 0.023, 0.031), diff_top: " grou~", accept: none),
)

#let meta2 = (
  prompt: "In the forest, a tiny fox",
  checkpoint_name: "step_0121566.npz",
  checkpoint_path: "/proj/giant-data/TiDAR/checkpoints/TinyStories_exp/tidar_stable/params/step_0121566.npz",
  target_position: 27,
  block_start: 24,
  block_end: 29,
  draft_len: 6,
  temperature: 0.0,
  top_k: 0,
  context_text: "Suddenly, he heard a loud noise.",
  context_start: 22,
  context_end: 29,
)

#let data2 = (
  (title: "pos 24", labels: (" he", " the", " a", " it", " some~"), ar: (0.868, 0.079, 0.034, 0.004, 0.003), diff: (0.829, 0.057, 0.030, 0.003, 0.004), diff_top: " he", accept: true),
  (title: "pos 25", labels: (" heard", " saw", " noti~", " spot~", " came"), ar: (0.405, 0.351, 0.072, 0.057, 0.019), diff: (0.300, 0.235, 0.069, 0.039, 0.012), diff_top: " heard", accept: true),
  (title: "pos 26", labels: (" a", " some~", " some~", " some", " the"), ar: (0.886, 0.048, 0.016, 0.015, 0.010), diff: (0.499, 0.085, 0.002, 0.002, 0.013), diff_top: " a", accept: true),
  (title: "pos 27", labels: (" loud", " voice", " noise", " stra~", " sound"), ar: (0.306, 0.234, 0.209, 0.045, 0.035), diff: (0.080, 0.036, 0.040, 0.016, 0.007), diff_top: " a", accept: false),
  (title: "pos 28", labels: (" voice", " loud", " noise", " stra~", " sound"), ar: (0.352, 0.187, 0.148, 0.045, 0.030), diff: (0.013, 0.014, 0.070, 0.003, 0.015), diff_top: ".", accept: false),
  (title: "pos 29", labels: (" He", " It", " The", " \"", " A"), ar: (0.434, 0.297, 0.122, 0.048, 0.034), diff: (0.439, 0.319, 0.115, 0.045, 0.028), diff_top: " He", accept: none),
)

#let meta3 = (
  prompt: "One day, the teacher said",
  checkpoint_name: "step_0121566.npz",
  checkpoint_path: "/proj/giant-data/TiDAR/checkpoints/TinyStories_exp/tidar_stable/params/step_0121566.npz",
  target_position: 29,
  block_start: 25,
  block_end: 30,
  draft_len: 6,
  temperature: 0.0,
  top_k: 0,
  context_text: "They ran to the park and saw a big slide.",
  context_start: 25,
  context_end: 35,
)

#let data3 = (
  (title: "pos 25", labels: (" They", " The", " When", " At", " As"), ar: (0.705, 0.112, 0.079, 0.018, 0.013), diff: (0.370, 0.070, 0.034, 0.015, 0.006), diff_top: " They", accept: true),
  (title: "pos 26", labels: (" ran", " all", " put", " quic~", " had"), ar: (0.241, 0.107, 0.094, 0.094, 0.042), diff: (0.157, 0.084, 0.031, 0.058, 0.012), diff_top: " ran", accept: true),
  (title: "pos 27", labels: (" to", " outs~", " arou~", " out", " and"), ar: (0.442, 0.185, 0.131, 0.067, 0.027), diff: (0.111, 0.022, 0.023, 0.019, 0.015), diff_top: " to", accept: true),
  (title: "pos 28", labels: (" the", " their", " get", " grab", " a"), ar: (0.904, 0.065, 0.021, 0.001, 0.001), diff: (0.134, 0.057, 0.003, 0.000, 0.006), diff_top: " the", accept: true),
  (title: "pos 29", labels: (" park", " play~", " gate", " bus", " swin~"), ar: (0.755, 0.126, 0.016, 0.013, 0.013), diff: (0.032, 0.009, 0.001, 0.002, 0.004), diff_top: " the", accept: false),
  (title: "pos 30", labels: (" park", " play~", " swin~", " gate", " big"), ar: (0.575, 0.250, 0.031, 0.015, 0.011), diff: (0.702, 0.161, 0.024, 0.009, 0.009), diff_top: " park", accept: none),
)

#let ar-hist-color = rgb(244, 67, 150)
#let diff-hist-color = rgb(33, 150, 243)
#let hist-width = 27mm
#let hist-height = 18mm
#let hist-label-height = 4mm
#let hist-gap = 0.8pt

#let hist-border(accept) = if accept == true { rgb(76, 175, 80) } else { rgb(190, 190, 190) }

#let hist-cell(labels, values, color, border) = {
  let n = labels.len()
  let inner-width = hist-width - 4pt
  let inner-height = hist-height - 4pt
  let plot-height = inner-height - hist-label-height - hist-gap
  let bar-width = (inner-width - hist-gap * (n - 1)) / n
  let plot-cells = range(0, n).map(i => {
    let value = values.at(i)
    let bar-height = plot-height * value
    align(bottom)[
      #rect(width: bar-width, height: bar-height, fill: color, radius: 1pt)
    ]
  })
  let label-cells = range(0, n).map(i => {
    let label = labels.at(i)
    box(width: bar-width, height: hist-label-height)[
      #align(center)[#text(size: 4.5pt)[#(label)]]
    ]
  })
  box(width: hist-width, height: hist-height, stroke: 0.6pt + border, radius: 2pt, inset: 2pt)[
    #grid(
      columns: (..range(0, n).map(i => bar-width)),
      rows: (plot-height, hist-label-height),
      gutter: hist-gap,
      align: center,
      ..plot-cells,
      ..label-cells,
    )
  ]
}

#let hist-panel(title, labels, values, color, border) = grid(
  columns: (auto),
  rows: (auto, auto),
  gutter: 2pt,
  align: center,
  text(size: 6pt, weight: "bold")[#(title)],
  hist-cell(labels, values, color, border),
)

#let diff-panel(labels, values, top1, border) = grid(
  columns: (auto),
  rows: (auto, auto),
  gutter: 2pt,
  align: center,
  hist-cell(labels, values, diff-hist-color, border),
  text(size: 5.5pt)[top1: #(top1)],
)

#let hist-columns(n) = (..range(0, n).map(_ => auto))

#let render-hist(meta, data) = [
  #text(size: 8pt)[Prompt: "#(meta.prompt)" | Checkpoint: "#(meta.checkpoint_name)"]
  #text(size: 8pt)[Target position: #(meta.target_position) | Block: #(meta.block_start)-#(meta.block_end) | Draft len: #(meta.draft_len)]

  #v(3pt)

  #text(size: 8pt, weight: "bold")[AR verify logits (top-5, sorted by AR)]
  #grid(
    columns: hist-columns(data.len()),
    gutter: 6pt,
    ..data.map(d => hist-panel(d.title, d.labels, d.ar, ar-hist-color, hist-border(d.accept))),
  )

  #v(3pt)

  #text(size: 8pt, weight: "bold")[Diff draft logits (same tokens, AR order)]
  #grid(
    columns: hist-columns(data.len()),
    gutter: 6pt,
    ..data.map(d => diff-panel(d.labels, d.diff, d.diff_top, hist-border(d.accept))),
  )

  #v(4pt)

  #text(size: 8pt, weight: "bold")[Sentence context]
  #text(size: 8pt)[#(meta.context_text)]
]

#render-hist(meta1, data1)

#v(6pt)

#render-hist(meta2, data2)

#v(6pt)

#render-hist(meta3, data3)
