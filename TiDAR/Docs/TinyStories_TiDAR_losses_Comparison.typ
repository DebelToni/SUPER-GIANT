#set page(width: 210mm, height: 297mm, margin: 18mm)
#set text(size: 11pt)
#set heading(numbering: none)

// Colors for the 4 runs
#let stable-color = rgb(33, 150, 243)      // Blue
#let kl-only-color = rgb(244, 67, 54)      // Red
#let kl-keep-color = rgb(76, 175, 80)      // Green
#let distill-color = rgb(156, 39, 176)     // Purple
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
  cut_x: none,
  cut_label: none,
  baseline_y: none,
) = {
  let margin-left = 22pt
  let margin-right = 8pt
  let margin-top = 6pt
  let margin-bottom = 18pt
  let y-label-gap = 4pt
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

#let delta-greedy-kl-only = delta-series(stable-greedy-90, kl-only-greedy-90)
#let delta-greedy-kl-keep = delta-series(stable-greedy-90, kl-keep-greedy-90)
#let delta-greedy-distill = delta-series(stable-greedy-90, distill-greedy-90)
#let delta-greedy-all = delta-greedy-kl-only + delta-greedy-kl-keep + delta-greedy-distill
#let delta-greedy-min = calc.min(..delta-greedy-all.map(d => d.at(1)))
#let delta-greedy-max = calc.max(..delta-greedy-all.map(d => d.at(1)))
#let delta-greedy-pad = (delta-greedy-max - delta-greedy-min) * 0.2
#let delta-greedy-y-min = delta-greedy-min - delta-greedy-pad
#let delta-greedy-y-max = delta-greedy-max + delta-greedy-pad

// ============================================================================
// DOCUMENT
// ============================================================================

= TinyStories TiDAR Greedy Acceptance Comparison

== Run Summary

All 4 runs branch from the stable run at step 90,000:

- *Stable (blue)*: `alpha=1, beta=1` -- baseline AR+Diff training, continued to 121k
- *KL-only (red)*: `alpha=0.2, beta=0, rho=0.5, delta=0.3` -- forward KL + greedy, no diff loss
- *KL-keep (green)*: `alpha=1, beta=1, rho=0.1, delta=0.3` -- forward KL + greedy, keeps AR+Diff
- *Distill (purple)*: `alpha=1, beta=1, eta=0.04, T=2` -- soft distillation AR->Diff

== Legend

#box(width: 100%, inset: 8pt, stroke: 0.6pt + rgb(200, 200, 200), radius: 4pt)[
  #grid(
    columns: (1fr, 1fr),
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
  )
]

== AR loss (raw)

#text(size: 9pt, weight: "bold")[Stable 0-121k]
#multi-line-chart(
  ((color: stable-color, data: stable-ar),),
  width: 170mm,
  height: 45mm,
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
  height: 45mm,
  y_label: "AR loss",
  x_min: branch-x-min,
  x_max: branch-x-max,
  y_min: ar-zoom-y-min,
  y_max: ar-zoom-y-max,
)

=== Diff loss (raw)

#text(size: 9pt, weight: "bold")[Stable 0-121k]
#multi-line-chart(
  ((color: stable-color, data: stable-diff),),
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

#text(size: 9pt, weight: "bold")[Stable 0-121k]
#multi-line-chart(
  ((color: stable-color, data: stable-greedy),),
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
  height: 30mm,
  y_label: "Delta",
  x_min: branch-x-min,
  x_max: branch-x-max,
  y_min: delta-greedy-y-min,
  y_max: delta-greedy-y-max,
  baseline_y: 0.0,
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
)
