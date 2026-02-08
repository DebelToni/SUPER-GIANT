#set page(width: 210mm, height: 297mm, margin: 14mm)
#set text(size: 10pt)
#set heading(numbering: none)

#let data_dir = "Benchmark_logs/tidar_vs_ar_single_forward_20260208_134219"
#let pairs = json(data_dir + "/best_tidar_vs_ar_by_model_prefill.json")
#let notes = read(data_dir + "/notes.txt")

#let model_order = ("100m", "500m", "1b", "2b", "3b")
#let prefill_order = (0, 1024, 4096)

#let color-ar = rgb("#1f77b4")
#let color-tidar = rgb("#d62728")
#let color-speed = rgb("#2ca02c")
#let color-grid = (rgb("#9ecae1"), rgb("#6baed6"), rgb("#3182bd"))

#let find-row(model, prefill) = {
  let cand = pairs.filter(r => r.model == model and int(r.prefill_tokens) == prefill)
  if cand.len() == 0 { none } else { cand.at(0) }
}

#let line_chart(prefill, width: 175mm, height: 58mm) = {
  let ml = 22pt
  let mr = 8pt
  let mt = 8pt
  let mb = 18pt
  let pw = width - ml - mr
  let ph = height - mt - mb

  let vals = ()
  for m in model_order {
    let r = find-row(m, prefill)
    if r != none {
      vals.push(float(r.ar_tps))
      vals.push(float(r.tidar_best_tps))
    }
  }
  let ymin = calc.min(..vals)
  let ymax = calc.max(..vals)
  let ypad = (ymax - ymin) * 0.08
  let ymin = ymin - ypad
  let ymax = ymax + ypad

  let ar_pts = ()
  let td_pts = ()
  for i in range(model_order.len()) {
    let m = model_order.at(i)
    let r = find-row(m, prefill)
    if r != none {
      let x = i * (pw / (model_order.len() - 1))
      let y_ar = ph - (float(r.ar_tps) - ymin) * ph / (ymax - ymin)
      let y_td = ph - (float(r.tidar_best_tps) - ymin) * ph / (ymax - ymin)
      ar_pts.push((x, y_ar))
      td_pts.push((x, y_td))
    }
  }

  box(width: width, height: height, stroke: 0.7pt + rgb("#c7c7c7"), radius: 6pt, inset: 4pt)[
    #place(top + left, dx: ml, dy: mt + ph)[#line(length: pw, stroke: 0.6pt + black)]
    #place(top + left, dx: ml, dy: mt)[#line(length: ph, angle: 90deg, stroke: 0.6pt + black)]

    #for i in range(model_order.len()) [
      #let x = i * (pw / (model_order.len() - 1))
      #place(top + left, dx: ml + x, dy: mt + ph)[#line(length: 2.5pt, angle: 90deg, stroke: 0.5pt + black)]
      #place(top + left, dx: ml + x - 7pt, dy: mt + ph + 3pt)[#text(size: 8pt)[#model_order.at(i)]]
    ]

    #place(top + left, dx: ml, dy: mt)[#path(stroke: 1.3pt + color-ar, fill: none, ..ar_pts)]
    #place(top + left, dx: ml, dy: mt)[#path(stroke: 1.3pt + color-tidar, fill: none, ..td_pts)]

    #for p in ar_pts [
      #place(top + left, dx: ml + p.at(0) - 1.8pt, dy: mt + p.at(1) - 1.8pt)[#circle(radius: 1.8pt, fill: color-ar)]
    ]
    #for p in td_pts [
      #place(top + left, dx: ml + p.at(0) - 1.8pt, dy: mt + p.at(1) - 1.8pt)[#circle(radius: 1.8pt, fill: color-tidar)]
    ]

    #place(top + left, dx: ml + 2pt, dy: mt + 2pt)[
      #grid(columns: (auto, auto), gutter: 4pt,
        [#rect(width: 8pt, height: 8pt, fill: color-ar)], [#text(size: 8pt)[AR baseline]],
        [#rect(width: 8pt, height: 8pt, fill: color-tidar)], [#text(size: 8pt)[TiDAR best]],
      )
    ]

    #place(top + left, dx: ml + pw/2 - 28pt, dy: height - 14pt)[#text(size: 8pt)[Model size]]
    #place(top + left, dx: 2pt, dy: mt + ph/2 - 16pt)[#rotate(-90deg)[#text(size: 8pt)[Tokens/s (decode only)]]]
  ]
}

#let grouped_bars(width: 175mm, height: 62mm) = {
  let ml = 24pt
  let mr = 8pt
  let mt = 8pt
  let mb = 20pt
  let pw = width - ml - mr
  let ph = height - mt - mb

  let allv = ()
  for p in prefill_order {
    for m in model_order {
      let r = find-row(m, p)
      if r != none { allv.push(float(r.speedup_tidar_over_ar)) }
    }
  }
  let ymax = calc.max(1.0, ..allv)

  box(width: width, height: height, stroke: 0.7pt + rgb("#c7c7c7"), radius: 6pt, inset: 4pt)[
    #place(top + left, dx: ml, dy: mt + ph)[#line(length: pw, stroke: 0.6pt + black)]
    #place(top + left, dx: ml, dy: mt)[#line(length: ph, angle: 90deg, stroke: 0.6pt + black)]

    #let group_w = pw / model_order.len()
    #let bar_w = group_w / 4
    #for i in range(model_order.len()) [
      #let m = model_order.at(i)
      #for j in range(prefill_order.len()) [
        #let p = prefill_order.at(j)
        #let r = find-row(m, p)
        #if r != none [
          #let v = float(r.speedup_tidar_over_ar)
          #let h = ph * v / ymax
          #let x = i * group_w + j * bar_w + 2pt
          #place(top + left, dx: ml + x, dy: mt + ph - h)[#rect(width: bar_w - 1.5pt, height: h, fill: color-grid.at(j))]
        ]
      ]
      #place(top + left, dx: ml + i * group_w + 2pt, dy: mt + ph + 3pt)[#text(size: 7pt)[#m]]
    ]

    #let y1 = ph * 1.0 / ymax
    #place(top + left, dx: ml, dy: mt + ph - y1)[#line(length: pw, stroke: (paint: rgb("#888"), thickness: 0.6pt, dash: (2pt, 2pt)))]

    #place(top + left, dx: ml + 4pt, dy: mt + 2pt)[
      #grid(columns: (auto, auto), gutter: 3pt,
        [#rect(width: 7pt, height: 7pt, fill: color-grid.at(0))], [#text(size: 7pt)[prefill 0]],
        [#rect(width: 7pt, height: 7pt, fill: color-grid.at(1))], [#text(size: 7pt)[prefill 1024]],
        [#rect(width: 7pt, height: 7pt, fill: color-grid.at(2))], [#text(size: 7pt)[prefill 4096]],
      )
    ]

    #place(top + left, dx: ml + pw/2 - 22pt, dy: height - 14pt)[#text(size: 8pt)[Model size]]
    #place(top + left, dx: 2pt, dy: mt + ph/2 - 16pt)[#rotate(-90deg)[#text(size: 8pt)[TiDAR/AR speedup]]]
  ]
}

= TiDAR vs AR Throughput (Updated Single-Forward)

Data files loaded dynamically:
- #data_dir + "/best_tidar_vs_ar_by_model_prefill.json"
- #data_dir + "/notes.txt"

Sweep:
- Models: 100m, 500m, 1b, 2b, 3b
- TiDAR grid: draft_len {4, 8, 16}, accept target {0.6, 0.8}, prefill {0, 1024, 4096}
- Decode steps: 256, repeat: 2
- AR baseline: reused prior run

== Line graphs: tokens/s vs model size

#text(weight: "bold")[Prefill = 0]
#line_chart(0)

#text(weight: "bold")[Prefill = 4096]
#line_chart(4096)

#pagebreak()

== Histogram-style grouped bars: TiDAR/AR speedup

#grouped_bars()

#let best = pairs.fold((none), (acc, r) => if acc == none or float(r.speedup_tidar_over_ar) > float(acc.speedup_tidar_over_ar) { r } else { acc })
#let worst = pairs.fold((none), (acc, r) => if acc == none or float(r.speedup_tidar_over_ar) < float(acc.speedup_tidar_over_ar) { r } else { acc })

== Key findings

- Best pair: model #best.model, prefill #best.prefill_tokens, speedup #str(calc.round(float(best.speedup_tidar_over_ar) * 1000) / 1000)x, TiDAR draft #best.tidar_best_draft_len at accept #best.tidar_best_accept_target.
- Worst pair: model #worst.model, prefill #worst.prefill_tokens, speedup #str(calc.round(float(worst.speedup_tidar_over_ar) * 1000) / 1000)x.
- Trend in this setup: speedup rises with model size and higher draft, but stays below 1.0x on A40 in current JAX path.

== Provenance

#text(size: 8pt)[#notes]
