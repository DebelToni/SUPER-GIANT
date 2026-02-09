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

#pagebreak()

= H100 Chat Session Addendum (Mask-Path Comparison)

This section documents the extra H100 PCIe runs executed during this chat so
the results are preserved in one place.

What was compared (described by runtime behavior, not git state names):

- Path A — Full-width decode mask path:
  - Decode bias is built over full keys as `[PREFIX | STEP]`
    (`build_decode_bias_template(cache_len, draft_len, ...)`).
  - Decode attention builds `k_full = [k_prefix, k_step]`.
  - Per-step prefix validity is added as explicit bias, then one dense
    `dot_product_attention` call is run.
  - This is the current baseline inference path used in the earlier H100 sweep.

- Path B — Local experimental mask-layout branch:
  - Decode bias source is switched to step-first layout
    (`build_decode_bias_template_step_prefix(...)`) in inference/throughput code.
  - Attention decode path in `Transformer_block.py` is expanded with:
    - single-call step-first K/V path using `key_value_seq_lengths`,
    - legacy fallback for prefix-first full-width masks,
    - step-only split-attention merge path (prefix attention + step attention).
  - Additional step-only bias builders were added in `tidar_core.py` and prefill
    bias wiring was changed to step-only for prompt+draft prefill.

== Hardware and run settings

- GPU: NVIDIA H100 PCIe (81,559 MiB)
- Remote host alias used: `gpu-box-1`
- Decode context setting: `context_length = 4608`
- Prefill setting: `prefill_tokens = 4096`
- Decode steps: `256`
- Seed: `0`
- Acceptance target for TiDAR throughput tests: `accept-rate = 0.8`

== Scripts used in this chat

- TiDAR throughput script:
  - `TiDAR/model/test_inference_througthput.py`
- AR decode-only helper used for matched 3B comparison:
  - `/tmp/ar_decode_throughput_current.py`
  - (temporary helper; uses the same TiDAR model stack and KV-cache path for
    one-token AR decode)

#let h100-path-a = (rgb("#ef3b2c"), rgb("#fb6a4a"), rgb("#cb181d"))
#let h100-path-b = (rgb("#fdae6b"), rgb("#fd8d3c"), rgb("#e6550d"))

#let h100_sweep_bars = (
  ("500m d4", 15.405662, color-grid.at(0)),
  ("500m d8", 16.033103, color-grid.at(1)),
  ("500m d16", 15.354966, color-grid.at(2)),
  ("1.5b d4", 13.519934, rgb("#8c96c6")),
  ("3b d4", 10.906801, h100-path-a.at(0)),
  ("3b d8", 12.730544, h100-path-a.at(1)),
  ("3b d16", 13.171004, h100-path-a.at(2)),
)

#let h100_matched_bars = (
  ("AR baseline", 14.744286, color-ar),
  ("Path A d4", 11.100869, h100-path-a.at(0)),
  ("Path A d8", 13.344361, h100-path-a.at(1)),
  ("Path A d16", 14.918518, h100-path-a.at(2)),
  ("Path B d4", 11.485874, h100-path-b.at(0)),
  ("Path B d8", 11.408315, h100-path-b.at(1)),
  ("Path B d16", 9.026591, h100-path-b.at(2)),
)

#let h100_ratio_bars = (
  ("Path A d4", 11.100869 / 14.744286, h100-path-a.at(0)),
  ("Path A d8", 13.344361 / 14.744286, h100-path-a.at(1)),
  ("Path A d16", 14.918518 / 14.744286, h100-path-a.at(2)),
  ("Path B d4", 11.485874 / 14.744286, h100-path-b.at(0)),
  ("Path B d8", 11.408315 / 14.744286, h100-path-b.at(1)),
  ("Path B d16", 9.026591 / 14.744286, h100-path-b.at(2)),
)

#let fmt3(v) = str(calc.round(float(v) * 1000) / 1000)

#let h100_hbars(title, bars, width: 175mm, reference: none, x_label: "tokens/s (decode only)") = {
  let ml = 56pt
  let mr = 30pt
  let mt = 16pt
  let mb = 16pt
  let row_h = 9pt
  let row_gap = 5pt
  let n = bars.len()
  let ph = n * row_h + (n - 1) * row_gap
  let height = mt + ph + mb
  let pw = width - ml - mr

  let vmax = bars.fold(0.0, (acc, b) => calc.max(acc, float(b.at(1))))
  let vmax = if reference != none { calc.max(vmax, float(reference)) } else { vmax }
  let vmax = if vmax <= 0.0 { 1.0 } else { vmax }

  box(width: width, height: height, stroke: 0.7pt + rgb("#c7c7c7"), radius: 6pt, inset: 4pt)[
    #place(top + left, dx: 4pt, dy: 0pt)[#text(size: 9pt, weight: "bold")[#title]]
    #place(top + left, dx: ml, dy: mt + ph)[#line(length: pw, stroke: 0.5pt + rgb("#666"))]

    #if reference != none [
      #let xref = pw * float(reference) / vmax
      #place(top + left, dx: ml + xref, dy: mt - 1pt)[#line(length: ph + 2pt, angle: 90deg, stroke: (paint: rgb("#555"), thickness: 0.6pt, dash: (2pt, 2pt)))]
      #place(top + left, dx: ml + xref + 2pt, dy: mt - 10pt)[#text(size: 7pt)[ref #fmt3(reference)]]
    ]

    #for i in range(n) [
      #let b = bars.at(i)
      #let label = b.at(0)
      #let val = float(b.at(1))
      #let col = b.at(2)
      #let y = i * (row_h + row_gap)
      #let bw = pw * val / vmax

      #place(top + left, dx: 2pt, dy: mt + y - 1pt)[#text(size: 7.5pt)[#label]]
      #place(top + left, dx: ml, dy: mt + y)[#rect(width: bw, height: row_h, fill: col)]
      #place(top + left, dx: ml + bw + 3pt, dy: mt + y - 1pt)[#text(size: 7.5pt)[#fmt3(val)]]
    ]

    #place(top + left, dx: ml, dy: mt + ph + 2pt)[#text(size: 7pt)[0]]
    #place(top + left, dx: ml + pw - 16pt, dy: mt + ph + 2pt)[#text(size: 7pt)[#fmt3(vmax)]]
    #place(top + left, dx: ml + pw / 2 - 26pt, dy: height - 10pt)[#text(size: 7pt)[#x_label]]
  ]
}

== Visual summaries

#h100_hbars("H100 sweep: TiDAR decode throughput", h100_sweep_bars)

#v(6pt)

#h100_hbars("H100 matched 3B: AR and both TiDAR paths", h100_matched_bars, reference: 14.744286)

#v(6pt)

#h100_hbars("H100 matched 3B: TiDAR/AR speed ratio", h100_ratio_bars, reference: 1.0, x_label: "speed ratio vs AR (1.0 = break-even)")

== Recorded H100 metrics from this chat

First H100 TiDAR sweep (Path A, decode-only tokens/s):

- 500m, draft 4: `15.405662` tok/s
- 500m, draft 8: `16.033103` tok/s
- 500m, draft 16: `15.354966` tok/s
- 1.5b, draft 4: `13.519934` tok/s
- 3b, draft 4: `10.906801` tok/s
- 3b, draft 8: `12.730544` tok/s
- 3b, draft 16: `13.171004` tok/s

Matched 3B run (same settings above), AR + TiDAR path comparison:

- AR baseline (decode-only):
  - generated tokens: `256`
  - iterations: `256`
  - decode time: `17.362658 s`
  - throughput: `14.744286 tok/s`

- TiDAR, Path A, draft 4:
  - generated tokens: `256`
  - iterations: `78`
  - observed avg accept/iter: `3.282051`
  - observed max accept/iter: `4`
  - decode time: `23.061257 s`
  - throughput: `11.100869 tok/s`

- TiDAR, Path A, draft 8:
  - generated tokens: `256`
  - iterations: `39`
  - observed avg accept/iter: `6.564103`
  - observed max accept/iter: `8`
  - decode time: `19.184134 s`
  - throughput: `13.344361 tok/s`

- TiDAR, Path A, draft 16:
  - generated tokens: `256`
  - iterations: `21`
  - observed avg accept/iter: `12.190476`
  - observed max accept/iter: `16`
  - decode time: `17.159882 s`
  - throughput: `14.918518 tok/s`

- TiDAR, Path B, draft 4:
  - generated tokens: `256`
  - iterations: `78`
  - observed avg accept/iter: `3.282051`
  - observed max accept/iter: `4`
  - decode time: `22.288248 s`
  - throughput: `11.485874 tok/s`

- TiDAR, Path B, draft 8:
  - generated tokens: `256`
  - iterations: `39`
  - observed avg accept/iter: `6.564103`
  - observed max accept/iter: `8`
  - decode time: `22.439773 s`
  - throughput: `11.408315 tok/s`

- TiDAR, Path B, draft 16:
  - generated tokens: `256`
  - iterations: `21`
  - observed avg accept/iter: `12.190476`
  - observed max accept/iter: `16`
  - decode time: `28.360653 s`
  - throughput: `9.026591 tok/s`

== Quick interpretation from these runs

- In this matched 3B run, Path A at draft 16 slightly exceeded AR baseline
  throughput (`14.918518` vs `14.744286` tok/s).
- Path B improved only draft 4 slightly, but regressed strongly for draft 8/16.
- Net result from this chat: the current local experimental branch did not
  improve overall TiDAR throughput on H100 in the tested settings.

#pagebreak()
