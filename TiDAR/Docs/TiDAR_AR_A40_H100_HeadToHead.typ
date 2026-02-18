/*
"""
TiDAR_AR_A40_H100_HeadToHead.typ

Purpose
- Head-to-head TiDAR vs AR decode throughput on A40 and H100.
- Steady-state decode timing only (compile excluded from charted metrics).

A40 run provenance (executed on 2026-02-09)
- Host: root@gpu-box-1
- GPU: NVIDIA A40
- Config: /tmp/config_3b.yml (copied from TiDAR/Docs/Benchmark_logs/tidar_updated_narrow_20260208_094343/configs/config_3b.yml)
- Global config: /tmp/Global_Config.yml (copied from TiDAR/Global_Config.yml)
- Matrix:
  - prefill_tokens: 1024, 2048
  - steps: 256
  - TiDAR: draft_len 8 and 16, accept-rate 0.8
  - AR: same model shape + tokenizer + context; no TiDAR branch logic
  - warmup_runs=1, bench_runs=5
- Command (raw):
  ssh root@gpu-box-1 'set -euo pipefail; cd /proj/SUPER-GIANT; CFG=/tmp/config_3b.yml; GC=/tmp/Global_Config.yml; \
    echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"; \
    for PREFILL in 1024 2048; do \
      echo "=== AR prefill=${PREFILL} ==="; \
      /opt/venv/bin/python /tmp/benchmark_ar_head2head.py --config ${CFG} --global_config ${GC} --context_length 4608 --steps 256 --prefill_tokens ${PREFILL} --temperature 1.0 --top_k 0 --warmup_runs 1 --bench_runs 5; \
      for K in 8 16; do \
        echo "=== TiDAR prefill=${PREFILL} K=${K} acc=0.8 ==="; \
        /opt/venv/bin/python TiDAR/model/test_inference_througthput.py --config ${CFG} --global_config ${GC} --context_length 4608 --steps 256 --draft_len ${K} --accept-rate 0.8 --prefill_tokens ${PREFILL} --silent --warmup_runs 1 --bench_runs 5 --component_breakdown_iters 64; \
      done; \
    done'
- Raw log:
  TiDAR/Docs/Benchmark_logs/head2head_a40_h100_20260209_193108/raw/a40_head2head.log
- Parsed summaries:
  TiDAR/Docs/Benchmark_logs/head2head_a40_h100_20260209_193108/raw/parsed_runs.json
  TiDAR/Docs/Benchmark_logs/head2head_a40_h100_20260209_193108/raw/parsed_runs.csv

H100 run provenance (executed on 2026-02-09)
- Existing stopped pod `c7nvx4td0svni2` could not be resumed due host-capacity error from RunPod.
- New pod created from same template/image:
  - pod id: `oz4wwwp8gwpvuw`
  - host: root@gpu-box-2
  - GPU: NVIDIA H100 PCIe
- Matrix and script arguments were identical to A40 run:
  - prefill_tokens: 1024, 2048
  - steps: 256
  - TiDAR: draft_len 8 and 16, accept-rate 0.8
  - AR: same model shape + tokenizer + context
  - warmup_runs=1, bench_runs=5
- Command (raw):
  ssh root@gpu-box-2 'set -euo pipefail; cd /proj/SUPER-GIANT; CFG=/tmp/config_3b.yml; GC=/tmp/Global_Config.yml; \
    echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"; \
    for PREFILL in 1024 2048; do \
      echo "=== AR prefill=${PREFILL} ==="; \
      /opt/venv/bin/python /tmp/benchmark_ar_head2head.py --config ${CFG} --global_config ${GC} --context_length 4608 --steps 256 --prefill_tokens ${PREFILL} --temperature 1.0 --top_k 0 --warmup_runs 1 --bench_runs 5; \
      for K in 8 16; do \
        echo "=== TiDAR prefill=${PREFILL} K=${K} acc=0.8 ==="; \
        /opt/venv/bin/python TiDAR/model/test_inference_througthput.py --config ${CFG} --global_config ${GC} --context_length 4608 --steps 256 --draft_len ${K} --accept-rate 0.8 --prefill_tokens ${PREFILL} --silent --warmup_runs 1 --bench_runs 5 --component_breakdown_iters 64; \
      done; \
    done'
- Raw log:
  TiDAR/Docs/Benchmark_logs/head2head_a40_h100_20260209_193108/raw/h100_head2head.log
- Parsed summaries:
  TiDAR/Docs/Benchmark_logs/head2head_a40_h100_20260209_193108/raw/h100_parsed/parsed_runs.json
  TiDAR/Docs/Benchmark_logs/head2head_a40_h100_20260209_193108/raw/h100_parsed/parsed_runs.csv
  TiDAR/Docs/Benchmark_logs/head2head_a40_h100_20260209_193108/raw/a40_h100_overlay_summary.json

Metric used for charts
- AR steady TPS: `steady_tokens_per_second_mean` from AR run.
- TiDAR steady TPS: `steady_tokens_per_second_mean` from TiDAR run.
- Speedup vs AR: `TiDAR_steady_tps / AR_steady_tps`.
"""
*/

#set page(width: 210mm, height: 297mm, margin: 12mm)
#set text(font: "New Computer Modern", size: 10pt)
#set heading(numbering: none)

#let c-a40-k8 = rgb("#1D4ED8")
#let c-a40-k16 = rgb("#DC2626")
#let c-h100-k8 = rgb("#059669")
#let c-h100-k16 = rgb("#7C3AED")

#let legend-item(label, color) = grid(
  columns: (auto, auto),
  gutter: 4pt,
  align: left,
  rect(width: 8pt, height: 8pt, fill: color, radius: 1.4pt),
  text(size: 8.5pt)[#label],
)

#let line-chart(
  title,
  series,
  width: 182mm,
  height: 70mm,
  x-label: "prefill tokens",
  y-label: "speedup vs AR (steady TPS)",
) = {
  let margin-left = 40pt
  let margin-right = 10pt
  let margin-top = 14pt
  let margin-bottom = 24pt
  let y-label-area = 16pt
  let plot-width = width - margin-left - margin-right
  let plot-height = height - margin-top - margin-bottom

  let min-x = calc.min(..series.map(s => calc.min(..s.data.map(d => d.at(0)))))
  let max-x = calc.max(..series.map(s => calc.max(..s.data.map(d => d.at(0)))))
  let max-y-data = calc.max(..series.map(s => calc.max(..s.data.map(d => d.at(1)))))
  let min-y = 0.0
  let max-y = max-y-data * 1.10

  let sx = if max-x == min-x { plot-width } else { plot-width / (max-x - min-x) }
  let sy = if max-y == min-y { plot-height } else { plot-height / (max-y - min-y) }

  let x-ticks = (1024, 2048)
  let y-ticks = range(0, 7)

  block(width: width, height: height, inset: 5pt, stroke: 0.7pt + rgb("#D1D5DB"), radius: 6pt)[
    #text(weight: "semibold", size: 9.6pt)[#title]
    #v(2pt)
    #place(top + left, dx: margin-left, dy: margin-top + plot-height)[
      #line(length: plot-width, stroke: 0.6pt + black)
    ]
    #place(top + left, dx: margin-left, dy: margin-top)[
      #line(length: plot-height, angle: 90deg, stroke: 0.6pt + black)
    ]

    #for t in y-ticks [
      #let yv = min-y + (max-y - min-y) * (t / 6)
      #let py = margin-top + plot-height - (yv - min-y) * sy
      #place(top + left, dx: margin-left, dy: py)[
        #line(length: plot-width, stroke: (paint: rgb("#E5E7EB"), thickness: 0.45pt, dash: (2pt, 2pt)))
      ]
      #place(top + left, dx: y-label-area, dy: py - 4pt)[
        #box(width: margin-left - y-label-area)[
          #align(right)[
            #text(size: 7pt, fill: rgb("#4B5563"))[#(str(calc.round(yv, digits: 2)))]
          ]
        ]
      ]
    ]

    #for xv in x-ticks [
      #let px = margin-left + (xv - min-x) * sx
      #place(top + left, dx: px, dy: margin-top + plot-height)[
        #line(length: 3pt, angle: 90deg, stroke: 0.6pt + black)
      ]
      #place(top + left, dx: px - 10pt, dy: margin-top + plot-height + 4pt)[
        #text(size: 8pt)[#(str(xv))]
      ]
    ]

    #for s in series [
      #place(top + left, dx: margin-left, dy: margin-top)[
        #path(
          stroke: 1.3pt + s.color,
          fill: none,
          ..s.data.map(d => (
            (d.at(0) - min-x) * sx,
            plot-height - (d.at(1) - min-y) * sy,
          )),
        )
      ]
      #for d in s.data [
        #let px = margin-left + (d.at(0) - min-x) * sx
        #let py = margin-top + plot-height - (d.at(1) - min-y) * sy
        #place(top + left, dx: px - 1.7pt, dy: py - 1.7pt)[
          #circle(radius: 1.7pt, fill: s.color)
        ]
      ]
    ]

    #place(top + left, dx: margin-left, dy: margin-top + plot-height + 12pt)[
      #text(size: 7.8pt)[#x-label]
    ]
    #place(top + left, dx: 0pt, dy: margin-top)[
      #box(width: y-label-area, height: plot-height)[
        #align(center)[
          #rotate(-90deg, origin: center, reflow: true)[
            #box(width: plot-height, height: y-label-area)[
              #align(center)[#text(size: 7.8pt)[#y-label]]
            ]
          ]
        ]
      ]
    ]
  ]
}

#let bar-chart(
  title,
  data,
  width: 182mm,
  height: 70mm,
) = {
  let margin-left = 30pt
  let margin-right = 10pt
  let margin-top = 14pt
  let margin-bottom = 30pt
  let plot-width = width - margin-left - margin-right
  let plot-height = height - margin-top - margin-bottom

  let n = data.len()
  let max-y = calc.max(..data.map(d => d.at(2))) * 1.15
  let bar-gap = 16pt
  let bar-w = (plot-width - (n - 1) * bar-gap) / n

  block(width: width, height: height, inset: 5pt, stroke: 0.7pt + rgb("#D1D5DB"), radius: 6pt)[
    #text(weight: "semibold", size: 9.6pt)[#title]
    #v(2pt)
    #place(top + left, dx: margin-left, dy: margin-top + plot-height)[
      #line(length: plot-width, stroke: 0.6pt + black)
    ]
    #place(top + left, dx: margin-left, dy: margin-top)[
      #line(length: plot-height, angle: 90deg, stroke: 0.6pt + black)
    ]

    #for i in range(0, n) [
      #let item = data.at(i)
      #let x = margin-left + i * (bar-w + bar-gap)
      #let h = (item.at(2) / max-y) * plot-height
      #place(top + left, dx: x, dy: margin-top + plot-height - h)[
        #rect(width: bar-w, height: h, fill: item.at(3), stroke: none, radius: 2pt)
      ]
      #place(top + left, dx: x + 1pt, dy: margin-top + plot-height + 4pt)[
        #text(size: 7.5pt)[#item.at(1)]
      ]
      #place(top + left, dx: x + 1pt, dy: margin-top + plot-height - h - 10pt)[
        #text(size: 7pt)[#(str(calc.round(item.at(2), digits: 2)) + "x")]
      ]
    ]
  ]
}

#let a40_speedup_lines = (
  (label: "A40 TiDAR K=8", color: c-a40-k8, data: ((1024, 5.494207), (2048, 5.059560))),
  (label: "A40 TiDAR K=16", color: c-a40-k16, data: ((1024, 8.693304), (2048, 7.817495))),
)

#let overlay_speedup_lines = (
  (label: "A40 TiDAR K=8", color: c-a40-k8, data: ((1024, 5.494207), (2048, 5.059560))),
  (label: "A40 TiDAR K=16", color: c-a40-k16, data: ((1024, 8.693304), (2048, 7.817495))),
  (label: "H100 TiDAR K=8", color: c-h100-k8, data: ((1024, 5.366022), (2048, 4.930489))),
  (label: "H100 TiDAR K=16", color: c-h100-k16, data: ((1024, 9.745146), (2048, 8.607205))),
)

= TiDAR vs AR Head-to-Head (A40 + H100, 3b config)

This document tracks steady-state decode throughput only (compile excluded).

#v(4pt)
#legend-item("A40 TiDAR K=8", c-a40-k8) #h(10pt) #legend-item("A40 TiDAR K=16", c-a40-k16)
#v(2pt)
#legend-item("H100 TiDAR K=8", c-h100-k8) #h(10pt) #legend-item("H100 TiDAR K=16", c-h100-k16)

#v(4pt)
#line-chart("Speedup vs AR by Prefill (steady decode TPS)", overlay_speedup_lines)

#v(6pt)
#let overlay_bars = (
  ("A40", "K8@1024", 5.494207, c-a40-k8),
  ("A40", "K16@1024", 8.693304, c-a40-k16),
  ("A40", "K8@2048", 5.059560, c-a40-k8),
  ("A40", "K16@2048", 7.817495, c-a40-k16),
  ("H100", "K8@1024", 5.366022, c-h100-k8),
  ("H100", "K16@1024", 9.745146, c-h100-k16),
  ("H100", "K8@2048", 4.930489, c-h100-k8),
  ("H100", "K16@2048", 8.607205, c-h100-k16),
)
#bar-chart("Speedup vs AR (steady TPS)", overlay_bars)

= A40 steady metrics snapshot

- AR steady TPS:
  - prefill 1024: `18.754672`
  - prefill 2048: `18.736833`
- TiDAR steady TPS:
  - prefill 1024, `K=8`: `103.042058`
  - prefill 1024, `K=16`: `163.040059`
  - prefill 2048, `K=8`: `94.800140`
  - prefill 2048, `K=16`: `146.475107`
- TiDAR postprocess share:
  - prefill 1024, `K=8`: `0.4775%`
  - prefill 1024, `K=16`: `0.2639%`
  - prefill 2048, `K=8`: `0.4029%`
  - prefill 2048, `K=16`: `0.1782%`

= H100 steady metrics snapshot

- AR steady TPS:
  - prefill 1024: `55.045774`
  - prefill 2048: `54.970468`
- TiDAR steady TPS:
  - prefill 1024, `K=8`: `295.376852`
  - prefill 1024, `K=16`: `536.429106`
  - prefill 2048, `K=8`: `271.031309`
  - prefill 2048, `K=16`: `473.142103`
- TiDAR postprocess share:
  - prefill 1024, `K=8`: `0.5775%`
  - prefill 1024, `K=16`: `0.3762%`
  - prefill 2048, `K=8`: `0.5070%`
  - prefill 2048, `K=16`: `0.2928%`

= Overlay takeaways

- Speedup vs AR is similar across A40 and H100 at `K=8`.
- At `K=16`, H100 shows a stronger speedup curve than A40.
- Absolute TiDAR steady TPS is much higher on H100:
  - roughly `2.86x` to `3.29x` TiDAR TPS vs A40 for the same config points.
