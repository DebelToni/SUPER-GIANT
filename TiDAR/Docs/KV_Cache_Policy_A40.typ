/*
"""
KV_Cache_Policy_A40.typ

Goal
- Compare three KV cache sizing policies for TiDAR decode on A40:
  - Red: full context length
  - Blue: current implementation (exact required_len)
  - Orange: bucketed cache length

Policies tested
- red/full:
  - cache_len = context_length
  - buffer_len = context_length
- blue/current exact:
  - cache_len = required_len
  - buffer_len = required_len
- orange/bucket:
  - bucket set: 32, 128, 256, 512, 1024, 1536, 2048, then +512
  - cache_len = next_bucket(required_len)
  - buffer_len = cache_len

Hardware/run context
- RunPod pod: giant-a40-kv-policy (id: g4sxuzcm4ew92i)
- Host: root@gpu-box-1
- GPU: NVIDIA A40
- Date: 2026-02-09

Model variants and settings
- 500m profile with draft_len=4
  - config: TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/configs/config_500m.yml
  - scenarios (prefill:steps):
    - 90:32, 95:32, 350:128, 380:128, 890:128, 900:128, 1380:128, 1390:128
- 3b profile with draft_len=16
  - config: TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/configs/config_3b.yml
  - scenarios (prefill:steps):
    - 78:32, 83:32, 338:128, 368:128, 878:128, 888:128, 1368:128, 1378:128

Sampling and benchmark controls
- accept_rate=0.8
- warmup_runs=1
- bench_runs=3
- steady metric used for primary charts: steady_tokens_per_second_mean (compile excluded)

How runs were launched
- Helper script (ephemeral during analysis): /tmp/benchmark_tidar_kv_policies.py
- 500m run:
  ssh root@gpu-box-1 'cd /proj/SUPER-GIANT; /opt/venv/bin/python /tmp/benchmark_tidar_kv_policies.py \
    --config /tmp/config_500m.yml --global_config /tmp/Global_Config.yml --draft_len 4 \
    --accept_rate 0.8 --scenarios 90:32,95:32,350:128,380:128,890:128,900:128,1380:128,1390:128 \
    --policies full,exact,bucket --warmup_runs 1 --bench_runs 3 --model_label 500m_k4 \
    --output_json /tmp/kv_policy_500m.json --output_csv /tmp/kv_policy_500m.csv'
- 3b run:
  ssh root@gpu-box-1 'cd /proj/SUPER-GIANT; /opt/venv/bin/python /tmp/benchmark_tidar_kv_policies.py \
    --config /tmp/config_3b.yml --global_config /tmp/Global_Config.yml --draft_len 16 \
    --accept_rate 0.8 --scenarios 78:32,83:32,338:128,368:128,878:128,888:128,1368:128,1378:128 \
    --policies full,exact,bucket --warmup_runs 1 --bench_runs 3 --model_label 3b_k16 \
    --output_json /tmp/kv_policy_3b.json --output_csv /tmp/kv_policy_3b.csv'

Artifacts in repo
- raw logs:
  - TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/raw/a40_500m_k4.log
  - TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/raw/a40_3b_k16.log
- parsed:
  - TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/parsed/kv_policy_500m.csv
  - TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/parsed/kv_policy_3b.csv
  - TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/parsed/summary.json
  - TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/parsed/summary_points.csv
"""
*/

#set page(width: 210mm, height: 297mm, margin: 12mm)
#set text(font: "New Computer Modern", size: 10pt)
#set heading(numbering: none)

#let c-red = rgb("#DC2626")
#let c-blue = rgb("#1D4ED8")
#let c-orange = rgb("#F59E0B")

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
  height: 66mm,
  x-label: "required_len",
  y-label: "tokens/s",
) = {
  let margin-left = 30pt
  let margin-right = 8pt
  let margin-top = 12pt
  let margin-bottom = 22pt
  let plot-width = width - margin-left - margin-right
  let plot-height = height - margin-top - margin-bottom

  let min-x = calc.min(..series.map(s => calc.min(..s.data.map(d => d.at(0)))))
  let max-x = calc.max(..series.map(s => calc.max(..s.data.map(d => d.at(0)))))
  let max-y-data = calc.max(..series.map(s => calc.max(..s.data.map(d => d.at(1)))))
  let min-y = 0.0
  let max-y = max-y-data * 1.08

  let sx = if max-x == min-x { plot-width } else { plot-width / (max-x - min-x) }
  let sy = if max-y == min-y { plot-height } else { plot-height / (max-y - min-y) }

  let x-ticks = (127, 132, 483, 513, 1023, 1033, 1513, 1523)
  let y-ticks = range(0, 7)

  block(width: width, height: height, inset: 5pt, stroke: 0.7pt + rgb("#D1D5DB"), radius: 6pt)[
    #text(weight: "semibold", size: 9.4pt)[#title]
    #v(1pt)
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
      #place(top + left, dx: 0pt, dy: py - 4pt)[
        #text(size: 7pt, fill: rgb("#4B5563"))[#(str(calc.round(yv, digits: 1)))]
      ]
    ]

    #for xv in x-ticks [
      #let px = margin-left + (xv - min-x) * sx
      #let tick_dx = if xv == 483 {
        -12pt
      } else if xv == 513 {
        -2pt
      } else if xv == 1023 {
        -16pt
      } else if xv == 1033 {
        1pt
      } else if xv == 1513 {
        -16pt
      } else if xv == 1523 {
        1pt
      } else {
        -8pt
      }
      #place(top + left, dx: px, dy: margin-top + plot-height)[
        #line(length: 3pt, angle: 90deg, stroke: 0.6pt + black)
      ]
      #place(top + left, dx: px + tick_dx, dy: margin-top + plot-height + 4pt)[
        #text(size: 7pt)[#(str(xv))]
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
        #place(top + left, dx: px - 1.4pt, dy: py - 1.4pt)[
          #circle(radius: 1.4pt, fill: s.color)
        ]
      ]
    ]

    #place(top + left, dx: margin-left, dy: margin-top + plot-height + 12pt)[
      #text(size: 7.8pt)[#x-label]
    ]
    #place(top + left, dx: 0pt, dy: margin-top + plot-height / 2 - 12pt)[
      #rotate(-90deg)[#text(size: 7.8pt)[#y-label]]
    ]
  ]
}

#let m500_steady = (
  (label: "Red - full context", color: c-red, data: ((127,133.650203),(132,133.285394),(483,168.658929),(513,147.266236),(1023,163.956736),(1033,159.733058),(1513,168.458277),(1523,155.666638))),
  (label: "Blue - current exact", color: c-blue, data: ((127,248.256352),(132,251.236825),(483,292.999185),(513,254.240288),(1023,255.317779),(1033,248.166361),(1513,241.692286),(1523,220.523478))),
  (label: "Orange - bucketed", color: c-orange, data: ((127,251.046496),(132,245.417031),(483,297.516370),(513,238.654099),(1023,265.068233),(1033,236.619782),(1513,250.536870),(1523,230.936287))),
)

#let m500_first = (
  (label: "Red - full context", color: c-red, data: ((127,2.467525),(132,130.233309),(483,167.781072),(513,146.880648),(1023,163.335482),(1033,159.543947),(1513,167.895496),(1523,154.880095))),
  (label: "Blue - current exact", color: c-blue, data: ((127,4.086278),(132,4.049319),(483,15.561391),(513,16.025941),(1023,15.675334),(1033,14.577054),(1513,15.453114),(1523,15.809066))),
  (label: "Orange - bucketed", color: c-orange, data: ((127,3.886539),(132,4.088763),(483,15.039732),(513,15.528990),(1023,264.307693),(1033,15.716320),(1513,249.131411),(1523,229.885270))),
)

#let m3b_steady = (
  (label: "Red - full context", color: c-red, data: ((127,149.730911),(132,149.759497),(483,136.946086),(513,103.241518),(1023,137.007872),(1033,153.953196),(1513,136.976577),(1523,112.169817))),
  (label: "Blue - current exact", color: c-blue, data: ((127,199.931700),(132,203.856625),(483,177.690885),(513,132.494983),(1023,166.210889),(1033,186.351425),(1513,157.873326),(1523,128.574819))),
  (label: "Orange - bucketed", color: c-orange, data: ((127,203.623786),(132,201.137554),(483,182.631646),(513,131.715602),(1023,174.786128),(1033,189.102448),(1513,168.164869),(1523,138.034379))),
)

#let m3b_first = (
  (label: "Red - full context", color: c-red, data: ((127,2.224853),(132,143.729788),(483,136.577239),(513,102.556402),(1023,135.926613),(1033,153.636249),(1513,135.723772),(1523,111.946787))),
  (label: "Blue - current exact", color: c-blue, data: ((127,3.612341),(132,3.403162),(483,12.869280),(513,12.677263),(1023,13.841286),(1033,13.275445),(1513,13.406807),(1523,12.894864))),
  (label: "Orange - bucketed", color: c-orange, data: ((127,3.548151),(132,2.968205),(483,10.462709),(513,10.931957),(1023,174.642184),(1033,11.041903),(1513,167.532464),(1523,137.562742))),
)

= KV Cache Policy Experiment on A40

#legend-item("Red - full context", c-red) #h(10pt) #legend-item("Blue - current exact", c-blue) #h(10pt) #legend-item("Orange - bucketed", c-orange)

#v(5pt)
= 500m model, draft_len=4

#line-chart(
  "Steady decode throughput (compile excluded)",
  m500_steady,
  y-label: "steady tokens/s",
)

#v(3pt)
#line-chart(
  "First request throughput (compile + first run)",
  m500_first,
  y-label: "compile+first tokens/s",
)

- Unique compile keys seen:
  - Red/full: `1`
  - Blue/current exact: `8`
  - Orange/bucketed: `5`
- Mean steady TPS:
  - Red/full: `153.834`
  - Blue/current exact: `251.554`
  - Orange/bucketed: `251.974`

#pagebreak()

= 3b model, draft_len=16

#line-chart(
  "Steady decode throughput (compile excluded)",
  m3b_steady,
  y-label: "steady tokens/s",
)

#v(3pt)
#line-chart(
  "First request throughput (compile + first run)",
  m3b_first,
  y-label: "compile+first tokens/s",
)

- Unique compile keys seen:
  - Red/full: `1`
  - Blue/current exact: `8`
  - Orange/bucketed: `5`
- Mean steady TPS:
  - Red/full: `134.973`
  - Blue/current exact: `169.123`
  - Orange/bucketed: `173.650`

= Notes

- Primary comparison metric is steady decode TPS (`steady_tokens_per_second_mean`), i.e. compile excluded.
- Blue/current exact gives the most shape variants (one per required_len point in this sweep).
- Orange bucketed reduces shape variants while keeping steady throughput near or above blue in this matrix.
