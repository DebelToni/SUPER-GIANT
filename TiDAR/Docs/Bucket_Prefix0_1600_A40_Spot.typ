/*
"""
Bucket_Prefix0_1600_A40_Spot.typ

Goal
- Run long decode sweeps with bucketed KV cache only, prefix=0, so throughput curves are smooth over generated length.
- Measure up to 1600 generated tokens on A40 Spot.
- Primary metric: steady decode TPS (compile excluded).

Hardware/run context
- RunPod pod: giant-a40-spot (id: 62xhy28r6e3fo1, interruptible=true)
- Host: root@gpu-box-1
- GPU: NVIDIA A40
- Date: 2026-02-10

Repository / code
- commit: 8bb73d2
- benchmark harness (ephemeral): /tmp/benchmark_tidar_bucket_longrun.py

Exact commands used
- 500m / draft_len=4:
  ssh root@gpu-box-1 'cd /proj/SUPER-GIANT; /opt/venv/bin/python /tmp/benchmark_tidar_bucket_longrun.py \
    --config /proj/SUPER-GIANT/TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/configs/config_500m.yml \
    --global_config /proj/SUPER-GIANT/TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/configs/Global_Config.yml \
    --draft_len 4 --accept_rate 0.8 --steps 32:1600:32 --prefill_tokens 0 \
    --warmup_runs 1 --bench_runs 2 --model_label 500m_k4_bucket_prefix0_spot \
    --output_json /tmp/bucket_prefix0_spot_500m.json --output_csv /tmp/bucket_prefix0_spot_500m.csv'

- 3b / draft_len=16:
  ssh root@gpu-box-1 'cd /proj/SUPER-GIANT; /opt/venv/bin/python /tmp/benchmark_tidar_bucket_longrun.py \
    --config /proj/SUPER-GIANT/TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/configs/config_3b.yml \
    --global_config /proj/SUPER-GIANT/TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/configs/Global_Config.yml \
    --draft_len 16 --accept_rate 0.8 --steps 32:1600:32 --prefill_tokens 0 \
    --warmup_runs 1 --bench_runs 2 --model_label 3b_k16_bucket_prefix0_spot \
    --output_json /tmp/bucket_prefix0_spot_3b.json --output_csv /tmp/bucket_prefix0_spot_3b.csv'

- 500m / full-context precompiled baseline (red line on first chart):
  ssh root@gpu-box-1 'cd /proj/SUPER-GIANT; /opt/venv/bin/python /tmp/benchmark_tidar_longrun_policy.py \
    --config /proj/SUPER-GIANT/TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/configs/config_500m.yml \
    --global_config /proj/SUPER-GIANT/TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/configs/Global_Config.yml \
    --draft_len 4 --policy full --accept_rate 0.8 --steps 32:1600:32 --prefill_tokens 0 \
    --warmup_runs 1 --bench_runs 2 --model_label 500m_k4_full_prefix0_spot \
    --output_json /tmp/full_prefix0_spot_500m.json --output_csv /tmp/full_prefix0_spot_500m.csv'

- 3b / full-context precompiled baseline (red line on second chart):
  ssh root@gpu-box-1 'cd /proj/SUPER-GIANT; /opt/venv/bin/python /tmp/benchmark_tidar_longrun_policy.py \
    --config /proj/SUPER-GIANT/TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/configs/config_3b.yml \
    --global_config /proj/SUPER-GIANT/TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/configs/Global_Config.yml \
    --draft_len 16 --policy full --accept_rate 0.8 --steps 32:1600:32 --prefill_tokens 0 \
    --warmup_runs 1 --bench_runs 2 --model_label 3b_k16_full_prefix0_spot \
    --output_json /tmp/full_prefix0_spot_3b.json --output_csv /tmp/full_prefix0_spot_3b.csv'

- 500m / AR full-context precompiled baseline (light green line):
  ssh root@gpu-box-1 'cd /proj/SUPER-GIANT; /opt/venv/bin/python /tmp/benchmark_ar_longrun_policy.py \
    --config /proj/SUPER-GIANT/TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/configs/config_500m.yml \
    --global_config /proj/SUPER-GIANT/TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/configs/Global_Config.yml \
    --draft_len 4 --policy full --steps 32:1600:32 --prefill_tokens 0 --temperature 1.0 --top_k 0 \
    --warmup_runs 1 --bench_runs 2 --model_label ar_500m_k4_full_prefix0_spot \
    --output_json /tmp/ar_full_prefix0_spot_500m.json --output_csv /tmp/ar_full_prefix0_spot_500m.csv'

- 3b / AR full-context precompiled baseline (light green line):
  ssh root@gpu-box-1 'cd /proj/SUPER-GIANT; /opt/venv/bin/python /tmp/benchmark_ar_longrun_policy.py \
    --config /proj/SUPER-GIANT/TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/configs/config_3b.yml \
    --global_config /proj/SUPER-GIANT/TiDAR/Docs/Benchmark_logs/kv_policy_a40_20260209_211437/configs/Global_Config.yml \
    --draft_len 16 --policy full --steps 32:1600:32 --prefill_tokens 0 --temperature 1.0 --top_k 0 \
    --warmup_runs 1 --bench_runs 2 --model_label ar_3b_k16_full_prefix0_spot \
    --output_json /tmp/ar_full_prefix0_spot_3b.json --output_csv /tmp/ar_full_prefix0_spot_3b.csv'

Artifacts copied to repo
- TiDAR/Docs/Benchmark_logs/bucket_prefix0_1600_a40_spot_20260210_091701/
  - raw/a40_spot_500m_k4.log
  - raw/a40_spot_3b_k16.log
  - raw/a40_spot_500m_k4_full.log
  - raw/a40_spot_3b_k16_full.log
  - raw/a40_spot_ar_500m.log
  - raw/a40_spot_ar_3b.log
  - parsed/bucket_prefix0_500m.csv
  - parsed/bucket_prefix0_3b.csv
  - parsed/full_prefix0_500m.csv
  - parsed/full_prefix0_3b.csv
  - parsed/ar_full_prefix0_500m.csv
  - parsed/ar_full_prefix0_3b.csv
  - parsed/bucket_prefix0_500m.json
  - parsed/bucket_prefix0_3b.json
"""
*/

#set page(width: 210mm, height: 297mm, margin: 12mm)
#set text(font: "New Computer Modern", size: 10pt)
#set heading(numbering: none)

#let c-red = rgb("#DC2626")
#let c-blue = rgb("#1D4ED8")
#let c-orange = rgb("#F59E0B")
#let c-green = rgb("#84CC16")
#let c-bound = rgb("#9CA3AF")

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
  height: 72mm,
  y-label: "steady tokens/s",
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

  let x-ticks = (32, 256, 512, 768, 1024, 1280, 1600)
  let y-ticks = range(0, 7)
  let bucket_edges = (128, 256, 512, 1024, 1536)

  block(width: width, height: height, inset: 5pt, stroke: 0.7pt + rgb("#D1D5DB"), radius: 6pt)[
    #text(weight: "semibold", size: 9.4pt)[#title]
    #v(1pt)

    #for edge in bucket_edges [
      #if edge >= min-x and edge <= max-x [
        #let px = margin-left + (edge - min-x) * sx
        #place(top + left, dx: px, dy: margin-top)[
          #line(length: plot-height, angle: 90deg, stroke: (paint: c-bound, thickness: 0.45pt, dash: (2pt, 2pt)))
        ]
      ]
    ]

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
        #text(size: 7pt, fill: rgb("#4B5563"))[#(str(calc.round(yv, digits: 0)))]
      ]
    ]

    #for xv in x-ticks [
      #let px = margin-left + (xv - min-x) * sx
      #place(top + left, dx: px, dy: margin-top + plot-height)[
        #line(length: 3pt, angle: 90deg, stroke: 0.6pt + black)
      ]
      #place(top + left, dx: px - 9pt, dy: margin-top + plot-height + 4pt)[
        #text(size: 7pt)[#(str(xv))]
      ]
    ]

    #for s in series [
      #place(top + left, dx: margin-left, dy: margin-top)[
        #path(
          stroke: 1.4pt + s.color,
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
        #place(top + left, dx: px - 1.25pt, dy: py - 1.25pt)[
          #circle(radius: 1.25pt, fill: s.color)
        ]
      ]
    ]

    #place(top + left, dx: margin-left, dy: margin-top + plot-height + 12pt)[
      #text(size: 7.8pt)[generated tokens]
    ]
    #place(top + left, dx: 0pt, dy: margin-top + plot-height / 2 - 12pt)[
      #rotate(-90deg)[#text(size: 7.8pt)[#y-label]]
    ]
  ]
}

#let m500_steady = ((32,1023.998711),(64,1098.827331),(96,1160.051361),(128,981.547668),(160,957.161739),(192,986.538686),(224,922.345066),(256,825.775492),(288,785.417626),(320,814.649501),(352,815.180307),(384,815.990748),(416,840.308621),(448,819.356661),(480,822.897426),(512,589.450028),(544,609.739382),(576,611.175457),(608,605.941673),(640,588.074410),(672,600.989939),(704,577.220119),(736,567.187762),(768,616.094176),(800,594.364347),(832,591.030695),(864,604.863500),(896,613.813608),(928,590.990801),(960,589.396653),(992,605.856333),(1024,468.409095),(1056,462.555231),(1088,462.542104),(1120,467.743629),(1152,459.642601),(1184,470.590243),(1216,460.974617),(1248,474.235846),(1280,465.949941),(1312,460.580543),(1344,454.590031),(1376,461.796164),(1408,451.372525),(1440,459.171557),(1472,457.643469),(1504,472.326003),(1536,384.857535),(1568,370.750068),(1600,382.025654))
#let m500_full_steady = ((32,351.595836),(64,370.413763),(96,391.972244),(128,371.526439),(160,361.300715),(192,372.426133),(224,350.688359),(256,383.385361),(288,365.108022),(320,379.938542),(352,378.587990),(384,377.324614),(416,389.728855),(448,381.256792),(480,383.592809),(512,375.479329),(544,384.220666),(576,387.876975),(608,385.586984),(640,374.736893),(672,384.417987),(704,369.181587),(736,359.400516),(768,393.717038),(800,375.022588),(832,374.520792),(864,383.529150),(896,389.492118),(928,375.314191),(960,375.607652),(992,385.020998),(1024,385.319028),(1056,379.936147),(1088,377.641977),(1120,383.271167),(1152,373.838112),(1184,386.986110),(1216,381.345215),(1248,388.202384),(1280,381.153389),(1312,377.340814),(1344,372.849580),(1376,378.042891),(1408,371.259714),(1440,375.307704),(1472,374.340545),(1504,384.595606),(1536,382.804849),(1568,369.479204),(1600,379.378272))
#let m500_ar_steady = ((32,404.244167),(64,408.566862),(96,410.355034),(128,410.157077),(160,409.861462),(192,411.520882),(224,411.344937),(256,411.999142),(288,412.053397),(320,412.421764),(352,411.291658),(384,411.341659),(416,411.909159),(448,412.738726),(480,412.652464),(512,412.261059),(544,413.461512),(576,413.838116),(608,412.558607),(640,410.751791),(672,411.009087),(704,412.009755),(736,413.822649),(768,412.109459),(800,412.336666),(832,413.444714),(864,412.834067),(896,411.934533),(928,413.454574),(960,412.715874),(992,412.303651),(1024,412.440032),(1056,412.481554),(1088,412.235350),(1120,412.477511),(1152,413.959094),(1184,412.181134),(1216,412.751654),(1248,412.975448),(1280,412.636525),(1312,411.999351),(1344,412.844444),(1376,412.890044),(1408,411.550614),(1440,412.448217),(1472,413.473629),(1504,412.676734),(1536,412.956204),(1568,413.251956),(1600,413.629982))
#let m3b_steady = ((32,2811.024542),(64,2926.033902),(96,2260.173522),(128,2497.424552),(160,2037.171336),(192,2284.233897),(224,2362.247054),(256,2349.476263),(288,1802.521596),(320,2000.592626),(352,2113.399899),(384,2064.205841),(416,2164.536210),(448,2335.896147),(480,1980.966377),(512,1703.280793),(544,1596.626218),(576,1686.903137),(608,1664.402422),(640,1749.293171),(672,1724.579923),(704,1629.849222),(736,1438.264040),(768,1658.382817),(800,1696.919240),(832,1554.345605),(864,1588.131017),(896,1650.324218),(928,1394.584376),(960,1499.634203),(992,1675.380793),(1024,1346.213349),(1056,1322.259317),(1088,1299.588796),(1120,1337.285065),(1152,1457.686619),(1184,1394.221921),(1216,1339.669915),(1248,1346.675205),(1280,1334.107266),(1312,1187.782459),(1344,1332.712840),(1376,1290.120083),(1408,1297.292981),(1440,1248.131618),(1472,1285.186336),(1504,1347.818617),(1536,1168.026273),(1568,1094.311227),(1600,1146.560866))
#let m3b_full_steady = ((32,1319.417800),(64,1337.646662),(96,1013.566365),(128,1205.096648),(160,971.451019),(192,1086.661095),(224,1118.271518),(256,1285.386777),(288,984.969002),(320,1089.135493),(352,1160.415932),(384,1129.673658),(416,1186.115006),(448,1272.744444),(480,1084.116374),(512,1184.394649),(544,1111.625846),(576,1172.921030),(608,1156.608991),(640,1219.119845),(672,1199.374795),(704,1138.458082),(736,1003.209168),(768,1155.551567),(800,1184.496242),(832,1081.662330),(864,1106.925392),(896,1148.151100),(928,971.067631),(960,1042.974754),(992,1167.283963),(1024,1140.472674),(1056,1120.183219),(1088,1099.432942),(1120,1131.482837),(1152,1237.005753),(1184,1182.596908),(1216,1136.003386),(1248,1140.679477),(1280,1131.486745),(1312,1003.595364),(1344,1131.057289),(1376,1094.402265),(1408,1099.058911),(1440,1056.193365),(1472,1090.675473),(1504,1143.190369),(1536,1166.302791),(1568,1093.801735),(1600,1144.035015))
#let m3b_ar_steady = ((32,385.643214),(64,407.254909),(96,408.917945),(128,406.401600),(160,409.272994),(192,411.305584),(224,412.537214),(256,411.763885),(288,411.925145),(320,410.974147),(352,411.835886),(384,412.587637),(416,411.904024),(448,412.212342),(480,412.043317),(512,413.253439),(544,411.425263),(576,410.837902),(608,411.379001),(640,411.756449),(672,411.707946),(704,412.393939),(736,413.247294),(768,413.182382),(800,412.685575),(832,412.010219),(864,411.351150),(896,412.462582),(928,412.691885),(960,412.559608),(992,412.346264),(1024,412.649726),(1056,413.499102),(1088,412.108307),(1120,412.193760),(1152,411.159135),(1184,411.767064),(1216,411.623208),(1248,412.786796),(1280,411.876294),(1312,412.529947),(1344,412.548790),(1376,411.539972),(1408,412.857282),(1440,412.895590),(1472,412.382641),(1504,412.942701),(1536,413.659745),(1568,412.925767),(1600,412.594604))

= A40 Spot: Bucketed Prefix=0 Long Decode (to 1600 tokens)

#legend-item("500m bucketed, draft_len=4", c-blue) #h(10pt) #legend-item("full-context TiDAR baseline", c-red) #h(10pt) #legend-item("AR baseline", c-green) #h(10pt) #legend-item("3b bucketed, draft_len=16", c-orange)

#v(4pt)
#line-chart(
  "500m / draft_len=4 / steady decode throughput (compile excluded)",
  (
    (label: "500m bucketed", color: c-blue, data: m500_steady),
    (label: "500m full-context precompiled", color: c-red, data: m500_full_steady),
    (label: "500m AR full-context precompiled", color: c-green, data: m500_ar_steady),
  ),
)

#v(4pt)
#line-chart(
  "3b / draft_len=16 / steady decode throughput (compile excluded)",
  (
    (label: "3b bucketed", color: c-orange, data: m3b_steady),
    (label: "3b full-context precompiled", color: c-red, data: m3b_full_steady),
    (label: "3b AR full-context precompiled", color: c-green, data: m3b_ar_steady),
  ),
)

#v(4pt)
- Prefix length: `0` (synthetic prefill)
- Steps swept: `32..1600` by `32`
- Acceptance rate target: `0.8`
- Warmup/bench: `1/2`
- Unique bucket compile keys: `6` (128, 256, 512, 1024, 1536, 2048)
- Mean steady TPS:
  - 500m/k4: `635.480`
  - 3b/k16: `1689.529`
  - AR 500m/k4: `412.103`
  - AR 3b/k16: `411.368`

- Vertical dashed lines are bucket transitions (`128`, `256`, `512`, `1024`, `1536`).
- Metric shown is steady decode TPS (compile excluded): measured after warmup on compiled executables.
