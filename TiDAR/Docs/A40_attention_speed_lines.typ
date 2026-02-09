#set page(width: 210mm, height: 297mm, margin: 12mm)
#set text(font: "New Computer Modern", size: 10pt)
#set heading(numbering: none)

#let c-k-structured = rgb("#1D4ED8")
#let c-k-dense = rgb("#DC2626")
#let c-k-densezero = rgb("#059669")
#let c-ar = rgb("#7C3AED")
#let c-n-structured = rgb("#1D4ED8")
#let c-n-dense = rgb("#DC2626")
#let c-mlp-ar = rgb("#7C3AED")
#let c-mlp-tidar = rgb("#0F766E")

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
  height: 62mm,
  x-label: "draft_len (K)",
  y-label: "ms / attention",
) = {
  let margin-left = 28pt
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

  let x-ticks = (2, 4, 8, 16)
  let y-ticks = range(0, 6)

  block(width: width, height: height, inset: 5pt, stroke: 0.7pt + rgb("#D1D5DB"), radius: 6pt)[
    #text(weight: "semibold", size: 9.4pt)[#title]
    #v(2pt)
    #place(top + left, dx: margin-left, dy: margin-top + plot-height)[
      #line(length: plot-width, stroke: 0.6pt + black)
    ]
    #place(top + left, dx: margin-left, dy: margin-top)[
      #line(length: plot-height, angle: 90deg, stroke: 0.6pt + black)
    ]

    #for t in y-ticks [
      #let yv = min-y + (max-y - min-y) * (t / 5)
      #let py = margin-top + plot-height - (yv - min-y) * sy
      #place(top + left, dx: margin-left, dy: py)[
        #line(length: plot-width, stroke: (paint: rgb("#E5E7EB"), thickness: 0.45pt, dash: (2pt, 2pt)))
      ]
      #place(top + left, dx: 1pt, dy: py - 4pt)[
        #text(size: 7pt, fill: rgb("#4B5563"))[#(str(calc.round(yv, digits: 3)))]
      ]
    ]

    #for xv in x-ticks [
      #let px = margin-left + (xv - min-x) * sx
      #place(top + left, dx: px, dy: margin-top + plot-height)[
        #line(length: 3pt, angle: 90deg, stroke: 0.6pt + black)
      ]
      #place(top + left, dx: px - 4pt, dy: margin-top + plot-height + 4pt)[
        #text(size: 8pt)[#(str(xv))]
      ]
    ]

    #for s in series [
      #place(top + left, dx: margin-left, dy: margin-top)[
        #path(
          stroke: 1.2pt + s.color,
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
    #place(top + left, dx: 0pt, dy: margin-top + plot-height / 2 - 16pt)[
      #rotate(-90deg)[#text(size: 7.8pt)[#y-label]]
    ]
  ]
}

#let chart-block(title, series) = [
  #block(breakable: false)[
    #text(weight: "semibold", size: 9.4pt)[#title]
    #v(1pt)
    #for s in series [
      #legend-item(s.label, s.color)
      #h(9pt)
    ]
    #v(1pt)
    #line-chart("", series)
  ]
]

#let k1b = (
  (label: "kernel structured", color: c-k-structured, data: ((2, 0.107145), (4, 0.126596), (8, 0.152638), (16, 0.181612))),
  (label: "kernel dense", color: c-k-dense, data: ((2, 0.071736), (4, 0.070505), (8, 0.074565), (16, 0.087710))),
  (label: "kernel dense+jax-zero", color: c-k-densezero, data: ((2, 0.038555), (4, 0.049070), (8, 0.086636), (16, 0.256265))),
  (label: "AR baseline (K x len1)", color: c-ar, data: ((2, 0.065535), (4, 0.131070), (8, 0.262139), (16, 0.524278))),
)

#let n1b = (
  (label: "native structured", color: c-n-structured, data: ((2, 0.412120), (4, 0.433505), (8, 0.451159), (16, 0.543223))),
  (label: "native dense", color: c-n-dense, data: ((2, 0.391434), (4, 0.410521), (8, 0.432730), (16, 0.547173))),
  (label: "AR baseline (K x len1)", color: c-ar, data: ((2, 0.522547), (4, 1.045094), (8, 2.090188), (16, 4.180376))),
)

#let k3b = (
  (label: "kernel structured", color: c-k-structured, data: ((2, 0.075724), (4, 0.076048), (8, 0.081382), (16, 0.097438))),
  (label: "kernel dense", color: c-k-dense, data: ((2, 0.069547), (4, 0.069569), (8, 0.074939), (16, 0.088723))),
  (label: "kernel dense+jax-zero", color: c-k-densezero, data: ((2, 0.048804), (4, 0.058910), (8, 0.085520), (16, 0.225689))),
  (label: "AR baseline (K x len1)", color: c-ar, data: ((2, 0.087481), (4, 0.174962), (8, 0.349925), (16, 0.699849))),
)

#let n3b = (
  (label: "native structured", color: c-n-structured, data: ((2, 0.531247), (4, 0.540576), (8, 0.554605), (16, 0.706373))),
  (label: "native dense", color: c-n-dense, data: ((2, 0.527241), (4, 0.535066), (8, 0.547967), (16, 0.699304))),
  (label: "AR baseline (K x len1)", color: c-ar, data: ((2, 0.697389), (4, 1.394777), (8, 2.789555), (16, 5.579110))),
)

#let k7b = (
  (label: "kernel structured", color: c-k-structured, data: ((2, 0.075419), (4, 0.078492), (8, 0.082303), (16, 0.171602))),
  (label: "kernel dense", color: c-k-dense, data: ((2, 0.070653), (4, 0.072270), (8, 0.075352), (16, 0.157649))),
  (label: "kernel dense+jax-zero", color: c-k-densezero, data: ((2, 0.056792), (4, 0.067157), (8, 0.109234), (16, 0.288356))),
  (label: "AR baseline (K x len1)", color: c-ar, data: ((2, 0.100167), (4, 0.200334), (8, 0.400669), (16, 0.801337))),
)

#let n7b = (
  (label: "native structured", color: c-n-structured, data: ((2, 0.793903), (4, 0.793006), (8, 0.815991), (16, 1.071996))),
  (label: "native dense", color: c-n-dense, data: ((2, 0.781417), (4, 0.786742), (8, 0.811170), (16, 1.061794))),
  (label: "AR baseline (K x len1)", color: c-ar, data: ((2, 1.315679), (4, 2.631358), (8, 5.262716), (16, 10.525433))),
)

= A40 Decode Attention Speed (Final)

Runs: `A40 spot`, `bf16`, `implementation=cudnn`, `prefix_len=1024`, `iterations=100`, `repeats=5`, `batch_size=1`.

#text(size: 8.5pt, fill: rgb("#4B5563"))[
Pallas series is intentionally omitted from these plots to keep the non-pallas curves readable on a single scale.
AR baseline is measured at batch=1, query_len=1 and scaled by `K` for each draft length point.
]

#v(7pt)

#chart-block("1B-style MHA (32/32, head_dim 64) - Kernel terms", k1b)
#v(7pt)
#chart-block("1B-style MHA (32/32, head_dim 64) - Native decode", n1b)
#v(7pt)
#chart-block("3B-style GQA (24/8, head_dim 128) - Kernel terms", k3b)
#v(7pt)
#chart-block("3B-style GQA (24/8, head_dim 128) - Native decode", n3b)
#v(7pt)
#chart-block("7B-style GQA (32/8, head_dim 128) - Kernel terms", k7b)
#v(7pt)
#chart-block("7B-style GQA (32/8, head_dim 128) - Native decode", n7b)

#pagebreak()

= A40 MLP Throughput — AR vs TiDAR Tokens

This section compares MLP-only cost using SwiGLU projections. AR baseline is measured on `len=1` and scaled by `K`, while TiDAR runs the full `L = K + K^2` token block in one pass.

#let mlp1b = (
  (label: "TiDAR MLP at L=K+K^2", color: c-mlp-tidar, data: ((2, 0.184439), (4, 0.193440), (8, 0.210385), (16, 0.406673))),
  (label: "AR MLP baseline (K x len1)", color: c-mlp-ar, data: ((2, 0.368354), (4, 0.736707), (8, 1.473414), (16, 2.946828))),
)

#let mlp3b = (
  (label: "TiDAR MLP at L=K+K^2", color: c-mlp-tidar, data: ((2, 0.264362), (4, 0.279314), (8, 0.304600), (16, 0.595783))),
  (label: "AR MLP baseline (K x len1)", color: c-mlp-ar, data: ((2, 0.539437), (4, 1.078875), (8, 2.157750), (16, 4.315500))),
)

#let mlp7b = (
  (label: "TiDAR MLP at L=K+K^2", color: c-mlp-tidar, data: ((2, 0.462071), (4, 0.484772), (8, 0.554393), (16, 1.085345))),
  (label: "AR MLP baseline (K x len1)", color: c-mlp-ar, data: ((2, 0.926167), (4, 1.852334), (8, 3.704667), (16, 7.409334))),
)

#chart-block("1B-style MHA (d_model=2048, d_ff=8192) - MLP only", mlp1b)
#v(7pt)
#chart-block("3B-style GQA (d_model=3072, d_ff=8192) - MLP only", mlp3b)
#v(7pt)
#chart-block("7B-style GQA (d_model=4096, d_ff=11008) - MLP only", mlp7b)
