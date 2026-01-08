#set page(width: 210mm, height: 297mm, margin: (x: 15mm, y: 15mm))
#set text(font: "Helvetica", size: 10pt)
#set heading(numbering: none)

#let line-chart(
  data,
  width: 130mm,
  height: 60mm,
  stroke: 1pt + blue,
  x_label: "Step",
  y_label: "Train-mix loss",
) = {
  let margin-left = 18pt
  let margin-right = 6pt
  let margin-top = 4pt
  let margin-bottom = 16pt
  let y-label-gap = 4pt
  let plot-width = width - margin-left - margin-right
  let plot-height = height - margin-top - margin-bottom
  let xs = data.map(d => d.at(0))
  let ys = data.map(d => d.at(1))
  let min-x = calc.min(..xs)
  let max-x = calc.max(..xs)
  let min-y = calc.min(..ys)
  let max-y = calc.max(..ys)
  let pad = (max-y - min-y) * 0.08
  let min-y = min-y - pad
  let max-y = max-y + pad
  let scale-x = if max-x == min-x { plot-width } else { plot-width / (max-x - min-x) }
  let scale-y = if max-y == min-y { plot-height } else { plot-height / (max-y - min-y) }

  box(width: width, height: height)[
    #place(top + left, dx: margin-left, dy: margin-top + plot-height)[
      #line(length: plot-width, stroke: 0.6pt + black)
    ]
    #place(top + left, dx: margin-left, dy: margin-top)[
      #line(length: plot-height, angle: 90deg, stroke: 0.6pt + black)
    ]
    #place(top + left, dx: margin-left, dy: margin-top)[
      #path(
        stroke: stroke,
        fill: none,
        ..data.map(d => (
          (d.at(0) - min-x) * scale-x,
          plot-height - (d.at(1) - min-y) * scale-y
        ))
      )
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

#let bar-chart(
  data,
  width: 130mm,
  height: 55mm,
  bar-color: rgb(20%, 50%, 80%),
  x_label: "Run",
  y_label: "Value",
) = {
  let margin-left = 18pt
  let margin-right = 6pt
  let margin-top = 4pt
  let margin-bottom = 24pt
  let y-label-gap = 4pt
  let plot-width = width - margin-left - margin-right
  let plot-height = height - margin-top - margin-bottom
  let values = data.map(d => d.at(1))
  let labels = data.map(d => d.at(0))
  let min-v = calc.min(..values)
  let max-v = calc.max(..values)
  let mixed = min-v <= 0 and max-v >= 0
  let scale = if mixed {
    plot-height / (max-v - min-v)
  } else if min-v > 0 {
    plot-height / max-v
  } else {
    plot-height / (0 - min-v)
  }
  let y0 = if mixed { max-v * scale } else if min-v > 0 { plot-height } else { 0pt }
  let gap = 4pt
  let bar-width = (plot-width - gap * (data.len() + 1)) / data.len()

  box(width: width, height: height)[
    #place(top + left, dx: margin-left, dy: margin-top + y0)[
      #line(length: plot-width, stroke: 0.6pt + black)
    ]
    #place(top + left, dx: margin-left, dy: margin-top)[
      #line(length: plot-height, angle: 90deg, stroke: 0.6pt + black)
    ]
    #for i in range(0, data.len()) [
      #let val = values.at(i)
      #let bar-h = calc.abs(val) * scale
      #let x = margin-left + gap + i * (bar-width + gap)
      #let y = if val >= 0 { y0 - bar-h } else { y0 }
      #place(top + left, dx: x, dy: margin-top + y)[
        #rect(width: bar-width, height: bar-h, fill: bar-color)
      ]
      #place(top + left, dx: x, dy: margin-top + plot-height + 2pt)[
        #box(width: bar-width, height: margin-bottom - 6pt)[
          #align(center)[#text(size: 7pt)[#(labels.at(i))]]
        ]
      ]
    ]
    #place(top + left, dx: margin-left, dy: height - margin-bottom + 8pt)[
      #box(width: plot-width, height: margin-bottom - 8pt)[
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

#let loss-curve-v4 = (
  (1000, 1.0471),
  (2000, 0.5969),
  (3000, 0.6157),
  (4000, 0.6111),
  (5000, 0.9905),
  (6000, 0.8445),
  (7000, 0.6101),
  (8000, 0.4612),
  (9000, 0.6068),
  (10000, 0.7350),
)

#let delta-mix = (
  ("v1", -0.0089),
  ("v2", 0.0251),
  ("v3", 0.0345),
  ("v4", 0.0364),
)

#let ood-success = (
  ("Homework", 1.0),
  ("Travel", 1.0),
  ("Cooking", 1.0),
  ("History", 1.0),
  ("Budget", 1.0),
)

= TRM Token Tool Experiments (SmolLM-135M Sudoku Calls)

#figure(
  image("LLM-token-TRM.png", width: 100%),
  caption: "TRM token call: the model emits a Sudoku payload between special tags.",
)

== Overview
SmolLM-135M is fine-tuned to emit `<TRM-sudoku> ... </TRM-sudoku>` calls so an external
Python solver can handle Sudoku. The dataset mixes Sudoku prompts with chat and general
text to reduce catastrophic forgetting. Training uses L2 regularization toward the base
checkpoint and bfloat16 compute.

#pagebreak()

== Experiment summary
#table(
  columns: 8,
  align: (left, left, right, left, right, right, right, left),
  [*Experiment*], [*Dataset*], [*Train rows*], [*Mix (S/C/T)*], [*Steps*], [*L2*],
  [*Simple wiki delta*], [*OOD prompt*],
  [Sudoku only], [trm_sudoku_calls_v1], [5000], [1.0/0.0/0.0], [5000], [0],
  [+0.8800], [n/a],
  [Mix v1], [trm_sudoku_mix_v1], [8000], [0.3/0.2/0.5], [8000], [1e-5],
  [-0.0089], [0/3 \@ 256],
  [Mix v2], [trm_sudoku_mix_v2], [10000], [0.4/0.2/0.4], [10000], [1e-5],
  [+0.0251], [0/3 \@ 256],
  [Mix v3], [trm_sudoku_mix_v3], [10000], [0.45/0.2/0.35], [10000], [1e-5],
  [+0.0345], [0/3 \@ 256],
  [Mix v4 (current)], [trm_sudoku_mix_v4], [10000], [0.5/0.2/0.3], [10000], [1e-5],
  [+0.0364], [3/3 \@ 512],
)

== Loss over time (mix v4)
Loss curve is evaluated on 10 batches (batch_size=2, max_rows=256) from the v4 train mix
at each 1k checkpoint.
#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    #table(
      columns: 2,
      align: (right, right),
      [*Step*], [*Train-mix loss*],
      [1000], [1.0471],
      [2000], [0.5969],
      [3000], [0.6157],
      [4000], [0.6111],
      [5000], [0.9905],
      [6000], [0.8445],
      [7000], [0.6101],
      [8000], [0.4612],
      [9000], [0.6068],
      [10000], [0.7350],
    )
  ],
  [
    #figure(
      line-chart(
        loss-curve-v4,
        width: 100%,
        x_label: "Step",
        y_label: "Train loss",
      ),
      caption: "Mix v4 loss snapshots vs checkpoint step.",
    )
  ],
)

== Performance summary
Mix runs keep general loss near the base model while improving OOD tool usage.
#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    #figure(
      bar-chart(
        delta-mix,
        width: 100%,
        bar-color: rgb(25%, 55%, 85%),
        x_label: "Mix version (v1-v4)",
        y_label: "SimpleWiki delta",
      ),
      caption: "Simple wiki delta (fine - base), mix runs only.",
    )
  ],
  [
    #figure(
      bar-chart(
        ood-success,
        width: 100%,
        bar-color: rgb(20%, 60%, 35%),
        x_label: "OOD prompt theme",
        y_label: "Success rate",
      ),
      caption: "OOD prompt success (10 runs each).",
    )
  ],
)

== Evaluation protocol
- General loss: `Run_eval_general_loss.py` on simple_wiki (20 batches, seq_len 256).
- OOD prompts: 5 long mixed-topic prompts from `trm_token_tool/ood_eval/prompts.json`,
  10 runs each with `Run_eval_ood_sudoku.py`; success means the parsed puzzle matches
  the input and the solver matches the generator solution.
