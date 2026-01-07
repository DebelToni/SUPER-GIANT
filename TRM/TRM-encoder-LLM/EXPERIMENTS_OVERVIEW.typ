#set page(width: 210mm, height: 297mm, margin: (x: 15mm, y: 15mm))
#set text(font: "Helvetica", size: 10pt)
#set heading(numbering: none)

= TRM Encoder + SmolLM Cross-Attn Experiments (Overview)

#figure(
  image("TRM-encoder-architecture.png", width: 100%),
  caption: "TRM encoder feeds a pretrained decoder via cross-attention.",
)

The encoder compresses a long prompt into a fixed slot sequence, applies TRM recursive refinement, then expands to a decoder-length memory. The decoder receives this memory through cross-attention (optionally restricted to early layers) and we compare validation loss with cross-attention enabled vs disabled.

== Experiment summary
Summary compares three setups: a short 30-layer baseline run, a longer run with larger gate init, and a recursive encoder run with cross-attn only in early layers and dropout.
#table(
  columns: 6,
  align: (left, left, right, right, right, right),
  [*Experiment*], [*Key changes*], [*Steps*], [*Final cross*], [*Final no-cross*], [*Delta*],
  [Initial 30L], [short run, encoder 512/128], [200], [1.5510], [1.6283], [+0.0773],
  [Medium-long], [gate init -2.0, encoder 512/128], [2000], [2.3770], [2.3774], [-0.0004],
  [Recursive 5090], [recursive Z, cross on layers 0-5, dropout 0.1], [2000], [2.0902], [2.0902], [+0.0000],
)

#pagebreak()

== Initial 30-layer run (200 steps)
Short 200-step run with 30-layer decoder, encoder 512/128, standard gates, no cross-attn dropout.
#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    Table for the short-run setup described above.
    #table(
      columns: 4,
      align: (right, right, right, right),
      [*Step*], [*Cross*], [*No-cross*], [*Delta*],
      [50],  [1.8221], [1.8156], [-0.0065],
      [100], [1.9193], [1.9388], [+0.0195],
      [150], [1.4897], [1.5367], [+0.0470],
      [200], [1.5510], [1.6283], [+0.0773],
    )
  ],
  [
    Chart for the short-run setup described above.
    #figure(
      image("figs/exp1_eval.png", width: 100%),
      caption: "Exp 1: loss vs step.",
    )
  ],
)

== Medium-long run (2000 steps)
Longer 2000-step run with 30-layer decoder, encoder 512/128, gate init -2.0, analytics on, no-cross eval.
#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    Table for the medium-long setup described above.
    #table(
      columns: 4,
      align: (right, right, right, right),
      [*Step*], [*Cross*], [*No-cross*], [*Delta*],
      [200],  [2.2036], [2.2415], [+0.0379],
      [400],  [2.4623], [2.4785], [+0.0162],
      [600],  [1.7670], [1.7717], [+0.0047],
      [800],  [1.8464], [1.8467], [+0.0003],
      [1000], [2.0721], [2.0736], [+0.0015],
      [1200], [2.1530], [2.1517], [-0.0013],
      [1400], [2.2641], [2.2610], [-0.0031],
      [1600], [2.2352], [2.2352], [+0.0000],
      [1800], [2.1117], [2.1104], [-0.0013],
      [2000], [2.3770], [2.3774], [+0.0004],
    )
  ],
  [
    Chart for the medium-long setup described above.
    #figure(
      image("figs/exp2_eval.png", width: 100%),
      caption: "Exp 2: loss vs step.",
    )
  ],
)

#pagebreak()

== Recursive 5090 run (2000 steps)
Recursive encoder updates (stride 64, short cycles 1/1, carry Z only), cross-attn limited to layers 0-5 with dropout 0.1.
#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    Chart for the recursive setup described above.
    #figure(
      image("figs/recursive_eval.png", width: 100%),
      caption: "Recursive run validation loss with/without cross-attn.",
    )
  ],
  [
    Table for the recursive setup described above.
    #table(
      columns: 4,
      align: (right, right, right, right),
      [*Step*], [*Cross*], [*No-cross*], [*Delta*],
      [200],  [1.3943], [1.4876], [+0.0933],
      [400],  [1.8237], [1.8965], [+0.0728],
      [600],  [2.3558], [2.3587], [+0.0029],
      [800],  [2.5846], [2.5832], [-0.0014],
      [1000], [2.3039], [2.3034], [-0.0005],
      [1200], [2.3475], [2.3482], [+0.0007],
      [1400], [2.0459], [2.0458], [-0.0001],
      [1600], [2.5572], [2.5571], [-0.0001],
      [1800], [2.4067], [2.4066], [-0.0001],
      [2000], [2.0902], [2.0902], [+0.0000],
    )
  ],
)

== Interpretation
Across all runs, cross-attention provides a small early boost, then collapses to near-zero delta. The decoder learns to minimize loss while effectively ignoring cross-attn, even with recursive encoder updates and cross-attn restricted to early layers. This supports the hypothesis that a pretrained decoder will discard alien cross-attn to recover baseline next-token performance.
