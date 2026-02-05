#set page(width: 297mm, height: 210mm, margin: 12mm)
#set text(font: "New Computer Modern", size: 10pt)

#let bg = rgb("#F5F7FB")
#let ink = rgb("#1F2430")
#let good = rgb("#1B9E77")
#let mid = rgb("#4C78A8")
#let bad = rgb("#D95F02")

#set page(fill: bg)
#set text(fill: ink)

#let card(title, body) = block(
  inset: 7pt,
  radius: 7pt,
  fill: white,
  stroke: (paint: rgb("#D9DFEA"), thickness: 0.7pt),
  width: 100%,
)[
  #text(size: 9pt, fill: rgb("#556070"))[#title]
  #v(4pt)
  #body
]

#let barpair(label, a, b, maxv) = {
  let wa = (a / maxv) * 210pt
  let wb = (b / maxv) * 210pt
  [
    #text(size: 8.5pt, fill: rgb("#4A5566"))[#label]
    #v(2pt)
    #stack(dir: ltr, spacing: 8pt)[
      #box(width: 56pt)[BigGamma]
      #block(width: wa, height: 10pt, fill: good, radius: 2pt)[]
      #text(size: 8.5pt)[#a]
    ]
    #v(1pt)
    #stack(dir: ltr, spacing: 8pt)[
      #box(width: 56pt)[TopK]
      #block(width: wb, height: 10pt, fill: mid, radius: 2pt)[]
      #text(size: 8.5pt)[#b]
    ]
  ]
}

#align(center)[
  #text(size: 18pt, weight: "bold")[TiDAR 105k Inference Head-to-Head]
  #linebreak()
  #text(size: 10pt, fill: rgb("#5B6677"))[Acceptance-throughput dashboard]
]

#v(8pt)

#grid(columns: (1fr, 1fr, 1fr, 1fr), gutter: 8pt,
  card([Overall Winner], [
    #text(size: 17pt, weight: "bold", fill: good)[BigGamma+Delta]
    #linebreak()
    #text(size: 9pt)[based on inference accept rate]
  ]),
  card([Mean `avg_accept/iter`], [
    #text(size: 16pt, weight: "bold")[2.2981 vs 2.2117]
    #linebreak()
    #text(size: 9pt, fill: good)[+0.0864 to BigGamma]
  ]),
  card([Accept-Prob Proxy], [
    #text(size: 16pt, weight: "bold")[(2.2981-1)/5 = 0.2596]
    #linebreak()
    #text(size: 9pt)[TopK: 0.2423]
  ]),
  card([Pairwise Wins], [
    #text(size: 16pt, weight: "bold")[19 / 36]
    #linebreak()
    #text(size: 9pt)[BigGamma better runs]
  ]),
)

#v(8pt)

#grid(columns: (1fr, 1fr), gutter: 10pt,
  card([Primary Sweep (12 prompts, 96 steps)], [
    #barpair("Greedy", 2.3092, 2.1942, 2.5)
    #v(5pt)
    #barpair("Sampled seed 0", 2.2900, 2.2250, 2.5)
    #v(5pt)
    #barpair("Sampled seed 1", 2.2950, 2.2158, 2.5)
  ]),
  card([Robustness (12 prompts, 256 greedy steps)], [
    #barpair("Long-run greedy", 2.3300, 1.9833, 2.5)
    #v(7pt)
    #text(size: 9pt)[Mean delta: ]#text(fill: good, weight: "bold")[+0.3467]
    #v(3pt)
    #text(size: 9pt)[Prompt winners: BigGamma 8, TopK 4]
  ]),
)

#v(8pt)

#card([Quick Take], [
  #text(size: 11pt, weight: "bold")[If priority is inference acceptance throughput, continue BigGamma+Delta to 121k.]
  #v(4pt)
  #text(size: 9pt, fill: bad)[Caveat: monitor repetition/quality in long greedy runs while optimizing acceptance.]
])

#v(4pt)
#text(size: 8pt, fill: rgb("#6A7484"))[
Checkpoints compared at equal step 105000:
`.../tidar_bigGamma_andDelta/params/step_0105000.npz` vs
`.../tidar_later_stage_topk/params/step_0105000.npz`.
]
