#set page(width: 210mm, height: 297mm, margin: 18mm)
#set page(width: 210mm, height: 297mm, margin: 18mm)
#set text(size: 11pt)
#set heading(numbering: none)

#let color-total = rgb(220, 50, 47)
#let color-ntp = rgb(33, 150, 243)
#let color-diff = rgb(244, 162, 97)
#let color-acc = rgb(46, 204, 113)

#let legend-item(label, color) = [
  #rect(width: 10pt, height: 10pt, fill: color, radius: 2pt)
  #h(4pt)
  #text(size: 8pt)[#(label)]
]

#let metric-legend = box(width: 100%, inset: 6pt, stroke: 0.6pt + rgb(200, 200, 200), radius: 4pt)[
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 8pt,
    align: left,
    [#legend-item("Total loss", color-total)],
    [#legend-item("NTP loss", color-ntp)],
    [#legend-item("Diff loss", color-diff)],
    [#legend-item("Accept rate", color-acc)],
  )
]

#let multi-line-chart(
  series,
  width: 170mm,
  height: 60mm,
  border: rgb(180, 180, 180),
  x_label: "Step",
  y_label: "Metric",
) = {
  let margin-left = 22pt
  let margin-right = 8pt
  let margin-top = 6pt
  let margin-bottom = 18pt
  let y-label-gap = 4pt
  let plot-width = width - margin-left - margin-right
  let plot-height = height - margin-top - margin-bottom

  let min-x = calc.min(..series.map(s => calc.min(..s.data.map(d => d.at(0)))))
  let max-x = calc.max(..series.map(s => calc.max(..s.data.map(d => d.at(0)))))
  let min-y = calc.min(..series.map(s => calc.min(..s.data.map(d => d.at(1)))))
  let max-y = calc.max(..series.map(s => calc.max(..s.data.map(d => d.at(1)))))
  let pad = (max-y - min-y) * 0.08
  let min-y = min-y - pad
  let max-y = max-y + pad
  let scale-x = if max-x == min-x { plot-width } else { plot-width / (max-x - min-x) }
  let scale-y = if max-y == min-y { plot-height } else { plot-height / (max-y - min-y) }

  box(width: width, height: height, stroke: 0.8pt + border, radius: 6pt, inset: 4pt)[
    #place(top + left, dx: margin-left, dy: margin-top + plot-height)[
      #line(length: plot-width, stroke: 0.6pt + black)
    ]
    #place(top + left, dx: margin-left, dy: margin-top)[
      #line(length: plot-height, angle: 90deg, stroke: 0.6pt + black)
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

#let sweep1-loss = ((500, 4.3628), (1000, 4.3013), (1500, 4.2282), (2000, 4.0440), (2500, 3.9061), (3000, 3.7951), (3500, 4.0864), (4000, 3.6130), (4500, 4.2244), (5000, 3.9442), (5500, 3.9334), (6000, 3.8638), (6500, 3.9011), (7000, 3.6011), (7500, 3.8506), (8000, 3.6267), (8500, 3.0966), (9000, 3.1776), (9500, 3.3967), (10000, 3.2198), (10500, 3.3218), (11000, 3.2222), (11500, 3.3432), (12000, 2.4401))
#let sweep1-ntp = ((500, 2.3146), (1000, 2.8199), (1500, 2.8795), (2000, 2.8083), (2500, 2.6764), (3000, 2.5492), (3500, 2.8048), (4000, 2.3767), (4500, 3.0371), (5000, 2.7761), (5500, 2.8521), (6000, 2.6978), (6500, 2.7316), (7000, 2.4241), (7500, 2.7020), (8000, 2.5456), (8500, 1.7269), (9000, 1.8192), (9500, 2.0024), (10000, 1.9145), (10500, 2.1788), (11000, 2.2716), (11500, 2.2648), (12000, 1.5233))
#let sweep1-diff = ((500, 6.4111), (1000, 5.7827), (1500, 5.5768), (2000, 5.2796), (2500, 5.1358), (3000, 5.0409), (3500, 5.3679), (4000, 4.8493), (4500, 5.4117), (5000, 5.1122), (5500, 5.0147), (6000, 5.0298), (6500, 5.0706), (7000, 4.7782), (7500, 4.9992), (8000, 4.7077), (8500, 4.4663), (9000, 4.5361), (9500, 4.7910), (10000, 4.5252), (10500, 4.4648), (11000, 4.1728), (11500, 4.4215), (12000, 3.3569))
#let sweep1-acc = ((500, 0.1900), (1000, 0.3240), (1500, 0.3340), (2000, 0.3430), (2500, 0.3550), (3000, 0.3530), (3500, 0.3720), (4000, 0.3610), (4500, 0.3600), (5000, 0.3760), (5500, 0.3820), (6000, 0.3590), (6500, 0.3610), (7000, 0.3520), (7500, 0.3490), (8000, 0.3830), (8500, 0.3310), (9000, 0.3260), (9500, 0.3460), (10000, 0.3310), (10500, 0.3780), (11000, 0.4380), (11500, 0.3770), (12000, 0.3930))

#let sweep2-loss = ((500, 5.5274), (1000, 5.0554), (1500, 4.8927), (2000, 4.6620), (2500, 4.5236), (3000, 4.4276), (3500, 4.7210), (4000, 4.2407), (4500, 4.8165), (5000, 4.5329), (5500, 4.5001), (6000, 4.4537), (6500, 4.4919), (7000, 4.2112), (7500, 4.4320), (8000, 4.1902), (8500, 3.8234), (9000, 3.9014), (9500, 4.1343), (10000, 3.9342), (10500, 3.9537), (11000, 3.7948), (11500, 3.9716), (12000, 3.0996))
#let sweep2-ntp = ((500, 2.3179), (1000, 2.8289), (1500, 2.8897), (2000, 2.8182), (2500, 2.6854), (3000, 2.5575), (3500, 2.8143), (4000, 2.3893), (4500, 3.0415), (5000, 2.7791), (5500, 2.8658), (6000, 2.7134), (6500, 2.7392), (7000, 2.4446), (7500, 2.7074), (8000, 2.5539), (8500, 1.7390), (9000, 1.8229), (9500, 2.0093), (10000, 1.9147), (10500, 2.1925), (11000, 2.2819), (11500, 2.2847), (12000, 1.5425))
#let sweep2-diff = ((500, 6.9159), (1000, 5.9797), (1500, 5.6756), (2000, 5.3469), (2500, 5.1992), (3000, 5.1065), (3500, 5.4221), (4000, 4.9119), (4500, 5.4544), (5000, 5.1638), (5500, 5.0774), (6000, 5.0726), (6500, 5.1212), (7000, 4.8332), (7500, 5.0431), (8000, 4.7622), (8500, 4.5522), (9000, 4.6302), (9500, 4.8815), (10000, 4.6247), (10500, 4.5506), (11000, 4.2949), (11500, 4.5429), (12000, 3.5657))
#let sweep2-acc = ((500, 0.1640), (1000, 0.3000), (1500, 0.3220), (2000, 0.3200), (2500, 0.3450), (3000, 0.3340), (3500, 0.3550), (4000, 0.3460), (4500, 0.3410), (5000, 0.3630), (5500, 0.3640), (6000, 0.3460), (6500, 0.3450), (7000, 0.3340), (7500, 0.3330), (8000, 0.3650), (8500, 0.3090), (9000, 0.2980), (9500, 0.3170), (10000, 0.3050), (10500, 0.3510), (11000, 0.3920), (11500, 0.3600), (12000, 0.3410))

#let sweep3-loss = ((500, 9.3719), (1000, 7.2575), (1500, 6.8415), (2000, 6.5122), (2500, 6.3550), (3000, 6.3192), (3500, 6.6234), (4000, 6.1036), (4500, 6.6068), (5000, 6.2987), (5500, 6.1803), (6000, 6.2115), (6500, 6.2662), (7000, 6.0034), (7500, 6.1824), (8000, 5.8651), (8500, 5.9599), (9000, 6.0234), (9500, 6.3141), (10000, 6.0455), (10500, 5.7618), (11000, 5.4162), (11500, 5.7256), (12000, 4.8577))
#let sweep3-ntp = ((500, 2.3205), (1000, 2.8391), (1500, 2.9100), (2000, 2.8403), (2500, 2.7047), (3000, 2.5818), (3500, 2.8400), (4000, 2.4222), (4500, 3.0682), (5000, 2.8013), (5500, 2.8994), (6000, 2.7588), (6500, 2.7851), (7000, 2.4923), (7500, 2.7664), (8000, 2.6107), (8500, 1.7847), (9000, 1.8620), (9500, 2.0529), (10000, 1.9573), (10500, 2.2697), (11000, 2.3319), (11500, 2.3566), (12000, 1.6205))
#let sweep3-diff = ((500, 7.7564), (1000, 6.1781), (1500, 5.8160), (2000, 5.4674), (2500, 5.3106), (3000, 5.2327), (3500, 5.5303), (4000, 5.0315), (4500, 5.5508), (5000, 5.2645), (5500, 5.1873), (6000, 5.1729), (6500, 5.2286), (7000, 4.9426), (7500, 5.1486), (8000, 4.8791), (8500, 4.7080), (9000, 4.7853), (9500, 5.0504), (10000, 4.7909), (10500, 4.7078), (11000, 4.4758), (11500, 4.7104), (12000, 3.8279))
#let sweep3-acc = ((500, 0.1380), (1000, 0.2780), (1500, 0.3070), (2000, 0.2880), (2500, 0.3250), (3000, 0.3020), (3500, 0.3160), (4000, 0.3140), (4500, 0.3100), (5000, 0.3360), (5500, 0.3330), (6000, 0.3120), (6500, 0.3160), (7000, 0.2970), (7500, 0.3000), (8000, 0.3340), (8500, 0.2730), (9000, 0.2650), (9500, 0.2700), (10000, 0.2690), (10500, 0.3210), (11000, 0.3390), (11500, 0.3270), (12000, 0.2720))

#let sweep4-loss = ((125, 13.7980), (250, 10.8582), (375, 7.0645), (500, 6.2412), (625, 5.6279), (750, 5.3656), (875, 5.5364), (1000, 5.2680), (1125, 5.2849), (1250, 5.2516), (1375, 5.2813), (1500, 5.0336), (1625, 5.1637), (1750, 5.1199), (1875, 5.2155), (2000, 5.0151), (2125, 4.8394), (2250, 4.7868), (2375, 4.7444), (2500, 4.7736), (2625, 4.4644), (2750, 4.6415), (2875, 4.5355), (3000, 3.9294))
#let sweep4-ntp = ((125, 2.6500), (250, 2.8544), (375, 2.7380), (500, 2.8343), (625, 2.6238), (750, 2.5836), (875, 2.8946), (1000, 2.6135), (1125, 2.7064), (1250, 2.7484), (1375, 2.7492), (1500, 2.5734), (1625, 2.7014), (1750, 2.6384), (1875, 2.7464), (2000, 2.6373), (2125, 1.8429), (2250, 1.7660), (2375, 1.8258), (2500, 1.8452), (2625, 2.2095), (2750, 2.3002), (2875, 2.3120), (3000, 1.3364))
#let sweep4-diff = ((125, 15.5048), (250, 12.0599), (375, 7.6726), (500, 6.6817), (625, 5.9947), (750, 5.6723), (875, 5.7975), (1000, 5.5024), (1125, 5.4819), (1250, 5.4275), (1375, 5.4588), (1500, 5.1954), (1625, 5.3288), (1750, 5.2838), (1875, 5.3797), (2000, 5.1652), (2125, 5.0180), (2250, 4.9770), (2375, 4.9086), (2500, 4.9378), (2625, 4.6146), (2750, 4.7911), (2875, 4.6710), (3000, 4.0648))
#let sweep4-acc = ((125, 0.0180), (250, 0.0340), (375, 0.1800), (500, 0.1910), (625, 0.2530), (750, 0.2410), (875, 0.2740), (1000, 0.3010), (1125, 0.2920), (1250, 0.3220), (1375, 0.3210), (1500, 0.3100), (1625, 0.3140), (1750, 0.2920), (1875, 0.3050), (2000, 0.3390), (2125, 0.2710), (2250, 0.2620), (2375, 0.2750), (2500, 0.2700), (2625, 0.3290), (2750, 0.3270), (2875, 0.3300), (3000, 0.2710))

#let sweep5-loss = ((500, 4.9435), (1000, 4.4115), (1500, 4.1325), (2000, 3.9033), (2500, 3.9391), (3000, 3.8798), (3500, 4.0319), (4000, 4.0403), (4500, 4.1278), (5000, 3.6843), (5500, 3.6571), (6000, 3.6410), (6500, 3.2304), (7000, 3.2042))
#let sweep5-ntp = ((500, 2.6744), (1000, 2.7333), (1500, 2.7530), (2000, 2.5921), (2500, 2.6510), (3000, 2.4947), (3500, 2.8360), (4000, 2.8672), (4500, 2.8422), (5000, 2.5903), (5500, 2.5163), (6000, 2.4677), (6500, 1.8466), (7000, 1.8458))
#let sweep5-diff = ((500, 4.6202), (1000, 4.1220), (1500, 3.8980), (2000, 3.6746), (2500, 3.7260), (3000, 3.6251), (3500, 3.8600), (4000, 3.8635), (4500, 3.9294), (5000, 3.5178), (5500, 3.4660), (6000, 3.4144), (6500, 2.8889), (7000, 2.8895))
#let sweep5-acc = ((500, 0.2190), (1000, 0.5530), (1500, 0.5750), (2000, 0.6070), (2500, 0.4250), (3000, 0.4500), (3500, 0.6300), (4000, 0.6190), (4500, 0.6120), (5000, 0.6430), (5500, 0.6400), (6000, 0.6310), (6500, 0.6230), (7000, 0.7650))

#let sweep1-series = (
  (label: "Total", color: color-total, data: sweep1-loss),
  (label: "NTP", color: color-ntp, data: sweep1-ntp),
  (label: "Diff", color: color-diff, data: sweep1-diff),
  (label: "Acc", color: color-acc, data: sweep1-acc),
)
#let sweep2-series = (
  (label: "Total", color: color-total, data: sweep2-loss),
  (label: "NTP", color: color-ntp, data: sweep2-ntp),
  (label: "Diff", color: color-diff, data: sweep2-diff),
  (label: "Acc", color: color-acc, data: sweep2-acc),
)
#let sweep3-series = (
  (label: "Total", color: color-total, data: sweep3-loss),
  (label: "NTP", color: color-ntp, data: sweep3-ntp),
  (label: "Diff", color: color-diff, data: sweep3-diff),
  (label: "Acc", color: color-acc, data: sweep3-acc),
)
#let sweep4-series = (
  (label: "Total", color: color-total, data: sweep4-loss),
  (label: "NTP", color: color-ntp, data: sweep4-ntp),
  (label: "Diff", color: color-diff, data: sweep4-diff),
  (label: "Acc", color: color-acc, data: sweep4-acc),
)
#let sweep5-series = (
  (label: "Total", color: color-total, data: sweep5-loss),
  (label: "NTP", color: color-ntp, data: sweep5-ntp),
  (label: "Diff", color: color-diff, data: sweep5-diff),
  (label: "Acc", color: color-acc, data: sweep5-acc),
)

= TiDAR Sweep 4-5 - smollm-135m loss trends

== Setup

All four runs use smollm-135m (draft_length=5, context_length=2048). The loss combines AR and Diff terms plus an agreement penalty when lambda > 0:

$L_(text("total")) = frac(alpha * L_(text("AR")) + L_(text("Diff")), 1 + alpha) + lambda * text("KL")(p_(text("AR")) || p_(text("Diff")))$

=== Sweep summaries
- Sweep 1: lr=1e-4, alpha=1.0, lambda=0.0, warmup=2,000, batch=8
- Sweep 2: lr=5e-5, alpha=0.7, lambda=0.1, warmup=1,500, batch=8
- Sweep 3: lr=3e-5, alpha=0.3, lambda=0.5, warmup=1,200, batch=8 (aggressive agreement)
- Sweep 4: lr=1e-5, alpha=0.5, lambda=0.2, warmup=800, batch=32 (4x batch)
- Sweep 5: lr=3e-5, alpha=0.2, lambda=0.8, KL temp=2.0, draft_len=2, ctx-mix (<=300M tokens)

== Metric legend
#metric-legend

== Sweep 1 - Balanced AR/Diff, no agreement penalty
This run keeps alpha=1.0 and lambda=0, so the total loss is the straight average of AR and Diff losses.

$L_(text("total")) = frac(1.0 * L_(text("AR")) + L_(text("Diff")), 2.0) + 0.0 * text("KL")(p_(text("AR")) || p_(text("Diff")))$

#multi-line-chart(sweep1-series, border: rgb(200, 80, 80))

== Sweep 2 - Gentle agreement regularization
Lower alpha with a small lambda nudges Diff toward AR while preserving a steady training curve.

$L_(text("total")) = frac(0.7 * L_(text("AR")) + L_(text("Diff")), 1.7) + 0.1 * text("KL")(p_(text("AR")) || p_(text("Diff")))$

#multi-line-chart(sweep2-series, border: rgb(90, 165, 255))

== Sweep 3 - Aggressive agreement emphasis
This run applies the heaviest lambda penalty, which keeps the total loss higher throughout.

$L_(text("total")) = frac(0.3 * L_(text("AR")) + L_(text("Diff")), 1.3) + 0.5 * text("KL")(p_(text("AR")) || p_(text("Diff")))$

#multi-line-chart(sweep3-series, border: rgb(255, 170, 70))

== Sweep 4 - Large-batch stabilization
Batch size is 4x larger with moderate alpha/lambda, giving a smoother loss trajectory.

$L_(text("total")) = frac(0.5 * L_(text("AR")) + L_(text("Diff")), 1.5) + 0.2 * text("KL")(p_(text("AR")) || p_(text("Diff")))$

#multi-line-chart(sweep4-series, border: rgb(120, 210, 120))

== Sweep 5 - Aggressive diffusion with KL temperature
Mixed-context dataset with higher lambda and temperature-scaled KL. Draft length reduced to 2.

$L_(text("total")) = frac(0.2 * L_(text("AR")) + L_(text("Diff")), 1.2) + 0.8 * text("KL")(p_(text("AR")) || p_(text("Diff")))$

#multi-line-chart(sweep5-series, border: rgb(140, 160, 255))

== Greedy inference acceptance (local checkpoints)
Greedy decoding, temperature=0, top_k=0, stop_on_eos=false, prompt: "In a distant future, a curious robot loved math and".
Approx acceptance probability: $(text("avg_accept/iter") - 1) / (K - 1)$.

#table(
  columns: (1.1fr, 1fr, 0.7fr, 1fr, 1fr),
  [Sweep], [Checkpoint], [K], [Avg accept/iter], [Approx acc prob],
  [Sweep 1], [step_0012298], [5], [4.85], [0.96],
  [Sweep 2], [step_0012200], [5], [4.85], [0.96],
  [Sweep 3], [step_0012200], [5], [4.85], [0.96],
  [Sweep 4], [step_0003000], [5], [1.19], [0.05],
  [Sweep 5], [step_0013903], [2], [1.21], [0.21],
)

Note: Sweeps 1-3 greedy outputs were largely empty/special tokens in this prompt, even though acceptance was high.

== Takeaways
- Sweep 1 reaches the lowest total loss, but acceptance stays moderate; it is a strong baseline for quality.
- Sweep 2 balances loss and acceptance with a mild lambda, making it a likely candidate for higher acceptance without heavy degradation.
- Sweep 3 emphasizes agreement too aggressively, producing higher total loss and lower acceptance gains.
- Sweep 4 shows the most stable curve under large batches, so it is a good throughput-focused option.
- Sweep 5 shows the highest training-time acceptance on the mixed-context run, but greedy inference acceptance still trails the best-case runs.
