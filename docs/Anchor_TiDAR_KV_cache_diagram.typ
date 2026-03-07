#import "@preview/fletcher:0.5.8": diagram, node, edge

#set page(width: 188mm, height: 128mm, margin: 4mm)
#set text(font: "Times New Roman", size: 10pt)

#let tok(lbl, fill: rgb("#E5E7EB"), stroke_color: rgb("#374151")) = box(
  width: 22pt,
  height: 12pt,
  inset: (x: 4pt, y: 2pt),
  stroke: 0.5pt + stroke_color,
  fill: fill,
  radius: 2pt,
)[#align(center)[#text(size: 7.5pt)[#lbl]]]

#grid(
  columns: (43%, 57%),
  gutter: 8pt,
  align: top,
  [
    #align(left)[
      #text(size: 9pt, weight: "bold")[A) One Anchor-TiDAR decode iteration]
      #v(3pt)
      #text(size: 7.8pt)[
        The cycle starts from a committed prefix length `L`. One single forward pass computes
        both verify logits and all predraft branches. After acceptance, only the valid prefix is
        committed, and the selected branch seeds the next iteration.
      ]
      #v(3pt)
      #diagram(
        cell-size: (16mm, 9mm),
        spacing: 6pt,
        node-stroke: 0.8pt + rgb("#4B5563"),
        edge-stroke: 0.8pt,
        mark-scale: 70%,

        node((0, 0), [Prefill\nL = prompt len], width: 36mm, fill: rgb("#E5E7EB")),
        node((0, 1), [Single forward\nverify + K predraft], width: 44mm, fill: rgb("#DBEAFE")),
        node((0, 2), [Acceptance check\nanchor + verify], width: 40mm, fill: rgb("#FDE68A")),
        node((0, 3), [Accepted prefix = r\n(1 <= r <= K)], width: 40mm, fill: rgb("#FECACA")),
        node((0, 4), [Pointer commit\nL := L + r], width: 34mm, fill: rgb("#DCFCE7")),
        node((0, 5), [Next draft from\nselected proposal], width: 40mm, fill: rgb("#E9D5FF")),

        edge((0, 0), (0, 1), "-|>"),
        edge((0, 1), (0, 2), "-|>"),
        edge((0, 2), (0, 3), "-|>"),
        edge((0, 3), (0, 4), "-|>"),
        edge((0, 4), (0, 5), "-|>"),
      )
    ]
  ],
  [
    #align(left)[
      #text(size: 9pt, weight: "bold")[B) KV cache pointer semantics]
      #v(2pt)

      #text(size: 7.9pt)[
        Initial state:\n`A B C D* E F` are already committed in KV cache.\nThe write pointer is at `L=6`.
      ]
      #v(2pt)
      #align(left)[#tok("A") #h(2pt) #tok("B") #h(2pt) #tok("C") #h(2pt) #tok("D*", fill: rgb("#FDE68A"), stroke_color: rgb("#DC2626")) #h(2pt) #tok("E", fill: rgb("#BFDBFE")) #h(2pt) #tok("F", fill: rgb("#BFDBFE"))]

      #v(3pt)
      #text(size: 7.9pt)[
        During decode, the pass writes temporary states `G' H I` after position `L`.\nThis advances the temporary pointer to `L+K=9`.
      ]
      #v(2pt)
      #align(left)[#tok("A") #h(2pt) #tok("B") #h(2pt) #tok("C") #h(2pt) #tok("D*", fill: rgb("#FDE68A"), stroke_color: rgb("#DC2626")) #h(2pt) #tok("E", fill: rgb("#BFDBFE")) #h(2pt) #tok("F", fill: rgb("#BFDBFE")) #h(2pt) #tok("G'", fill: rgb("#C4B5FD")) #h(2pt) #tok("H", fill: rgb("#C4B5FD")) #h(2pt) #tok("I", fill: rgb("#FCA5A5"), stroke_color: rgb("#B91C1C"))]

      #v(3pt)
      #text(size: 7.9pt)[
        If verification fails at the last drafted token (`I' != I`), only `G' H` remain valid.\nThe invalid suffix (`I`) is removed logically via pointer rollback.
      ]
      #v(2pt)
      #align(left)[#tok("A") #h(2pt) #tok("B") #h(2pt) #tok("C") #h(2pt) #tok("D*", fill: rgb("#FDE68A"), stroke_color: rgb("#DC2626")) #h(2pt) #tok("E", fill: rgb("#BFDBFE")) #h(2pt) #tok("F", fill: rgb("#BFDBFE")) #h(2pt) #tok("G'", fill: rgb("#C4B5FD"), stroke_color: rgb("#166534")) #h(2pt) #tok("H", fill: rgb("#C4B5FD"), stroke_color: rgb("#166534")) #h(2pt) #tok("I", fill: rgb("#F3F4F6"), stroke_color: rgb("#9CA3AF"))]
      #v(2pt)
      #text(size: 7.9pt)[
        Final state for this cycle:\n`L := 8`, committed cache is `A B C D* E F G' H`.\nThe next cycle starts from this exact committed boundary.
      ]
    ]
  ],
)
