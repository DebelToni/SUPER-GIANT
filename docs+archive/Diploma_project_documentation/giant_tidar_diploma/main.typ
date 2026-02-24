#set page(width: 210mm, height: 297mm, margin: 20mm)
#set page(footer: context align(center)[#text(size: 9pt)[#counter(page).display()]])
#set text(font: "Times New Roman", size: 12pt)
#set par(justify: true, leading: 1.3em)
#set heading(numbering: none)
#show heading.where(level: 1): it => [
  #align(center, it)
  #v(0.8em)
]
#show heading.where(level: 2): it => [
  #align(center, it)
  #v(1.6em)
]
#show figure.caption: it => context {
  let n = it.counter.display(it.numbering)
  text(fill: rgb("#374151"))[#it.supplement #h(2pt)#n#it.separator #it.body]
}

#let placeholder_figure(title, body: none) = figure(
  block(
    width: 100%,
    inset: 10pt,
    radius: 6pt,
    stroke: 0.8pt + rgb("8a8f98"),
    fill: luma(245),
    [
      *#title*
      #if body != none [
        #v(6pt)
        #body
      ]
    ],
  ),
  caption: [#title],
)

#align(center + horizon)[
  #text(size: 24pt, weight: "bold")[СКАНИРАНО ЗАДАНИЕ]
]

#pagebreak()
#align(center + horizon)[
  #text(size: 24pt, weight: "bold")[СКАНИРАНО СТАНОВИЩЕ]
]

#pagebreak()
#include "sections/02-ch1.typ"

#pagebreak()
#include "sections/03-ch2.typ"

#pagebreak()
#include "sections/03b-transformer-model.typ"
#include "sections/03c-speculative-decoding.typ"

#pagebreak()
#include "sections/04-ch3.typ"

#pagebreak()
#include "sections/05-ch4.typ"

#pagebreak()
#include "sections/06-conclusion.typ"

#pagebreak()
#include "sections/07-bibliography.typ"

#pagebreak()
#include "sections/08-appendix-resources.typ"
