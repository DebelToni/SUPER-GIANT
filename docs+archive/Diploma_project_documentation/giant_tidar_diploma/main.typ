#set page(width: 210mm, height: 297mm, margin: 20mm)
#set page(footer: context align(center)[#text(size: 9pt)[#counter(page).display()]])
#set text(font: "Times New Roman", size: 11pt)
#set par(justify: true, leading: 0.6em)
#set heading(numbering: none)

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

#include "sections/00-cover.typ"

#pagebreak()
#include "sections/01-intro.typ"

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

#pagebreak()
#include "sections/09-experiments-embedded.typ"
