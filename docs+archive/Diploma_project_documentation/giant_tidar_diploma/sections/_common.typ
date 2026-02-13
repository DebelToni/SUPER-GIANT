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
