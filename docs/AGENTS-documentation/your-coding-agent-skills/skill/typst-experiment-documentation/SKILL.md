---
name: typst-experiment-documentation
description: Create detailed documentation in TYPST covering the architecture and experiments conduceted in the current subproject. 
---

# Paper Documentation

When you are asked by the user to create experiment docs at the end of the session, you should create a detailed TYPST document covering the architecture of the model and how it works, what experiments were conduceted in detail with charts and final findings and insights. You should compile it always to check for errors and use TYPST native features.

## Rules:
1. The file you output at the end should compile without errors.
2. Do not use images unless absolutely necessary (eg the user points to alwready created pngs to incorporate). Prefer TYPST native charts and code blocks.

## Examples:
1. Example of rendering colored python code block:
<typst>
</t#set page(width: 210mm, height: 297mm, margin: 18mm)
#set text(size: 11pt)

== Rendering with the codelst package:

#import "@preview/codelst:2.0.2": sourcecode

#box(width: 50%, inset: 0pt)[
#sourcecode[
```py
from dataclasses import dataclass

@dataclass
class Point:
x: float
y: float

p = Point(1.0, 2.0)
print(p)
```
]]
</typst>

2. Library for creating diagrams with arrows between them:
https://typst.app/universe/package/fletcher
Usage by `#import "@preview/fletcher:0.5.8"`

Example diagrams with it
<typst>
#diagram(cell-size: 15mm, $
	G edge(f, ->) edge("d", pi, ->>) & im(f) \
	G slash ker(f) edge("ur", tilde(f), "hook-->")
$)
</typst>

<typst>
#set text(10pt)
#diagram(
	node-stroke: .1em,
	node-fill: gradient.radial(blue.lighten(80%), blue, center: (30%, 20%), radius: 80%),
	spacing: 4em,
	edge((-1,0), "r", "-|>", `open(path)`, label-pos: 0, label-side: center),
	node((0,0), `reading`, radius: 2em),
	edge(`read()`, "-|>"),
	node((1,0), `eof`, radius: 2em),
	edge(`close()`, "-|>"),
	node((2,0), `closed`, radius: 2em, extrude: (-2.5, 0)),
	edge((0,0), (0,0), `read()`, "--|>", bend: 130deg),
	edge((0,0), (2,0), `close()`, "-|>", bend: -40deg),
)
</typst>

<typst>
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import fletcher.shapes: house, hexagon
#set page(width: auto, height: auto, margin: 5mm, fill: white)
#set text(font: "New Computer Modern")

#let blob(pos, label, tint: white, ..args) = node(
	pos, align(center, label),
	width: 28mm,
	fill: tint.lighten(60%),
	stroke: 1pt + tint.darken(20%),
	corner-radius: 5pt,
	..args,
)

#diagram(
	spacing: 8pt,
	cell-size: (8mm, 10mm),
	edge-stroke: 1pt,
	edge-corner-radius: 5pt,
	mark-scale: 70%,

	blob((0,1), [Add & Norm], tint: yellow, shape: hexagon),
	edge(),
	blob((0,2), [Multi-Head\ Attention], tint: orange),
	blob((0,4), [Input], shape: house.with(angle: 30deg),
		width: auto, tint: red),

	for x in (-.3, -.1, +.1, +.3) {
		edge((0,2.8), (x,2.8), (x,2), "-|>")
	},
	edge((0,2.8), (0,4)),

	edge((0,3), "l,uu,r", "--|>"),
	edge((0,1), (0, 0.35), "r", (1,3), "r,u", "-|>"),
	edge((1,2), "d,rr,uu,l", "--|>"),

	blob((2,0), [Softmax], tint: green),
	edge("<|-"),
	blob((2,1), [Add & Norm], tint: yellow, shape: hexagon),
	edge(),
	blob((2,2), [Feed\ Forward], tint: blue),
)
</typst>


**2. Example of documentation alwready created at:**
- TiDAR/Docs/Bucket_Prefix0_1600_A40_Spot.typ
- TiDAR/Docs/Finding_Free_token_slots_A40.typ
- TiDAR/Docs/TinyStories_TiDAR_losses_Comparison.typ (again really long and really good charts and histograms)
- TRM/TRM-token-tool/EXPERIMENTS_TRM_TOKEN_TOOL.typ
- TiDAR/TiDAR_summary.typ
- GIANT/v3/docs/Long_Natural_Record_Transition.typ
- GIANT/v3/docs/Long_LM_vs_ANS_Schedule_Sweep.typ
- GIANT/v3/docs/Long_Grokking_Probe_10k.typ
- GIANT/v3/docs/Long_Grokking_Probe_100k.typ
- TiDAR/docs/Results_Greedy_runs_135_360.typ (really good example of charts and logs documented)
- GIANT/v3/docs/LongDSL_Night1_Bootstrap_Report.typ

## Workflow
1. Read the A-Logs.md file to see the results from ran commands or other log commands from recent experiments.
2. **Read some of the examples above to see how to create TYPST documentation.**
3. Start creating documentation part by part and on every added thing try to compile with `typst compile <file>.typ`. If there are errors, fix them. (point the resulting compiled pdf to next to the .typ file)
4. When done just say "Documentation created at <file>.typ." and then super shortly what it touched on + add the path to the created documentation file as an example in the matching skill file inside `<PROJECT_ROOT>/your-coding-agent-skills/skill/`.
