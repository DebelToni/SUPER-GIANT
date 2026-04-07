#set page(width: 210mm, height: 297mm, margin: 18mm)
#set text(size: 11pt)
#set heading(numbering: none)

#let ink = rgb(28, 34, 40)
#let soft = rgb(104, 114, 126)
#let line = rgb(214, 218, 224)
#let blue = rgb(39, 102, 180)
#let green = rgb(38, 153, 93)
#let orange = rgb(214, 131, 46)
#let red = rgb(203, 67, 53)

#let card(title, value, note: none, tint: blue) = box(
  width: 100%,
  inset: 8pt,
  radius: 6pt,
  stroke: 0.7pt + line,
  fill: tint.lighten(88%),
)[
  #text(size: 9pt, fill: soft)[#title]
  #v(2pt)
  #text(size: 15pt, weight: "bold", fill: ink)[#value]
  #if note != none [
    #v(2pt)
    #text(size: 8pt, fill: soft)[#note]
  ]
]

#let pct-bar(frac, color, width: 62mm) = {
  let inner = frac * (width - 2pt)
  box(width: width, inset: 1pt, radius: 3pt, stroke: 0.6pt + line, fill: rgb(248, 249, 251))[
    #box(width: inner, height: 8pt, radius: 2pt, fill: color)[]
  ]
}

= Long Natural Record Transition

#text(fill: soft)[GIANT v3 Long folder migration from raw opcode DSL to hidden-world natural-language records.]

== Why the Long folder changed

The first LongGIANT attempt proved that a tiny opcode-style synthetic task was controllable, but it also made the core weakness obvious: the surface form looked too much like execution and not enough like language.

The new direction keeps the *latent* symbolic world but replaces the visible text with coherent record-style sentences. The benchmark now sits in the middle ground we wanted:

- exact latent semantics
- natural-looking evidence sentences
- short exact answers
- scalable filler for longer contexts
- clean debugging because the hidden world remains explicit

== Legacy raw DSL snapshot

The previous task looked like this on the surface:

```text
DEF e3 v7
ALIAS e9 e3
SET e3 v2
ASK e9
```

That form was useful for mechanism screening, but it was too procedural for the new goal. It encouraged execution-like pattern matching rather than language-like retrieval over records.

The previous results are preserved in `GIANT/v3/docs/LongDSL_Night1_Bootstrap_Report.typ` and the legacy training configs are archived under `GIANT/v3/Configs/Training/Long/legacy_rawdsl/`.

== Current minimal pipeline

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  card("Model", "38.64M params", note: "448 dim, 7 heads, 12 layers, FF 1792", tint: blue),
  card("Surface Genre", "admin_record", note: "deterministic templates only", tint: orange),
  card("Current Run", "27 / 128", note: "21.09% exact match at level 1 / ctx 128", tint: green),
)

#v(6pt)

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  card("Latent ops", "BIND / ALIAS / LINK / SET", note: "level 1 uses BIND+ALIAS+LINK, level 2 adds SET", tint: blue),
  card("Artifacts", "/proj/giant-data/GIANT/Long", note: "tokenizer + raw records + dataset config", tint: orange),
  card("GPU", "1x RTX A6000", note: "RunPod secure cloud training run", tint: green),
)

== Design in one paragraph

Each sample builds a hidden world of named entities and relations such as `locker`, `room`, `role`, or `status`. The model never sees the latent ops. Instead it sees record sentences such as assignments, alias notes, and later updates. The query names either a primary profile or an alias, and the answer is always the final one-token value associated with that name and relation.

== Concrete examples

=== Example A: level 1

Hidden world:

```text
BIND(Marek, person_001)
LINK(person_001, locker, 36)
ALIAS(Noel, Marek)
ASK(Noel, locker)
```

Rendered sample:

```text
Context: Luca was assigned locker 35. Toma served as coordinator. Marek was assigned locker 36. Records show Risa using locker 27. Noel appears as an alternate name for Marek.
Question: Which locker is currently associated with Noel?
Answer: 36
```

=== Example B: level 2

Hidden world:

```text
BIND(Selma, person_000)
LINK(person_000, role, analyst)
ALIAS(Nera, Selma)
SET(person_000, role, operator)
ASK(Nera, role)
```

Rendered sample:

```text
Context: Records show Selma in the role of analyst. Nera appears as an alternate name for Selma. Later notes reassigned Selma as operator. Records show the status for Elin as flagged. Records show Niko using locker 29.
Question: Which role is associated with Nera now?
Answer: operator
```

== Current file and data layout

Repo entrypoints:

- `GIANT/v3/Long/longdsl.py`
- `GIANT/v3/Long/prepare_longdsl.py`
- `GIANT/v3/Long/write_training_configs.py`
- `GIANT/v3/Long/eval_longdsl.py`

Local generated artifacts:

- tokenizer: `/proj/giant-data/GIANT/Long/tokenizers/long_wordlevel`
- raw JSONL: `/proj/giant-data/GIANT/Long/records/raw/level{N}_ctx{L}`
- dataset config: `/proj/giant-data/GIANT/Long/configs/long_records_datasets.yml`
- Arrow shards: `/proj/giant-data/GIANT/dataset_artifacts/long_records`

Saved training outputs from this session:

- local checkpoint copy: `/proj/giant-data/GIANT/single-gpu/checkpoints/long/records_bootstrap/40m/l1_ctx128/step_0003584.npz`
- local logs copy: `/proj/giant-data/GIANT/single-gpu/checkpoints/long/records_bootstrap/40m/l1_ctx128/logs.txt`
- S3 mirror: `s3://giant-data/GIANT/single-gpu/checkpoints/long/records_bootstrap/40m/l1_ctx128/`

== First training result

The first real run used `GIANT/v3/Configs/Training/Long/long_40m_l1_ctx128.yml` on the new natural-record dataset.

- training schedule: `1` LM epoch + `6` answer-only epochs
- total steps: `3584`
- final checkpoint: `step_0003584.npz`
- final reported train loss: `0.009`
- held-out exact match on `128` test samples: `27 / 128 = 21.09%`

Result bar:

#grid(
  columns: (36mm, 70mm, 30mm, 1fr),
  gutter: 8pt,
  align: (left, center, right, left),
  [#text(size: 9.5pt, weight: "semibold")[Level 1 at 128]],
  [#pct-bar(0.2109, green)],
  [#text(size: 9.5pt)[27 / 128]],
  [#text(size: 8.5pt, fill: soft)[meaningful signal, but still far from solved]],
)

=== What this means

- the new natural-language pipeline runs end to end
- the 40M-class model learns something real on held-out data
- the current deterministic template set is still not enough to make the task easy
- train loss collapses much faster than held-out exact match, so generalization is the real bottleneck now

== Cleanup and migration notes

The migration kept the old report but moved the active experiment surface to the natural-record pipeline. The practical cleanup is:

- new active config names use `long_40m_...`
- legacy raw DSL configs move under `GIANT/v3/Configs/Training/Long/legacy_rawdsl/`
- new dataset artifacts live in `dataset_artifacts/long_records` instead of `dataset_artifacts/longdsl`

This keeps the old work reproducible without mixing it into the active path.

== Recommended next steps

The next improvement should focus on data quality before bigger scale:

- reduce repetitive filler-note patterns
- create a template holdout split so eval is less tied to seen phrasing
- try a short-context bootstrap such as `64 -> 128` on the new record renderer
- only add offline paraphrase-bank expansion after the deterministic renderer feels stable

The migration itself is complete: the Long folder now screens language-like retrieval rather than opcode imitation.
