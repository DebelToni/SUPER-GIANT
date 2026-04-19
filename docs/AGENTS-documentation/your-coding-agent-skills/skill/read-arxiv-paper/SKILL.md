---
name: read-arxiv-paper
description: Read an arXiv paper from TeX source and write a repo-focused summary. Use when the user references a arXiv url for a paper.
---

# Read arXiv paper

Use this skill when the user provides an arXiv URL/ID and wants a technical summary.

## Core rules

* Always read the **source TeX** (`/src/`), not the PDF.
* Use the helper script configured for this checkout. The bundled starter-pack copy lives at `your-coding-agent-skills/bin/arxiv-src` until the user chooses a final location.
* Write summaries to the `docs/paper_summaries/summary_<arxiv-tag>_<short-name>/summary.md`. The name should be something like short title or the technology presented in the paper. In the created folder you will also put any refernece code that you get from the paper's repo or similiar place. 
* Keep the summary focused on how the paper applies to this repo (GIANT/TiDAR/JAX/inference/training).

## Workflow

1. Fetch + unpack source:

```bash
your-coding-agent-skills/bin/arxiv-src "<arxiv-url-or-id>"
```

Use the printed fields (`EXTRACT_DIR`, `ENTRYPOINT`).

2. Read the entrypoint `.tex` and recursively follow relevant `\input{...}` / `\include{...}` files.

3. If needed, read related local docs/code for context before writing conclusions.

4. Create summary markdown in `docs/paper_summaries/` with this structure:
   - Problem and core idea
   - Method details (short)
   - Key results
   - What is relevant for SUPER-GIANT
   - Concrete experiments to run next (3-6 bullets)
   - Risks / open questions

--- 

Conditional/Optional last step:
5. Look in the content of the .tex if there is a direct link to github repo or use a link the user has provided. Clone it to /tmp/ and read its readme and source code. Directly copy or pseudo trnaslate the importatn implementation code to the folder where the summary.md is and update the summary.md at the end stating about the code and some additional notes around it. 

Example will be if the user asked for the MLA paper by deepseek to plug the raw pytorch code of the exact MLA mechanism into the folder, not the hole DeepSeek LLM code.

## Notes

* Do not overwrite an existing summary unless explicitly requested.
* If the user relocates the helper script, update this file to point at the chosen final path.
