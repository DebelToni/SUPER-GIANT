# Attention Residuals (arXiv:2603.15031) - Repo-Focused Summary

## Problem and core idea
- Standard PreNorm residuals use fixed additive accumulation (`h = h + f(h)`), which can cause depth-wise contribution dilution as depth grows.
- Attention Residuals (AttnRes) replaces fixed residual mixing with depth-wise softmax routing over earlier representations.
- Each layer/sub-layer uses a learned pseudo-query vector to select useful earlier depth states.
- The paper frames this as a time-depth duality: replacing depth recurrence with attention, analogous to how transformers replaced temporal recurrence.

## Method details (short)
- Full AttnRes:
  - For each depth step `l`, attend over all prior depth sources (`v_0..v_{l-1}`), where keys are RMS-normalized.
  - Pseudo-query `w_l` is layer-specific and learnable.
  - Zero-initialize `w_l` so initial routing is uniform and stable.
- Block AttnRes:
  - Partition depth into `N` blocks, compress each completed block to a block summary, and attend over block summaries instead of all per-layer outputs.
  - Keeps a current intra-block partial sum and adds it as an extra source.
  - Cuts memory/communication pressure from `O(Ld)` style behavior to `O(Nd)` style behavior in their setup.
- Infra ideas in the paper:
  - Cross-stage cache to reduce repeated pipeline communication.
  - Two-phase inference with inter-block batched attention + intra-block sequential merge using online softmax statistics.

## Key results
- Scaling-law section reports consistent validation-loss improvements for Full and Block AttnRes over baseline residuals.
- Reported compute-equivalent gain for Block AttnRes is about `1.25x` at a representative scale point in their fit.
- Their larger model experiments report improvements on reasoning, code, and language tasks.
- They report low system overhead after optimization (small training overhead and very small inference overhead in their infra).

## What is relevant for SUPER-GIANT
- GIANT v3 currently uses standard PreNorm residual additions in `GIANT/v3/model/Transformer_block.py`.
- AttnRes is directly applicable to that path: replace fixed residual adds with learned depth-wise aggregation.
- GIANT v3 already has useful prerequisites:
  - RMSNorm in-place.
  - Config-driven toggles for model behavior.
  - Clean Flax/JAX modular block structure.
- Lowest-risk adoption order:
  - First implement a small `attnres_full` prototype (quality test only).
  - Then evaluate a block variant for memory/throughput tradeoff.
  - Later add two-phase decode optimization if quality gains justify engineering effort.

## Concrete experiments to run next
- Add `model.residual_mode: standard|attnres_full|attnres_block` and begin with `attnres_full` only.
- Keep pseudo-query zero-init and key RMSNorm-only behavior exactly as in paper.
- Run controlled equal-budget ablation on existing v3 dev training config:
  - baseline vs attnres_full
  - same data/token budget, same optimizer schedule.
- Track more than loss:
  - loss vs walltime
  - per-depth activation magnitude
  - per-depth gradient norm.
- If attnres_full helps, add `attnres_block` with fixed small block count and compare memory/throughput.

## Risks and open questions
- Paper formulation is per sub-layer (attention and MLP routes can differ); GIANT block wiring may need careful placement to preserve that behavior.
- Full-history depth sources can increase activation pressure and distributed complexity.
- GIANT v3 decode/KV path is already sensitive; defer complex depth-cache inference optimizations until KV path is robust.

## Extracted reference code
- Source repo used: `https://github.com/kyegomez/attn_res` (unofficial implementation).
- Important extracted files in this folder:
  - `docs/paper_summaries/summary_2603.15031_attention_residuals/reference_code/attn_res_core_torch.py`
  - `docs/paper_summaries/summary_2603.15031_attention_residuals/reference_code/block_attnres_forward_torch.py`
- Notes:
  - This reference is useful for mechanism understanding and quick prototyping.
  - It is not a production parity implementation of Moonshot's full distributed infrastructure.
