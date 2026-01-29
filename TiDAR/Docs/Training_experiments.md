# TiDAR Training Experiments

Purpose
- This file summarizes past TiDAR experiments for future agents.
- The .typ files are for human-readable reports; this file is a concise run log with context and takeaways.

Primary sources
- `TiDAR/Docs/Results_Sweep_4_runs_smollm135.typ`
- `TiDAR/Docs/TiDAR_inference_experiments_summary.typ`

Sweep summary (smollm-135m)
- Sweep 1: lr=1e-4, alpha=1.0, lambda=0.0, warmup=2000, batch=8, draft_len=5, ctx=2048.
  - Training acc ~0.33-0.39, lower total loss baseline.
- Sweep 2: lr=5e-5, alpha=0.7, lambda=0.1, warmup=1500, batch=8, draft_len=5.
  - Similar acceptance to sweep 1, slightly higher loss.
- Sweep 3: lr=3e-5, alpha=0.3, lambda=0.5, warmup=1200, batch=8, draft_len=5.
  - More aggressive KL; loss higher, acc not improved.
- Sweep 4: lr=1e-5, alpha=0.5, lambda=0.2, warmup=800, batch=32, draft_len=5.
  - More stable loss; acc roughly 0.27-0.33 in the logged window.
- Sweep 5: lr=3e-5, alpha=0.2, lambda=0.8, KL temp=2.0, draft_len=2, ctx-mix dataset (<=300M tokens).
  - Training acc rose quickly to ~0.62-0.77 in the first 7k steps.
  - Greedy inference on step_0013903 reported avg_accept/iter ~1.29 (K=2), implying acc_prob ~0.29.

Inference throughput summary (.typ)
- `TiDAR_inference_experiments_summary.typ` includes iter/s measurements across
  contexts (512..8192) and draft lengths (4..32) for several kernel variants.
- These results are throughput-only and do not reflect acceptance/quality.

Important interpretation notes
- Training-time `acc` is a theoretical acceptance probability in [0, 1] derived
  from AR/Diff logits (top-k truncated, last batch row only).
- Greedy inference acceptance is measured from actual accept/reject events:
  `acc_prob ~= (avg_accept/iter - 1) / (K - 1)`.
- Training acc and inference acc can diverge (especially with top-k truncation).

Implementation updates that affect comparisons
- Agreement losses are aligned by +1 position (AR t vs Diff t+1).
- Training loss/metrics now live in `TiDAR/model/Training_step.py` and are
  called by `TiDAR/model/Run_training.py`.
- **Loss formula update (Jan 2026)**: The loss function was refactored from a 2-term
  formula with normalization to a 5-term configurable formula:
  `Loss = alpha * L_AR + beta * L_Diff + rho * KL_fwd + chi * KL_rev + delta * L_hard`
  See `TiDAR/Docs/TiDAR_docs.md` Section 1.2 for full documentation.

Open questions for next experiments
- Should acceptance logging average across multiple batch rows to reduce noise?
- Do we need a warmup schedule for lambda or temperature to improve inference acc?
- Should training acc be computed without top-k truncation for closer inference tracking?
