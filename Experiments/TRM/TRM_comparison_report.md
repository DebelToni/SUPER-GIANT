# TRM comparison report (JAX vs original)

## Scope
- JAX repo: `/Users/antonhristov/Documents/ML/SUPER-GIANT/TRM`
- Original repo: `/Volumes/SSD/TRM_original_repo`
- Focus: TRM architecture, training loop, and performance/optimization gaps.

## Major architectural differences
- ACT/halting loop vs fixed-step unroll: the original wraps the inner TRM in an ACT-style carry and dynamically halts per sample (`/Volumes/SSD/TRM_original_repo/models/recursive_reasoning/trm.py`). Your JAX version always runs a fixed number of `H_cycles` and `L_cycles` and then repeats this for a fixed `supervision_steps` count; there is no per-sample early stop even though config contains `enable_early_stop` (`/Users/antonhristov/Documents/ML/SUPER-GIANT/TRM/model/TRM.py`, `/Users/antonhristov/Documents/ML/SUPER-GIANT/TRM/model/Config.yml`).
- Q-head definition and usage: original uses a 2-logit Q head (halt vs continue) with special initialization and (optionally) a Q-continue bootstrapping loss (`/Volumes/SSD/TRM_original_repo/models/recursive_reasoning/trm.py`, `/Volumes/SSD/TRM_original_repo/models/losses.py`). Your JAX model uses a single logit pooled over the sequence and trains it every step with a binary halt loss (`/Users/antonhristov/Documents/ML/SUPER-GIANT/TRM/model/TRM.py`, `/Users/antonhristov/Documents/ML/SUPER-GIANT/TRM/sudoku/Training_step.py`).
- Embedding scaling and init: original uses truncated LeCun init and explicit scaling by `sqrt(hidden_size)` for token embeddings, and special zero/bias init for the Q head (`/Volumes/SSD/TRM_original_repo/models/common.py`, `/Volumes/SSD/TRM_original_repo/models/recursive_reasoning/trm.py`). Your JAX version uses a fixed `std=0.02` normal init and no embedding scale or Q-head bias init (`/Users/antonhristov/Documents/ML/SUPER-GIANT/TRM/model/TRM.py`).
- Positional encoding path: original uses either RoPE or learned positions (mutually exclusive). JAX version can apply RoPE in attention and also add a learned position embedding if `add_positional_embedding` is true, which deviates from the original design and adds extra compute (`/Volumes/SSD/TRM_original_repo/config/arch/trm.yaml`, `/Users/antonhristov/Documents/ML/SUPER-GIANT/TRM/model/TRM.py`).
- Norm order: original blocks are post-norm (residual then RMSNorm) for both attention and MLP (`/Volumes/SSD/TRM_original_repo/models/recursive_reasoning/trm.py`, `/Volumes/SSD/TRM_original_repo/models/layers.py`). Your `TinyTRMLayer` is pre-norm (RMSNorm before attention/MLP), which changes training dynamics (`/Users/antonhristov/Documents/ML/SUPER-GIANT/TRM/model/TRM_block.py`).
- Puzzle embeddings: original prepends learned puzzle-identifier embeddings and optimizes them with a sparse SignSGD step (`/Volumes/SSD/TRM_original_repo/models/sparse_embedding.py`, `/Volumes/SSD/TRM_original_repo/puzzle_dataset.py`). Your JAX code uses an optional additive augmentation embedding only; no separate sparse optimizer and no extra tokens (`/Users/antonhristov/Documents/ML/SUPER-GIANT/TRM/model/TRM.py`).
- Attention kernel: original uses `scaled_dot_product_attention` (FlashAttention path in PyTorch) with casted weights (`/Volumes/SSD/TRM_original_repo/models/layers.py`). Your JAX attention is a naive `einsum + softmax` implementation (no flash kernel) (`/Users/antonhristov/Documents/ML/SUPER-GIANT/TRM/model/TRM_block.py`).
- Loss function: original defaults to `stablemax_cross_entropy` (custom) for LM loss (`/Volumes/SSD/TRM_original_repo/models/losses.py`, `/Volumes/SSD/TRM_original_repo/config/arch/trm.yaml`). JAX uses standard softmax cross-entropy (`/Users/antonhristov/Documents/ML/SUPER-GIANT/TRM/sudoku/Training_step.py`, `/Users/antonhristov/Documents/ML/SUPER-GIANT/TRM/json_reformat/Training_step.py`).

## Missed optimization tricks and FLOP sinks
- Recomputing encodings per supervision step: inside each `train_step`, `model.encode` is called once per step, even though the input does not change (`/Users/antonhristov/Documents/ML/SUPER-GIANT/TRM/sudoku/Training_step.py`, `/Users/antonhristov/Documents/ML/SUPER-GIANT/TRM/json_reformat/Training_step.py`). This repeats embedding, positional, and dropout work K times.
- Multiple `value_and_grad` passes per step: you compute gradients separately for every supervision step and then average. This is K separate backward passes, which is a major FLOP/memory sink. Because you already stop-gradient between steps, the loss could be accumulated in a single forward scan and differentiated once.
- Fragmented `model.apply` calls: encode, initial state, and step are invoked as separate `model.apply` calls. This prevents whole-model fusion and adds call overhead in JAX. A single `model.apply` that returns `x`, `(y,z)` and final logits would compile more efficiently.
- No ACT/early-stop: all samples run full `H_cycles * L_cycles * supervision_steps`, even when solved. The original uses per-sample halting and stops updating halted samples, saving FLOPs (`/Volumes/SSD/TRM_original_repo/models/recursive_reasoning/trm.py`).
- Python loops for recursion: `step_from_x` uses Python loops over `H_cycles` and `L_cycles`. In JAX this leads to large unrolled graphs and long compile times. A `jax.lax.scan` or `fori_loop` would reduce compilation overhead and improve remat behavior (`/Users/antonhristov/Documents/ML/SUPER-GIANT/TRM/model/TRM.py`).
- Attention implementation: `einsum` attention is slower and memory-heavy compared to `scaled_dot_product_attention`/Flash. This is a direct compute sink (`/Users/antonhristov/Documents/ML/SUPER-GIANT/TRM/model/TRM_block.py`).
- Extra argmax and pooling work each step: `step_from_x` always computes `pred` and sequence-mean pooled `q_logit`. If metrics are not needed every step, these can be moved out of the hot path.

## Other notable gaps vs original
- No EMA option (`/Volumes/SSD/TRM_original_repo/models/ema.py` vs no EMA in JAX).
- No custom optimizer (AdamATan2 and sparse SignSGD are used in original) (`/Volumes/SSD/TRM_original_repo/pretrain.py`, `/Volumes/SSD/TRM_original_repo/models/sparse_embedding.py`).
- No ACT exploration (randomized halting) and no Q-continue target bootstrapping logic (`/Volumes/SSD/TRM_original_repo/models/recursive_reasoning/trm.py`).
- Different FFN shape strategy: original uses SwiGLU with hidden size rounded to a multiple of 256; JAX uses a fixed `d_ff` and splits it in half, which may be less kernel-efficient (`/Volumes/SSD/TRM_original_repo/models/layers.py`, `/Users/antonhristov/Documents/ML/SUPER-GIANT/TRM/model/TRM_block.py`).

## High-impact fixes (priority order)
1. Implement ACT-style halting or at least early-stop on `q_logit` so solved samples stop consuming compute.
2. Compute `x = encode(...)` once per batch and reuse across supervision steps.
3. Replace per-step `value_and_grad` with a single scan + one backward pass; keep stop-grad semantics inside the scan.
4. Switch attention to a fused kernel (JAX `dot_product_attention` or FlashAttention-equivalent).
5. Align init/scaling with original (sqrt(hidden) scaling, truncated normal, Q-head bias) and consider stablemax loss if you want closer behavior.

