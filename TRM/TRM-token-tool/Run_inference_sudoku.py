from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import core as flax_core
from jax import config as jax_config
from omegaconf import OmegaConf

from config_utils import load_config
from tokenizer_utils import build_custom_tokenizer, load_tokenizer


from GIANT.v2.smol.GiantGPT import GiantGPT
from GIANT.v2.smol.checkpoint_io import load_npz
from GIANT.v2.smol.jit_inference import init_inference_state, make_prefill_and_decode_fns


TRM_OPEN = "<TRM-sudoku>"
TRM_CLOSE = "</TRM-sudoku>"
TRM_ROOT = Path(__file__).resolve().parent.parent
TRM_DEFAULT_CHECKPOINT_DIR = "checkpoints/trm_sudoku"


@dataclass
class TrmSudokuSolver:
    model: Any
    params: Any
    infer_fn: Any
    checkpoint_path: Path
    step: int

    def solve(self, puzzle: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        puzzle = np.asarray(puzzle, dtype=np.int32).reshape(-1)

        if puzzle.size != 81:
            raise ValueError("Puzzle must be 81 entries for TRM solver.")
        tokens = jnp.asarray(puzzle[None, :], dtype=jnp.int32)
        aug_ids = jnp.zeros((1,), dtype=jnp.int32)
        _logits, q_logit, pred, _state, steps, halted = self.infer_fn(
            self.params, tokens, aug_ids
        )
        pred_np = np.array(pred[0], dtype=np.int32)
        meta = {
            "q_logit": float(np.array(q_logit[0])),
            "steps": int(np.array(steps)),
            "halted": bool(np.array(halted[0])),
        }
        return pred_np, meta




def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("TRM Sudoku inference (LLM-only).")
    ap.add_argument("--checkpoint", default=None, help="Path to .npz checkpoint (default: latest in cfg).")
    ap.add_argument("--puzzle", default=None, help="81 digits (spaces optional) with 0 for blanks.")
    ap.add_argument("--user_prompt", default=None, help="Override user prompt text. Use {puzzle} placeholder.")
    ap.add_argument("--user_prompt_file", default=None, help="Path to a text file with user prompt.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_trm_solver", action="store_true", help="Solve with TRM sudoku model instead of Python solver.")
    ap.add_argument("--trm_checkpoint", default=None, help="Path to TRM sudoku checkpoint (.npz).")
    ap.add_argument(
        "--trm_checkpoint_dir",
        default=TRM_DEFAULT_CHECKPOINT_DIR,
        help="TRM checkpoint dir (relative to TRM data root).",
    )
    ap.add_argument(
        "--trm_config",
        default=None,
        help="Path to TRM sudoku config (default: TRM/sudoku/Config.yml).",
    )
    ap.add_argument(
        "--trm_data_root",
        default=None,
        help="Override TRM data root for checkpoints/cache.",
    )
    return ap.parse_args()



def _to_dtype(name: str) -> jnp.dtype:
    try:
        return getattr(jnp, name)
    except AttributeError:
        return jnp.dtype(name)



def _apply_compute_dtype_override(cfg) -> None:
    desired = None
    if "model" in cfg and "compute_dtype" in cfg.model:
        desired = cfg.model.compute_dtype
    if not desired:
        return
    dtype = _to_dtype(str(desired))
    try:
        import GIANT.v2.smol.GiantGPT as smol_gpt_module
        import GIANT.v2.smol.Transformer_block as smol_block_module
    except Exception as exc:
        print(f"[dtype] Failed to import smol modules for override: {exc}")
        return

    smol_gpt_module.COMPUTE_DTYPE = dtype
    smol_block_module.COMPUTE_DTYPE = dtype
    try:
        smol_gpt_module.MODEL_CFG.compute_dtype = str(desired)
        smol_block_module.MODEL_CFG.compute_dtype = str(desired)
    except Exception:
        pass
    jax_config.update("jax_default_matmul_precision", str(desired))
    print(f"[dtype] Overriding SmolLM compute dtype to {desired}")



def _format_puzzle(puzzle: np.ndarray) -> str:
    return " ".join(str(int(x)) for x in puzzle.reshape(-1))



def _generate_puzzle(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    from TRM.sudoku.sudoku_dataset import make_puzzle, random_solved_grid

    rng = np.random.default_rng(seed)
    solution = random_solved_grid(rng)
    puzzle = make_puzzle(solution, rng, min_clues=24)
    return puzzle, solution



def _build_prompt(cfg, puzzle_str: str, user_override: Optional[str] = None) -> str:
    system_line = f"{cfg.data.system_prefix} {cfg.data.system_prompt}"
    if user_override:
        prompt_text = user_override.format(puzzle=puzzle_str)
        user_line = f"{cfg.data.user_prefix} {prompt_text}"
    else:
        user_line = f"{cfg.data.user_prefix} Solve this Sudoku: {puzzle_str}"
    assistant_line = f"{cfg.data.assistant_prefix}"
    return "\n".join([system_line, user_line, assistant_line])



def _extract_trm_digits(
    tokenizer,
    full_ids: np.ndarray,
    prompt_len: int,
) -> Tuple[Optional[np.ndarray], Optional[str], bool, bool]:
    open_id = tokenizer.convert_tokens_to_ids(TRM_OPEN)
    close_id = tokenizer.convert_tokens_to_ids(TRM_CLOSE)
    ids = full_ids.tolist()
    gen_ids = ids[prompt_len:]

    open_pos = None
    close_pos = None
    open_found = open_id is not None and open_id in gen_ids
    close_found = close_id is not None and close_id in gen_ids
    if open_found:
        open_pos = prompt_len + gen_ids.index(open_id)
    if close_found:
        close_pos = prompt_len + gen_ids.index(close_id)

    if open_pos is not None:
        start_idx = open_pos + 1
    else:
        start_idx = prompt_len
    if close_pos is not None and close_pos > start_idx:
        end_idx = close_pos
    else:
        end_idx = len(ids)

    segment = ids[start_idx:end_idx]
    if not segment:
        return None, None, open_found, close_found
    inner = tokenizer.decode(segment, skip_special_tokens=False)
    digits = [int(ch) for ch in re.findall(r"[0-9]", inner)]
    if len(digits) < 81:
        return None, inner, open_found, close_found
    return np.array(digits[:81], dtype=np.int32), inner, open_found, close_found



def solve_sudoku(puzzle: np.ndarray) -> Optional[np.ndarray]:
    puzzle = np.array(puzzle, dtype=np.int32, copy=True).reshape(-1)
    if puzzle.size != 81:
        raise ValueError("Puzzle must be 81 entries.")

    side = 9
    box = 3
    all_mask = (1 << side) - 1
    row_of = [i // side for i in range(81)]
    col_of = [i % side for i in range(81)]
    box_of = [(r // box) * box + (c // box) for r in range(side) for c in range(side)]

    row_mask = [0] * side
    col_mask = [0] * side
    box_mask = [0] * side
    empties = []

    for i, v in enumerate(puzzle):
        if v == 0:
            empties.append(i)
            continue
        bit = 1 << (int(v) - 1)
        r = row_of[i]
        c = col_of[i]
        b = box_of[i]
        if (row_mask[r] & bit) or (col_mask[c] & bit) or (box_mask[b] & bit):
            return None
        row_mask[r] |= bit
        col_mask[c] |= bit
        box_mask[b] |= bit


    def rec() -> bool:

        if not empties:
            return True

        best_i = -1
        best_mask = 0
        best_bits = 10
        for idx, pos in enumerate(empties):
            r = row_of[pos]
            c = col_of[pos]
            b = box_of[pos]
            used = row_mask[r] | col_mask[c] | box_mask[b]
            mask = all_mask & (~used)
            bits = mask.bit_count()
            if bits == 0:
                return False
            if bits < best_bits:
                best_bits = bits
                best_mask = mask
                best_i = idx
                if bits == 1:
                    break

        pos = empties.pop(best_i)
        r = row_of[pos]
        c = col_of[pos]
        b = box_of[pos]
        mask = best_mask

        while mask:
            bit = mask & -mask
            digit = int(bit.bit_length())
            puzzle[pos] = digit
            row_mask[r] |= bit
            col_mask[c] |= bit
            box_mask[b] |= bit

            if rec():
                return True

            row_mask[r] ^= bit
            col_mask[c] ^= bit
            box_mask[b] ^= bit
            puzzle[pos] = 0
            mask ^= bit

        empties.insert(best_i, pos)
        return False

    return puzzle if rec() else None


 

def _load_trm_sudoku_cfg(config_path: Optional[str], data_root: Optional[str]) -> Tuple[OmegaConf, Path]:
    cfg = OmegaConf.merge(
        OmegaConf.load(TRM_ROOT / "Global_Config.yml"),
        OmegaConf.load(TRM_ROOT / "model" / "Config.yml"),
        OmegaConf.load(config_path or (TRM_ROOT / "sudoku" / "Config.yml")),
    )
    if data_root is not None:
        base_root = Path(str(data_root)).resolve()
        if "paths" not in cfg:
            cfg.paths = OmegaConf.create({})
        cfg.paths.data_root = str(base_root)
    else:
        base_root = Path(cfg.paths.data_root) if "paths" in cfg and cfg.paths.get("data_root") else TRM_ROOT
    return cfg, base_root


 
def _build_trm_sudoku_model(cfg: OmegaConf, *, trm_module) -> Any:
    m = cfg.model
    return trm_module.TRM(
        vocab_size=int(m.vocab_size),
        context_length=int(m.context_length),
        d_model=int(m.embedding_size),
        tiny_layers=int(m.tiny_layers),
        variant=str(m.variant),
        num_heads=int(m.num_heads),
        rope_dim=int(m.rope_dim),
        d_ff=int(m.feed_forward_size),
        mixer_hidden=int(m.mixer_hidden),
        dropout_rate=float(m.dropout_rate),
        activation=str(m.activation),
        add_positional_embedding=bool(getattr(m, "add_positional_embedding", True)),
        L_cycles=int(m.recursion.L_cycles),
        H_cycles=int(m.recursion.H_cycles),
        max_supervision_steps=int(m.recursion.max_supervision_steps),
        enable_early_stop=bool(m.recursion.enable_early_stop),
        halt_threshold_logit=float(m.recursion.halt_threshold_logit),
        halt_exploration_prob=float(getattr(m.recursion, "halt_exploration_prob", 0.0)),
        no_act_continue=bool(getattr(m.recursion, "no_act_continue", True)),
        aug_enabled=bool(m.augmentation.enabled),
        aug_num_embeddings=int(m.augmentation.num_embeddings),
        aug_default_id=int(m.augmentation.default_id),
    )


 
def _resolve_trm_checkpoint(base_root: Path, checkpoint_path: Optional[str], checkpoint_dir: str) -> Path:
    if checkpoint_path:
        path = Path(checkpoint_path)
        if not path.is_absolute():
            path = (base_root / path).resolve()
        return path

    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = (base_root / ckpt_dir).resolve()
    from TRM.common import checkpoint_manager as trm_checkpoint_manager

    latest = trm_checkpoint_manager.latest(str(ckpt_dir))
    if latest is None:
        raise FileNotFoundError(f"No TRM checkpoints found in {ckpt_dir}")
    return Path(latest)


 
def build_trm_sudoku_solver(
    *,
    config_path: Optional[str],
    data_root: Optional[str],
    checkpoint_path: Optional[str],
    checkpoint_dir: str,
) -> TrmSudokuSolver:

    cfg, base_root = _load_trm_sudoku_cfg(config_path, data_root)
    ckpt_path = _resolve_trm_checkpoint(base_root, checkpoint_path, checkpoint_dir)

    from TRM.common import checkpoint_manager as trm_checkpoint_manager
    from TRM.sudoku import jit_inference as sudoku_inference

    params_np, step = trm_checkpoint_manager.load(str(ckpt_path))
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), params_np)
    if isinstance(params, dict):
        params = flax_core.freeze(params)

    model = _build_trm_sudoku_model(cfg, trm_module=sudoku_inference)
    infer_fn = sudoku_inference.make_jitted_inference(model)
    return TrmSudokuSolver(
        model=model,
        params=params,
        infer_fn=infer_fn,
        checkpoint_path=ckpt_path,
        step=int(step),
    )


 
def main() -> None:

    args = parse_args()
    cfg = load_config()

    build_custom_tokenizer(force=False)
    tokenizer = load_tokenizer()
    _apply_compute_dtype_override(cfg)

    trm_solver = None
    if args.use_trm_solver:
        trm_solver = build_trm_sudoku_solver(
            config_path=args.trm_config,
            data_root=args.trm_data_root,
            checkpoint_path=args.trm_checkpoint,
            checkpoint_dir=args.trm_checkpoint_dir,
        )
        print(f"[trm] Loaded solver checkpoint {trm_solver.checkpoint_path} (step {trm_solver.step})")

    prompt_override = args.user_prompt
    if args.user_prompt_file:
        prompt_override = Path(args.user_prompt_file).read_text(encoding="utf-8").strip()

    if args.puzzle:
        digits = [int(ch) for ch in re.findall(r"[0-9]", args.puzzle)]
        if len(digits) != 81:
            raise ValueError("Expected 81 digits for --puzzle.")
        puzzle = np.array(digits, dtype=np.int32)
        solution = None
    else:
        puzzle, solution = _generate_puzzle(args.seed)

    puzzle_str = _format_puzzle(puzzle)
    prompt = _build_prompt(cfg, puzzle_str, prompt_override)

    max_seq_len = max(stage.seq_len for stage in cfg.stages)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) >= max_seq_len:
        prompt_ids = prompt_ids[-(max_seq_len - 1) :]
    prompt_ids = np.asarray(prompt_ids, dtype=np.int32)

    model = GiantGPT(
        vocab_size=len(tokenizer),
        context_length=max_seq_len,
        d_model=cfg.model.embedding_size,
        n_heads=cfg.model.num_heads,
        d_ff=cfg.model.feed_forward_size,
        n_layers=cfg.model.num_layers,
        dropout_rate=0.0,
    )

    ckpt_path = args.checkpoint or str(Path(cfg.paths.checkpoint_root) / "step_0005000.npz")
    params = load_npz(ckpt_path)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)

    key_params, key_dropout = jax.random.split(jax.random.PRNGKey(args.seed), 2)
    _, nonparam = init_inference_state(
        model,
        key_params,
        key_dropout,
        batch_size=1,
        pad_token_id=tokenizer.pad_token_id or 0,
        use_kv_cache=True,
    )
    prefill_fn, decode_fn = make_prefill_and_decode_fns(model)

    prompt_ids_b = prompt_ids[None, :]
    nonparam, last_pos, last_tok = prefill_fn(params, nonparam, prompt_ids_b)
    new_ids, _ = decode_fn(
        params,
        nonparam,
        last_tok,
        last_pos,
        steps=int(args.max_new_tokens),
        do_sample=False,
        top_k=0,
        temperature=1.0,
        rng_key=None,
    )

    full_ids = np.concatenate([prompt_ids, np.array(new_ids[0])], axis=0)
    full_text = tokenizer.decode(full_ids, skip_special_tokens=False)

    extracted, segment_text, open_found, close_found = _extract_trm_digits(
        tokenizer, full_ids, len(prompt_ids)
    )
    print(f"TRM tokens present in generated output: open={open_found} close={close_found}")
    if extracted is None:
        print("No valid TRM digits found in model output.")
        if segment_text:
            print("=== RAW GENERATED SEGMENT ===")
            print(segment_text[:800])
        else:
            print("=== RAW OUTPUT (tail) ===")
            print(full_text[-800:])
        return

    solver_label = "python"
    if trm_solver is not None:
        solved, trm_meta = trm_solver.solve(extracted)
        solver_label = "trm"
        print(
            f"[trm] q_logit={trm_meta['q_logit']:+.3f} steps={trm_meta['steps']} "
            f"halted={trm_meta['halted']}"
        )
    else:
        solved = solve_sudoku(extracted)

    print("=== LLM PARSED TRM CALL ===")
    print(f"{TRM_OPEN} {_format_puzzle(extracted)} {TRM_CLOSE}")
    matches_input = bool(np.all(extracted.reshape(-1) == puzzle.reshape(-1)))
    print(f"Parsed puzzle matches input: {matches_input}")
    if segment_text:
        print("=== LLM GENERATED SEGMENT ===")
        print(segment_text[:800])

    if solved is None:
        print("Solver failed on extracted puzzle.")
        return

    print(f"=== SOLVER OUTPUT ({solver_label}) ===")
    print(_format_puzzle(solved))
    if solution is not None:
        match = bool(np.all(solved.reshape(-1) == solution.reshape(-1)))
        print(f"Solved matches generator solution: {match}")


if __name__ == "__main__":
    main()
