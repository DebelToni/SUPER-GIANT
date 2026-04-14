from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from GIANT.v3.model.GiantGPT import GiantGPT
from GIANT.v3.model.Chat import generate_tokens
from GIANT.v3.model.checkpoint_manager import load_npz, latest as latest_ckpt
from GIANT.v3.model.jit_inference import init_inference_state, make_prefill_and_decode_fns
try:
    from GIANT.v3.device_utils import select_default_device
except ImportError:  # pragma: no cover - temporary fallback for dirty worktrees
    from GIANT.v3.tests.device_utils import select_default_device

@dataclass
class ModelConfig:
    embedding_size: int = 640
    num_heads: int = 10
    num_kv_heads: Optional[int] = None
    num_layers: int = 16
    feed_forward_size: int = 2560
    rope_dim: Optional[int] = None
    context_length: int = 2048
    dropout_rate: float = 0.0
    activation: str = "silu"
    use_remat: bool = False
    enable_xsa: bool = False
    param_dtype: str = "float32"
    compute_dtype: str = "bfloat16"


@dataclass
class InferenceConfig:
    stop_on_eos: bool = True
    max_decode_steps: int = 128
    temperature: float = 0.0
    top_k: int = 0
    draft_length: Optional[int] = None
    kv_cache_buckets: Optional[list[int]] = None


@dataclass
class TokenizerConfig:
    name: str = "clowman/Llama-3.2-3B-Instruct-AWQ-Int4"
    use_custom: bool = False
    custom_path: Optional[str] = None
    cache_dir: Optional[str] = None
    pad_token_override: Optional[str] = None
    mask_token_override: Optional[str] = None


@dataclass
class PathsConfig:
    data_root: str = ""
    processed_data_root: str = ""
    dataloader_state_root: str = ""
    logs_root: str = ""
    checkpoints_root: str = "checkpoints"
    hf_cache_root: str = ""


@dataclass
class GenerateFasterConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    qa_finetune: dict = field(default_factory=dict)


def load_typed_config(
    model_config_path: Optional[str] = None,
    global_config_path: Optional[str] = None,
) -> GenerateFasterConfig:
    """Merge global + model config into a typed schema and resolve relative paths."""
    model_dir = Path(__file__).resolve().parent
    project_root = model_dir.parent

    def resolve_config_path(value: Optional[str], default: Path) -> Path:
        if value is None:
            return default
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        return candidate

    resolved_model_cfg = resolve_config_path(model_config_path, model_dir / "Config.yml")
    resolved_global_cfg = resolve_config_path(global_config_path, project_root / "Global_Config.yml")

    schema = OmegaConf.structured(GenerateFasterConfig)
    OmegaConf.set_struct(schema, False)

    global_cfg = OmegaConf.load(resolved_global_cfg) if resolved_global_cfg.exists() else OmegaConf.create()
    model_cfg = OmegaConf.load(resolved_model_cfg) if resolved_model_cfg.exists() else OmegaConf.create()
    cfg = OmegaConf.merge(schema, global_cfg, model_cfg)

    base_prefix_str = str(cfg.paths.data_root) if cfg.paths.data_root else ""
    base_prefix = Path(base_prefix_str) if base_prefix_str else None

    def resolve_path(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        path = Path(str(value))
        if path.is_absolute() or base_prefix is None:
            return str(path)
        return str((base_prefix / path).resolve())

    if base_prefix is not None:
        cfg.paths.data_root = str(base_prefix)
    else:
        cfg.paths.data_root = str(project_root)

    for key in (
        "processed_data_root",
        "dataloader_state_root",
        "logs_root",
        "checkpoints_root",
        "hf_cache_root",
    ):
        value = getattr(cfg.paths, key, None)
        if value is not None:
            resolved = resolve_path(value)
            if resolved is not None:
                setattr(cfg.paths, key, resolved)

    cache_dir = cfg.tokenizer.cache_dir
    if cache_dir:
        cache_path = Path(str(cache_dir))
        if not cache_path.is_absolute():
            cfg.tokenizer.cache_dir = str(Path(cfg.paths.data_root) / cache_path)

    custom_path = cfg.tokenizer.custom_path
    if custom_path:
        custom_path_path = Path(str(custom_path))
        if not custom_path_path.is_absolute():
            cfg.tokenizer.custom_path = str(Path(cfg.paths.data_root) / custom_path_path)

    qa_ft = getattr(cfg, "qa_finetune", None)
    if qa_ft is not None:
        if "answers_arrow" in qa_ft and qa_ft.answers_arrow is not None:
            resolved = resolve_path(qa_ft.answers_arrow)
            if resolved is not None:
                qa_ft.answers_arrow = resolved
        if "checkpoint_dir" in qa_ft and qa_ft.checkpoint_dir is not None:
            resolved = resolve_path(qa_ft.checkpoint_dir)
            if resolved is not None:
                qa_ft.checkpoint_dir = resolved

    return cfg  # type: ignore[return-value]


def load_configs(
    model_config_path: Optional[str] = None,
    global_config_path: Optional[str] = None,
) -> GenerateFasterConfig:
    return load_typed_config(model_config_path, global_config_path)


def resolve_with_data_root(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def resolve_params_dir(root: Path) -> Path:
    return root if root.name == "params" else root / "params"


def resolve_checkpoint_path(cfg: GenerateFasterConfig, checkpoint: Optional[str], checkpoint_dir: Optional[str]) -> Path:
    base_root = Path(str(cfg.paths.data_root))
    effective_ckpt_dir = checkpoint_dir or str(cfg.paths.checkpoints_root)

    if checkpoint and checkpoint.lower() != "latest":
        path = Path(checkpoint)
        if not path.is_absolute():
            path = resolve_with_data_root(base_root, checkpoint)
        if path.is_dir():
            params_dir = resolve_params_dir(path)
            latest = latest_ckpt(str(params_dir))
            if latest is None:
                raise FileNotFoundError(f"No checkpoints found under {params_dir}")
            return Path(latest)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")
        return path

    ckpt_dir = Path(effective_ckpt_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = resolve_with_data_root(base_root, effective_ckpt_dir)
    params_dir = resolve_params_dir(ckpt_dir)
    latest = latest_ckpt(str(params_dir))
    if latest is None:
        raise FileNotFoundError(f"No checkpoints found under {params_dir}")
    return Path(latest)


def load_tokenizer(cfg: GenerateFasterConfig):
    tok_cfg = cfg.tokenizer
    if tok_cfg.use_custom:
        if not tok_cfg.custom_path:
            raise ValueError("tokenizer.use_custom=true requires tokenizer.custom_path to be set.")
        tokenizer = AutoTokenizer.from_pretrained(tok_cfg.custom_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tok_cfg.name,
            use_fast=True,
            cache_dir=tok_cfg.cache_dir,
        )
    if tok_cfg.pad_token_override:
        tokenizer.pad_token = tok_cfg.pad_token_override
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def build_model(cfg: GenerateFasterConfig, vocab_size: int, context_length: int) -> GiantGPT:
    model_cfg = cfg.model
    return GiantGPT(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_cfg.embedding_size,
        n_heads=model_cfg.num_heads,
        d_ff=model_cfg.feed_forward_size,
        n_layers=model_cfg.num_layers,
        num_kv_heads=model_cfg.num_kv_heads,
        rotary_dim=model_cfg.rope_dim,
        dropout_rate=0.0,
        param_dtype=model_cfg.param_dtype,
        compute_dtype=model_cfg.compute_dtype,
        use_remat=bool(model_cfg.use_remat),
        enable_xsa=bool(model_cfg.enable_xsa),
        causal=bool(getattr(model_cfg, "causal", True)),
    )


def tokenize_prompt(tokenizer, prompt: str, max_len: int, *, strip_eos: bool) -> np.ndarray:
    if strip_eos:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if ids and tokenizer.eos_token_id is not None and ids[-1] == tokenizer.eos_token_id:
            ids = ids[:-1]
    else:
        ids = tokenizer(prompt, return_tensors="np").input_ids[0].tolist()

    if len(ids) >= max_len:
        ids = ids[-max_len:]
    return np.asarray(ids, dtype=np.int32)


def load_params(path: Path):
    params = load_npz(path)
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)


def _lecun_embedding_init(key: jax.Array, shape: tuple[int, ...], dtype: jnp.dtype) -> jnp.ndarray:
    fan_in = shape[-1]
    std = 1.0 / jnp.sqrt(jnp.asarray(fan_in, dtype=jnp.float32))
    return jax.random.normal(key, shape, dtype) * std


def align_tokenizer_and_params_vocab(
    tokenizer,
    params,
    *,
    rng_key: jax.Array,
) -> tuple[object, object, int, int]:
    """Utility alignment only: make tokenizer vocab size and embed rows match."""
    embedding = params["Embed_0"]["embedding"]
    param_vocab_size, hidden = embedding.shape
    tokenizer_vocab_size = len(tokenizer)
    added_token_rows = 0
    added_param_rows = 0

    if tokenizer_vocab_size < param_vocab_size:
        missing = int(param_vocab_size - tokenizer_vocab_size)
        additional = [f"[EXTRA_TOKEN_{i}]" for i in range(missing)]
        tokenizer.add_special_tokens({"additional_special_tokens": additional})
        added_token_rows = len(tokenizer) - tokenizer_vocab_size
        if len(tokenizer) != param_vocab_size:
            raise ValueError(
                f"Tokenizer/params vocab mismatch after tokenizer expansion: "
                f"tokenizer={len(tokenizer)}, params={param_vocab_size}"
            )
        return tokenizer, params, added_token_rows, added_param_rows

    if tokenizer_vocab_size > param_vocab_size:
        missing = int(tokenizer_vocab_size - param_vocab_size)
        new_rows = _lecun_embedding_init(rng_key, (missing, hidden), embedding.dtype)
        params["Embed_0"]["embedding"] = jnp.concatenate([embedding, new_rows], axis=0)
        added_param_rows = missing
    return tokenizer, params, added_token_rows, added_param_rows


def block_until_ready(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def parse_bool_flag(value: Optional[str], *, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value '{value}'. Use one of: true/false, yes/no, 1/0.")


def _top_k_logits(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    if k <= 0:
        return logits
    topk_vals, _ = jax.lax.top_k(logits, k)
    kth = topk_vals[..., -1, None]
    return jnp.where(logits < kth, -jnp.inf, logits)


def parse_int_list(value: Optional[str]) -> list[int]:
    if value is None:
        return []
    raw = [part.strip() for part in value.split(",")]
    out: list[int] = []
    for part in raw:
        if not part:
            continue
        parsed = int(part)
        if parsed <= 0:
            raise ValueError(f"Bucket sizes must be positive, got {parsed}.")
        out.append(parsed)
    return out


def normalize_buckets(buckets: list[int], *, max_context: int) -> list[int]:
    uniq = sorted(set(int(x) for x in buckets if int(x) > 0 and int(x) <= max_context))
    return uniq


def default_auto_buckets(max_context: int) -> list[int]:
    base = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    return [x for x in base if x <= max_context]


def choose_bucketed_context_length(required_len: int, *, max_context: int, buckets: list[int]) -> int:
    if required_len > max_context:
        raise ValueError(
            f"Required length {required_len} exceeds context_length {max_context}. "
            "Increase --context_length/--max_context or lower --steps."
        )
    for b in buckets:
        if b >= required_len:
            return b
    return max_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast text generation using the current SUPER-GIANT layout.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model config YAML (defaults to GIANT/v3/model/Config.yml).")
    parser.add_argument("--global_config", type=str, default=None,
                        help="Path to global config YAML (defaults to GIANT/v3/Global_Config.yml).")
    parser.add_argument("--checkpoint", type=str, default="latest",
                        help="Path to a checkpoint (.npz). Defaults to the newest file in --checkpoint_dir.")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory (relative to data_root) used when --checkpoint is omitted or set to 'latest'.")
    parser.add_argument("--prompt", type=str, default="Once upon",
                        help="Prompt to feed the model.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature. Zero switches to greedy decoding.")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Optional top-k sampling cutoff (0 disables it).")
    parser.add_argument("--greedy", action="store_true",
                        help="Shortcut for --temperature 0.0.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for sampling.")
    parser.add_argument("--max_context", "--context_length", type=int, default=None, dest="max_context",
                        help="Override context length from config.")
    parser.add_argument("--stop_on_eos", type=str, default=None,
                        help="Override inference.stop_on_eos (true/false).")
    parser.add_argument("--kv_cache_buckets", type=str, default=None,
                        help="Comma-separated KV cache buckets, e.g. 128,256,512. Smallest fitting bucket is used.")
    parser.add_argument("--disable_kv_buckets", action="store_true",
                        help="Disable KV bucket selection and use full context_length directly.")
    parser.add_argument("--strip_eos", "--no_eos", action="store_true", dest="strip_eos",
                        help="Drop a trailing EOS token from the prompt before generation.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print timing stats.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_configs(args.config, args.global_config)
    jax.config.update("jax_default_matmul_precision", cfg.model.compute_dtype)
    device = select_default_device()

    cfg_temperature = float(getattr(cfg.inference, "temperature", 0.0))
    cfg_top_k = int(getattr(cfg.inference, "top_k", 0))
    cfg_steps = int(getattr(cfg.inference, "max_decode_steps", 128))
    cfg_stop_on_eos = bool(getattr(cfg.inference, "stop_on_eos", True))

    max_steps = args.steps if args.steps is not None else cfg_steps
    input_temperature = args.temperature if args.temperature is not None else cfg_temperature
    temperature = 0.0 if args.greedy else max(float(input_temperature), 0.0)
    top_k = int(args.top_k) if args.top_k is not None else cfg_top_k
    stop_on_eos = parse_bool_flag(args.stop_on_eos, default=cfg_stop_on_eos)

    if max_steps <= 0:
        raise ValueError("steps must be > 0")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    do_sample = temperature > 0.0

    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint, args.checkpoint_dir)
    print(f"Using checkpoint: {checkpoint_path}")

    tokenizer = load_tokenizer(cfg)
    model_context_length = int(cfg.model.context_length)
    requested_context_length = args.max_context or model_context_length
    if requested_context_length <= 0:
        raise ValueError("context_length must be > 0")
    if requested_context_length > model_context_length:
        raise ValueError(
            f"context_length {requested_context_length} exceeds model max {model_context_length}"
        )

    prompt_ids = tokenize_prompt(
        tokenizer,
        args.prompt,
        requested_context_length,
        strip_eos=args.strip_eos,
    )
    if prompt_ids.size == 0:
        raise ValueError("Prompt produced zero tokens. Provide non-empty text.")
    prompt_len = int(prompt_ids.shape[0])
    required_len = prompt_len + max_steps

    cfg_buckets_raw = getattr(cfg.inference, "kv_cache_buckets", None)
    cfg_buckets: list[int] = []
    if cfg_buckets_raw is not None:
        if isinstance(cfg_buckets_raw, str):
            cfg_buckets = parse_int_list(cfg_buckets_raw)
        else:
            cfg_buckets = [int(x) for x in cfg_buckets_raw]
    cli_buckets = parse_int_list(args.kv_cache_buckets)

    if args.disable_kv_buckets:
        active_buckets: list[int] = []
        bucket_source = "disabled"
    elif cli_buckets:
        active_buckets = normalize_buckets(cli_buckets, max_context=requested_context_length)
        bucket_source = "cli"
    elif cfg_buckets:
        active_buckets = normalize_buckets(cfg_buckets, max_context=requested_context_length)
        bucket_source = "config"
    else:
        active_buckets = default_auto_buckets(requested_context_length)
        bucket_source = "auto"

    context_length = choose_bucketed_context_length(
        required_len,
        max_context=requested_context_length,
        buckets=active_buckets,
    )

    params = load_params(checkpoint_path)
    rng = jax.random.PRNGKey(args.seed)
    rng, vocab_align_key = jax.random.split(rng)
    tokenizer, params, added_token_rows, added_param_rows = align_tokenizer_and_params_vocab(
        tokenizer,
        params,
        rng_key=vocab_align_key,
    )

    model = build_model(cfg, len(tokenizer), context_length)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    key_params, key_dropout, key_sample = jax.random.split(rng, 3)
    _, nonparam = init_inference_state(
        model,
        key_params,
        key_dropout,
        batch_size=1,
        pad_token_id=pad_token_id,
        use_kv_cache=True,
    )

    params = jax.device_put(params, device)
    nonparam = jax.device_put(nonparam, device)

    base_state = nonparam
    prefill_fn, decode_fn = make_prefill_and_decode_fns(model)
    tokens_new, prefill_time, decode_time, key_sample = generate_tokens(
        params=params,
        base_state=base_state,
        prefill_fn=prefill_fn,
        decode_fn=decode_fn,
        prompt_ids=prompt_ids,
        steps=max_steps,
        temperature=temperature,
        top_k=top_k,
        do_sample=do_sample,
        rng_key=key_sample,
    )

    full_tokens = np.concatenate([prompt_ids, tokens_new], axis=0)
    text = tokenizer.decode(full_tokens, skip_special_tokens=True)

    eos_id = tokenizer.eos_token_id
    if stop_on_eos and eos_id is not None:
        idx = np.where(full_tokens == eos_id)[0]
        if idx.size > 0:
            cut = int(idx[0])
            text = tokenizer.decode(full_tokens[:cut], skip_special_tokens=True) + "<EOS>"

    print("\n==================== RESULT ====================")
    print(text)
    print("================================================")

    if args.verbose:
        toks_per_s = (max_steps / decode_time) if decode_time > 0 else float("inf")
        print("\n[perf]")
        print(f"prompt_tokens: {len(prompt_ids)}")
        print(f"generated_tokens: {max_steps}")
        print(f"prefill_time_s: {prefill_time:.6f}")
        print(f"decode_time_s:  {decode_time:.6f}")
        print(f"tokens_per_second_decode: {toks_per_s:.6f}")
        print("\n[resolved_settings]")
        print(f"config: {args.config if args.config is not None else 'GIANT/v3/model/Config.yml'}")
        print(f"global_config: {args.global_config if args.global_config is not None else 'GIANT/v3/Global_Config.yml'}")
        print(f"requested_context_length: {requested_context_length}")
        print(f"context_length: {context_length}")
        print(f"steps: {max_steps}")
        print(f"temperature: {temperature}")
        print(f"top_k: {top_k}")
        print(f"stop_on_eos: {stop_on_eos}")
        print(f"kv_bucket_source: {bucket_source}")
        print(f"kv_buckets: {active_buckets if active_buckets else 'none'}")
        print(f"tokenizer_vocab_added_rows: {added_token_rows}")
        print(f"params_vocab_added_rows: {added_param_rows}")


if __name__ == "__main__":
    main()
