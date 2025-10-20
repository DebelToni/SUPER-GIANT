#!/usr/bin/env python3
# chat_generate.py
# Interactive chat CLI with <user>/<assistant> tags + top-p / temperature / greedy
import os, sys, argparse, pickle, json, time, pathlib
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import numpy as np
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from transformers import AutoTokenizer
from GiantGPT import GiantGPT

# Optional helpers if present in your repo
try:
    from checkpoint_manager import load as ckpt_load
except Exception:
    ckpt_load = None

# Our JIT helper (wraps your existing jit_inference.py if available)
from jit_chat_step import build_jitted_step

# ------------------------- prompt formatting ------------------------- #

DEFAULT_USER_TAG = "<user>"
DEFAULT_ASSIST_TAG = "<assistant>"
DEFAULT_CLOSE_TAG = ""  # if you trained with closing tags, put them here, e.g. "</assistant>"

def build_chat_preamble(system: str | None) -> str:
    if not system:
        return ""
    # If you used a system tag in training, adapt here; else just prepend as a comment
    return f"{system.strip()}\n\n"

def render_chat(history, user_tag, assist_tag, close_tag):
    """
    history: list[('user'|'assistant', text)]
    Produces the *full* prompt string up to the point where we should start generating.
    IMPORTANT: We end with the assistant open tag so generation continues there.
    """
    parts = []
    for role, text in history:
        if role == "user":
            parts.append(f"{user_tag}{text.strip()}\n")
        else:
            parts.append(f"{assist_tag}{text.strip()}{close_tag}\n")
    # generation should start after we place the assistant tag without text:
    parts.append(assist_tag)
    return "".join(parts)

# ------------------------- tokenizer loading ------------------------- #

def load_tokenizer(Config):
    # Prefer local snapshot saved during training
    if os.path.exists("tokenizer.pkl"):
        try:
            with open("tokenizer.pkl", "rb") as f:
                tok = pickle.load(f)
            # fast tokenizer saves nicely; slow might not—so fall back if it’s not usable
            if hasattr(tok, "encode") and hasattr(tok, "decode"):
                return tok
        except Exception:
            pass
    # Else from HF by name (use snapshot path if you saved to a dir)
    return AutoTokenizer.from_pretrained(
        Config.tokenizer_name,
        revision=Config.get("tokenizer_revision", None),
        use_fast=True,
        trust_remote_code=Config.get("trust_remote_code", False),
    )

# ------------------------- checkpoint loading ------------------------- #

def load_params(checkpoint_path):
    """
    Tries your checkpoint_manager first; falls back to pickle.
    Supports .npz (your checkpoint_manager), or .pkl dumps.
    """
    if ckpt_load is not None and checkpoint_path.endswith(".npz"):
        params, step = ckpt_load(checkpoint_path)
        print(f"Loaded params via checkpoint_manager (step={step})")
        return params
    # Try pickle
    try:
        with open(checkpoint_path, "rb") as f:
            obj = pickle.load(f)
        # Either params directly or a dict with "params"
        return obj.get("params", obj)
    except Exception as e:
        raise RuntimeError(
            f"Could not load checkpoint '{checkpoint_path}'. "
            f"Expected .npz (with checkpoint_manager) or .pkl. Error: {e}"
        )

# ------------------------- sampling utils ------------------------- #

def nucleus_sample(logits, top_p=0.9, temperature=0.8, rng=None):
    """Top-p sampling on a single logits vector (jnp.ndarray [V])."""
    if temperature <= 0.0:
        # greedy
        return int(jnp.argmax(logits))
    # temperature
    logits = logits / temperature
    probs = jax.nn.softmax(logits, axis=-1)
    # sort
    sorted_idx = jnp.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cdf = jnp.cumsum(sorted_probs)
    # mask to smallest set with sum >= top_p
    mask = cdf <= top_p
    # ensure at least one token
    mask = mask.at[0].set(True)
    # renormalize
    p = jnp.where(mask, sorted_probs, 0.0)
    p = p / jnp.sum(p)
    if rng is None:
        rng = jax.random.PRNGKey(int(time.time()))
    k = jax.random.categorical(rng, jnp.log(p))
    return int(sorted_idx[k])

def greedy_sample(logits):
    return int(jnp.argmax(logits))

# ------------------------- main generator ------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Chat generation with <user>/<assistant> tags (JAX/Flax).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .npz (checkpoint_manager) or .pkl params.")
    parser.add_argument("--steps", type=int, default=256, help="Max tokens to generate per turn.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Softmax temperature (0 → greedy).")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling cumulative prob.")
    parser.add_argument("--greedy", action="store_true", help="Force greedy decoding (overrides temperature/top_p).")
    parser.add_argument("--system", type=str, default=None, help="Optional system preamble.")
    parser.add_argument("--user_tag", type=str, default=DEFAULT_USER_TAG, help="User tag prefix.")
    parser.add_argument("--assistant_tag", type=str, default=DEFAULT_ASSIST_TAG, help="Assistant tag prefix.")
    parser.add_argument("--assistant_close_tag", type=str, default=DEFAULT_CLOSE_TAG, help="Assistant closing tag (if used in training).")
    parser.add_argument("--eos", type=str, default=None, help="Optional EOS string to stop generation when encountered.")
    parser.add_argument("--max_context", type=int, default=None, help="Override model input length (defaults to Config.context_length-1).")
    args = parser.parse_args()

    # Load config & tokenizer
    script_dir = pathlib.Path(__file__).resolve().parent
    Config = OmegaConf.load(script_dir / "Config.yml")
    tokenizer = load_tokenizer(Config)
    vocab_size = len(tokenizer)
    print(f"Tokenizer vocab_size={vocab_size}")

    # Build model
    ctx_len = args.max_context or (Config.context_length - 1)
    model = GiantGPT(
        vocab_size=vocab_size,
        context_length=ctx_len,
        d_model=Config.embedding_size,
        n_heads=Config.num_heads,
        d_ff=Config.feed_forward_size,
        n_layers=Config.num_layers,
        dropout_rate=0.0,  # eval
    )
    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((1, ctx_len), dtype=jnp.int32)
    params = model.init(rng, dummy, deterministic=True)["params"]

    # Load checkpoint params (shape-safe): if vocab mismatch, error clearly
    loaded = load_params(args.checkpoint)
    # quick shape guard
    def emb_shape(p):
        try:
            return p["Embed_0"]["embedding"].shape
        except Exception:
            return None
    want, got = emb_shape(params), emb_shape(loaded)
    if want is not None and got is not None and want[1] != got[1]:
        print(f"WARNING: model hidden size mismatch: init={want}, loaded={got}")
    params = loaded

    # JIT single step (tries your jit_inference helper; otherwise a simple apply)
    step_fn = build_jitted_step(model, params)

    # CLI loop
    print("\n=== Chat mode ===")
    print("Type your message and press Enter. Ctrl+C or empty line to exit.\n")
    if args.system:
        print(f"[system] {args.system}\n")

    history = []  # list of (role, text)
    eos_id = tokenizer.eos_token_id
    eos_str = args.eos

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user_text:
            print("Bye.")
            break

        history.append(("user", user_text))
        preamble = build_chat_preamble(args.system)
        prompt = preamble + render_chat(history, args.user_tag, args.assistant_tag, args.assistant_close_tag)

        # Encode prompt; keep only the last ctx_len tokens
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(input_ids) > ctx_len:
            input_ids = input_ids[-ctx_len:]
        x = jnp.asarray([input_ids], dtype=jnp.int32)

        # Autoregressive decode
        generated = []
        cur_rng = jax.random.PRNGKey(np.random.randint(0, 2**31 - 1))
        for t in range(args.steps):
            logits = step_fn(x)  # (1, T, V) or (1, V); helper returns last-pos logits
            if logits.ndim == 3:
                logits = logits[:, -1, :]
            logits = logits[0]  # (V,)

            if args.greedy or args.temperature <= 0.0:
                next_id = greedy_sample(logits)
            else:
                cur_rng, sub = jax.random.split(cur_rng)
                next_id = nucleus_sample(logits, top_p=args.top_p, temperature=args.temperature, rng=sub)

            # append
            generated.append(int(next_id))
            # stop on EOS (if defined)
            if eos_id is not None and next_id == eos_id:
                break

            # update context window (append token, keep last ctx_len)
            x = jnp.asarray([np.array(list(x[0]) + [next_id])[-ctx_len:]], dtype=jnp.int32)

            # Optional: if --eos string provided, decode partial and check
            if eos_str:
                partial_text = tokenizer.decode(generated, skip_special_tokens=False)
                if eos_str in partial_text:
                    break

        # Decode assistant turn
        assistant_text = tokenizer.decode(generated, skip_special_tokens=False).strip()
        print("\nAssistant:", assistant_text, "\n")

        # Push assistant reply to history
        history.append(("assistant", assistant_text))

if __name__ == "__main__":
    main()
