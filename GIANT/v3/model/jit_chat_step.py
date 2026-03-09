# jit_chat_step.py
# A tiny adapter around your jit_inference.py if available; otherwise fall back
import importlib
import jax
import jax.numpy as jnp

def _fallback_step(model, params):
    """
    Returns a function x -> logits(last position), where x is (1, T) int32.
    No KV-cache, but JIT-compiled so subsequent calls are fast enough for chats.
    """
    @jax.jit
    def step(x):
        logits = model.apply({"params": params}, x, deterministic=True)  # (1, T, V)
        return logits[:, -1, :]  # (1, V)
    return step

def build_jitted_step(model, params):
    """
    If your repo has a 'jit_inference.py' with a compatible API, use it.
    Expected optional functions:
      - build_step(model, params) -> callable(x)->(1,V)
    Otherwise, return a safe JIT fallback.
    """
    try:
        module_name = "jit_inference"
        if __package__:
            module_name = f"{__package__}.jit_inference"
        mod = importlib.import_module(module_name)
        if hasattr(mod, "build_step"):
            return mod.build_step(model, params)
        # or your older helpers:
        if hasattr(mod, "jit_forward_last"):
            return mod.jit_forward_last(model, params)
    except Exception:
        pass
    # Fallback
    return _fallback_step(model, params)
