from __future__ import annotations

from typing import Any


VALID_MODEL_MODES = {"decoder"}


def normalize_model_mode(value: Any) -> str:
    mode = str(value or "decoder").strip().lower()
    if mode not in VALID_MODEL_MODES:
        raise ValueError("GIANT v3 only supports decoder mode")
    return mode


def resolve_model_mode(model_cfg: Any) -> str:
    mode_value = getattr(model_cfg, "mode", None)
    getter = getattr(model_cfg, "get", None)
    if mode_value is None and callable(getter):
        mode_value = getter("mode", None)
    if mode_value is not None:
        return normalize_model_mode(mode_value)

    causal_value = getattr(model_cfg, "causal", None)
    if causal_value is None and callable(getter):
        causal_value = getter("causal", True)
    if causal_value is False:
        raise ValueError("GIANT v3 only supports causal decoder models")
    return "decoder"


def model_mode_is_causal(mode: str) -> bool:
    normalize_model_mode(mode)
    return True


__all__ = ["VALID_MODEL_MODES", "model_mode_is_causal", "normalize_model_mode", "resolve_model_mode"]
