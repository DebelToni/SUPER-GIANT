"""LoRA primitives shared by GIANT v3 and TiDAR.

Base model weights remain in Flax's ``params`` collection. LoRA weights live in
an independent ``adapters`` collection so adapter-only training does not create
base-model gradients or optimizer state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict


ADAPTER_COLLECTION = "adapters"
VALID_LORA_TARGETS = frozenset({"qkv_proj", "o_proj", "fc1", "fc2"})
VALID_LORA_ROUTING = frozenset({"global", "token"})


@dataclass(frozen=True)
class LoRAConfig:
    enabled: bool = False
    rank: int = 0
    alpha: float = 1.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ()
    routing: str = "global"
    layer_indices: Optional[tuple[int, ...]] = None
    stop_gradient_before_lora: bool = False

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if self.rank <= 0:
            raise ValueError("LoRA rank must be > 0 when LoRA is enabled")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be > 0 when LoRA is enabled")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must satisfy 0 <= dropout < 1")
        if self.routing not in VALID_LORA_ROUTING:
            raise ValueError(
                f"Unsupported LoRA routing {self.routing!r}; expected one of {sorted(VALID_LORA_ROUTING)}"
            )
        unknown = set(self.target_modules) - VALID_LORA_TARGETS
        if unknown:
            raise ValueError(
                f"Unsupported LoRA target modules {sorted(unknown)}; "
                f"expected a subset of {sorted(VALID_LORA_TARGETS)}"
            )
        if not self.target_modules:
            raise ValueError("LoRA target_modules must not be empty when LoRA is enabled")
        if self.layer_indices is not None:
            if not self.layer_indices:
                raise ValueError("LoRA layer_indices must not be empty when specified")
            if any(index < 0 for index in self.layer_indices):
                raise ValueError("LoRA layer_indices must be non-negative")
            if len(set(self.layer_indices)) != len(self.layer_indices):
                raise ValueError("LoRA layer_indices must not contain duplicates")

    @property
    def scale(self) -> float:
        return self.alpha / self.rank if self.enabled else 0.0

    def targets(self, module_name: str) -> bool:
        return self.enabled and module_name in self.target_modules

    def applies_to_layer(self, layer_index: int) -> bool:
        return self.enabled and (
            self.layer_indices is None or layer_index in self.layer_indices
        )

    def first_adapter_layer(self) -> int:
        if not self.enabled:
            raise ValueError("Disabled LoRA has no adapter layer")
        return 0 if self.layer_indices is None else min(self.layer_indices)

    def validate_layer_count(self, n_layers: int) -> None:
        if self.layer_indices is None:
            return
        invalid = [index for index in self.layer_indices if index >= n_layers]
        if invalid:
            raise ValueError(
                f"LoRA layer_indices {invalid} exceed model layer range 0..{n_layers - 1}"
            )


def lora_config_from_mapping(raw: Optional[Any]) -> LoRAConfig:
    """Build a hashable LoRA config from OmegaConf, dict, dataclass, or None."""
    if raw is None:
        return LoRAConfig()
    if isinstance(raw, LoRAConfig):
        return raw

    getter = getattr(raw, "get", None)

    def get(name: str, default: Any) -> Any:
        if callable(getter):
            return getter(name, default)
        if isinstance(raw, Mapping):
            return raw.get(name, default)
        return getattr(raw, name, default)

    enabled = bool(get("enabled", False))
    targets_raw = get("target_modules", ()) or ()
    if isinstance(targets_raw, str):
        targets = (targets_raw,)
    else:
        targets = tuple(str(value) for value in targets_raw)
    layer_indices_raw = get("layer_indices", None)
    layer_indices = (
        None
        if layer_indices_raw is None
        else tuple(int(value) for value in layer_indices_raw)
    )
    return LoRAConfig(
        enabled=enabled,
        rank=int(get("rank", 0 if not enabled else 16)),
        alpha=float(get("alpha", 1.0 if not enabled else 32.0)),
        dropout=float(get("dropout", 0.0)),
        target_modules=targets,
        routing=str(get("routing", "global")).strip().lower(),
        layer_indices=layer_indices,
        stop_gradient_before_lora=bool(get("stop_gradient_before_lora", False)),
    )


def validate_adapter_mask(
    config: LoRAConfig,
    adapter_mask: Optional[jax.Array],
    token_shape: tuple[int, ...],
) -> None:
    """Validate routing without turning mask values into static JIT arguments."""
    if not config.enabled:
        return
    if config.routing == "token" and adapter_mask is None:
        raise ValueError("Token-routed LoRA requires adapter_mask on every model call")
    if adapter_mask is None:
        return
    if tuple(adapter_mask.shape) != tuple(token_shape):
        raise ValueError(
            f"adapter_mask shape {tuple(adapter_mask.shape)} must match token shape {tuple(token_shape)}"
        )
    if adapter_mask.dtype != jnp.bool_:
        raise TypeError(f"adapter_mask must have bool dtype, got {adapter_mask.dtype}")


class LoRAUpdate(nn.Module):
    """Apply a low-rank update while preserving the precomputed base result."""

    out_features: int
    rank: int
    alpha: float
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        base: jax.Array,
        *,
        adapter_mask: Optional[jax.Array],
        deterministic: bool,
    ) -> jax.Array:
        in_features = int(x.shape[-1])
        a_init = nn.initializers.variance_scaling(1.0, "fan_in", "uniform")
        b_init = nn.initializers.zeros

        # Keep make_rng inside the lazy initializer. Ordinary apply() therefore
        # needs no adapter RNG after the variables have been initialized.
        lora_a = self.variable(
            ADAPTER_COLLECTION,
            "a",
            lambda: a_init(
                self.make_rng(ADAPTER_COLLECTION),
                (in_features, self.rank),
                self.param_dtype,
            ),
        ).value
        lora_b = self.variable(
            ADAPTER_COLLECTION,
            "b",
            lambda: b_init(
                self.make_rng(ADAPTER_COLLECTION),
                (self.rank, self.out_features),
                self.param_dtype,
            ),
        ).value

        adapter_input = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        adapter_input = adapter_input.astype(self.dtype)
        delta = jnp.matmul(adapter_input, lora_a.astype(self.dtype))
        delta = jnp.matmul(delta, lora_b.astype(self.dtype))
        adapted = base + jnp.asarray(self.alpha / self.rank, dtype=self.dtype) * delta

        if adapter_mask is None:
            return adapted.astype(base.dtype)
        return jnp.where(adapter_mask[..., None], adapted, base).astype(base.dtype)


def assert_tree_compatible(template: Any, loaded: Any, *, label: str) -> None:
    """Fail early when checkpoint keys or shapes do not match a model template."""
    template_flat = flatten_dict(template)
    loaded_flat = flatten_dict(loaded)
    template_keys = set(template_flat)
    loaded_keys = set(loaded_flat)
    if template_keys != loaded_keys:
        missing = sorted("/".join(path) for path in template_keys - loaded_keys)
        extra = sorted("/".join(path) for path in loaded_keys - template_keys)
        raise ValueError(f"{label} tree mismatch: missing={missing}, extra={extra}")
    mismatched = []
    for path, expected in template_flat.items():
        actual = loaded_flat[path]
        if tuple(expected.shape) != tuple(actual.shape):
            mismatched.append(
                f"{'/'.join(path)} expected {tuple(expected.shape)}, got {tuple(actual.shape)}"
            )
    if mismatched:
        raise ValueError(f"{label} shape mismatch: " + "; ".join(mismatched))


def count_parameters(tree: Any) -> int:
    return sum(int(value.size) for value in jax.tree_util.tree_leaves(tree))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_manifest_payload(
    *,
    base_checkpoint: str | Path,
    config: LoRAConfig,
    base_parameter_count: int,
    adapter_parameter_count: int,
) -> dict[str, Any]:
    base_path = Path(base_checkpoint).expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"Adapter base checkpoint is not a file: {base_path}")
    lora_payload = asdict(config)
    lora_payload["target_modules"] = list(config.target_modules)
    if config.layer_indices is None:
        lora_payload.pop("layer_indices")
    else:
        lora_payload["layer_indices"] = list(config.layer_indices)
    if not config.stop_gradient_before_lora:
        lora_payload.pop("stop_gradient_before_lora")
    return {
        "format": "super-giant-lora-v2",
        "base_checkpoint": {
            "path": str(base_path),
            "size_bytes": int(base_path.stat().st_size),
            "sha256": _sha256_file(base_path),
        },
        "lora": lora_payload,
        "base_parameter_count": int(base_parameter_count),
        "adapter_parameter_count": int(adapter_parameter_count),
    }


def _validate_manifest_payload(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    differing = []
    for key in ("format", "lora", "base_parameter_count", "adapter_parameter_count"):
        if actual.get(key) != expected.get(key):
            differing.append(key)
    actual_base = actual.get("base_checkpoint", {})
    expected_base = expected.get("base_checkpoint", {})
    for key in ("size_bytes", "sha256"):
        if actual_base.get(key) != expected_base.get(key):
            differing.append(f"base_checkpoint.{key}")
    if not differing:
        return
    raise ValueError(
        "Adapter provenance mismatch in " + ", ".join(sorted(differing)) + ". "
        "Use the original base checkpoint and LoRA configuration."
    )


def write_adapter_manifest(
    path: str | Path,
    *,
    base_checkpoint: str | Path,
    config: LoRAConfig,
    base_parameter_count: int,
    adapter_parameter_count: int,
) -> Path:
    """Create or validate the immutable metadata for an adapter run."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    expected = _adapter_manifest_payload(
        base_checkpoint=base_checkpoint,
        config=config,
        base_parameter_count=base_parameter_count,
        adapter_parameter_count=adapter_parameter_count,
    )
    if output.exists():
        actual = json.loads(output.read_text(encoding="utf-8"))
        _validate_manifest_payload(actual, expected)
        return output
    output.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def validate_adapter_checkpoint_manifest(
    adapter_checkpoint: str | Path,
    *,
    base_checkpoint: str | Path,
    config: LoRAConfig,
    base_parameter_count: int,
    adapter_parameter_count: int,
) -> Path:
    """Validate adapter configuration and exact frozen-base SHA256 before inference."""
    adapter_path = Path(adapter_checkpoint).expanduser().resolve()
    candidates = [adapter_path.parent / "adapter_config.json"]
    if adapter_path.parent.name == "adapters":
        candidates.insert(0, adapter_path.parent.parent / "adapter_config.json")
    manifest_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if manifest_path is None:
        raise FileNotFoundError(
            f"No adapter_config.json found for adapter checkpoint {adapter_path}"
        )
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = _adapter_manifest_payload(
        base_checkpoint=base_checkpoint,
        config=config,
        base_parameter_count=base_parameter_count,
        adapter_parameter_count=adapter_parameter_count,
    )
    _validate_manifest_payload(actual, expected)
    return manifest_path


__all__ = [
    "ADAPTER_COLLECTION",
    "LoRAConfig",
    "LoRAUpdate",
    "VALID_LORA_TARGETS",
    "assert_tree_compatible",
    "count_parameters",
    "lora_config_from_mapping",
    "validate_adapter_checkpoint_manifest",
    "validate_adapter_mask",
    "write_adapter_manifest",
]
