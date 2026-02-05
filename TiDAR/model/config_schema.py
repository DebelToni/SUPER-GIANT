"""
Structured config schema for TiDAR using OmegaConf's structured configs.

This module defines dataclasses that mirror the YAML config structure,
providing type hints and IDE/LSP support for OmegaConf configurations.

Usage:
    from TiDAR.model.config_schema import TiDARConfig
    from omegaconf import OmegaConf
    
    # Load YAML and merge with structured schema
    cfg = OmegaConf.structured(TiDARConfig)
    cfg = OmegaConf.merge(cfg, OmegaConf.load("Config.yml"))
    
    # Now cfg.model.embedding_size has proper type hints
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    embedding_size: int = 576
    num_heads: int = 9
    num_kv_heads: int = 3
    num_layers: int = 30
    feed_forward_size: int = 1536
    rope_dim: int = 64
    context_length: int = 2048
    dropout_rate: float = 0.0
    use_remat: bool = False
    param_dtype: str = "float32"
    compute_dtype: str = "bfloat16"


@dataclass
class TiDARSpecificConfig:
    """TiDAR-specific configuration (draft tokens, attention)."""
    draft_length: int = 8
    attn_bias_value: float = -1.0e10


@dataclass
class InferenceConfig:
    """Inference-time configuration."""
    stop_on_eos: bool = True
    max_decode_steps: int = 64
    temperature: float = 1.0
    top_k: int = 0
    draft_length: int = 8


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    base_learning_rate: float = 1.0e-4
    min_learning_rate: float = 1.0e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    gradient_clip_norm: float = 0.5
    weight_decay_exclusions: List[str] = field(default_factory=lambda: ["bias", "scale"])


@dataclass
class LossConfig:
    """Loss function coefficients for 7-term loss.
    
    Loss = alpha*L_AR + beta*L_Diff + rho*KL_fwd + chi*KL_rev
           + delta*L_hard + eta*L_distill + gamma*L_topk
    
    Each term with coefficient=0 is skipped entirely (no compute, different JIT trace).
    """
    alpha: float = 1.0    # AR next-token prediction CE loss
    beta: float = 1.0     # Diffusion denoising CE loss
    rho: float = 0.0      # Forward KL: KL(P_AR || Q_Diff)
    chi: float = 0.0      # Reverse KL: KL(Q_Diff || P_AR)
    delta: float = 0.0    # Hard agreement: CE(onehot(argmax P_AR), logits_diff)
    eta: float = 0.0      # Soft distillation: KL(softmax(AR/T) || softmax(Diff/T))
    eta_T: float = 1.0    # Distillation temperature
    gamma: float = 0.0    # Top-K set distillation loss
    gamma_topk: int = 8   # Top-K size for set distillation


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 8
    gradient_accumulation: int = 4
    max_epochs: int = 1
    seed: int = 0
    dataset_dir: str = "dataset_artifacts"
    log_every: int = 500
    checkpoint_every: int = 20000
    mini_checkpoint_every: int = 100
    mini_max_to_keep: int = 2
    scan_chunk: int = 1
    prefetch_size: Optional[int] = None
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class StageLossConfig:
    """Optional per-stage loss coefficient overrides."""
    alpha: Optional[float] = None
    beta: Optional[float] = None
    rho: Optional[float] = None
    chi: Optional[float] = None
    delta: Optional[float] = None
    eta: Optional[float] = None
    eta_T: Optional[float] = None
    gamma: Optional[float] = None
    gamma_topk: Optional[int] = None


@dataclass
class StageConfig:
    """Training stage configuration."""
    name: str = ""
    dataset: str = ""
    seq_len: int = 2048
    epochs: int = 1
    end_ratio: float = 1.0
    shuffle: bool = True
    fraction: float = 1.0
    loss: Optional[StageLossConfig] = None


@dataclass
class TokenizerConfig:
    """Tokenizer configuration."""
    name: str = "HuggingFaceTB/cosmo2-tokenizer"
    use_custom: bool = False
    custom_path: Optional[str] = None
    cache_dir: Optional[str] = None
    pad_token_override: Optional[str] = None
    mask_token_override: Optional[str] = None


@dataclass
class PathsConfig:
    """Paths configuration (typically from Global_Config.yml)."""
    data_root: str = ""
    processed_data_root: str = ""
    dataloader_state_root: str = ""
    logs_root: str = ""
    checkpoints_root: str = ""
    hf_cache_root: str = ""


@dataclass
class TiDARConfig:
    """
    Root configuration for TiDAR training and inference.
    
    This dataclass mirrors the structure of Config.yml + Global_Config.yml
    and provides full type hints for IDE/LSP support.
    
    Example:
        >>> from omegaconf import OmegaConf
        >>> from TiDAR.model.config_schema import TiDARConfig
        >>> 
        >>> # Create typed config from YAML
        >>> schema = OmegaConf.structured(TiDARConfig)
        >>> yaml_cfg = OmegaConf.load("Config.yml")
        >>> cfg = OmegaConf.merge(schema, yaml_cfg)
        >>> 
        >>> # Now with full type hints:
        >>> print(cfg.model.embedding_size)  # int
        >>> print(cfg.training.loss.alpha)   # float
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    tidar: TiDARSpecificConfig = field(default_factory=TiDARSpecificConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    stages: List[StageConfig] = field(default_factory=list)
    qa_finetune: dict = field(default_factory=dict)
    global_seed: Optional[int] = None


# Type aliases for convenience
Config = TiDARConfig


def load_typed_config(
    model_config_path: Optional[str] = None,
    global_config_path: Optional[str] = None,
) -> TiDARConfig:
    """
    Load and merge configs with structured schema for full type support.
    
    This is a drop-in replacement for load_configs() that returns a typed config.
    
    Args:
        model_config_path: Path to model Config.yml (default: TiDAR/model/Config.yml)
        global_config_path: Path to Global_Config.yml (default: TiDAR/Global_Config.yml)
    
    Returns:
        Merged configuration with full type annotations.
    """
    from omegaconf import OmegaConf
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

    schema = OmegaConf.structured(TiDARConfig)
    OmegaConf.set_struct(schema, False)

    global_cfg = OmegaConf.load(resolved_global_cfg) if resolved_global_cfg.exists() else OmegaConf.create()
    model_cfg = OmegaConf.load(resolved_model_cfg) if resolved_model_cfg.exists() else OmegaConf.create()
    cfg = OmegaConf.merge(schema, global_cfg, model_cfg)

    base_prefix_str = cfg.paths.data_root if cfg.paths.data_root else ""
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
        custom_path = Path(str(custom_path))
        if not custom_path.is_absolute():
            cfg.tokenizer.custom_path = str(Path(cfg.paths.data_root) / custom_path)

    return cfg  # type: ignore[return-value]
