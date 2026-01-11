from __future__ import annotations

from pathlib import Path
from typing import Tuple
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from jax.experimental import jax2tf
from omegaconf import OmegaConf

from GiantGPT import GiantGPT


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
EXPORT_DIR = MODEL_DIR / "saved_model_tflite"
TFLITE_PATH = MODEL_DIR / "transformer.tflite"


def load_configs() -> OmegaConf:
    """Merge global + local configs to pull model hyperparameters."""
    global_cfg = OmegaConf.load(PROJECT_ROOT / "Global_Config.yml")
    local_cfg = OmegaConf.load(MODEL_DIR / "Config.yml")
    return OmegaConf.merge(global_cfg, local_cfg)


def build_model(cfg: OmegaConf, vocab_size: int) -> GiantGPT:
    mcfg = cfg.model
    return GiantGPT(
        vocab_size=vocab_size,
        context_length=mcfg.context_length,
        d_model=mcfg.embedding_size,
        n_heads=mcfg.num_heads,
        d_ff=mcfg.feed_forward_size,
        n_layers=mcfg.num_layers,
        dropout_rate=0.0,
    )


def init_params(model: GiantGPT, batch_size: int, seq_len: int, seed: int = 0):
    """Initialize params for the tiny model (no cache, deterministic)."""
    rng = jax.random.PRNGKey(seed)
    key_params, key_dropout = jax.random.split(rng)
    dummy_tokens = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    variables = model.init(
        {"params": key_params, "dropout": key_dropout},
        dummy_tokens,
        deterministic=True,
        use_kv_cache=False,
    )
    return variables["params"]


def run_jax(model: GiantGPT, params, tokens: jnp.ndarray) -> jnp.ndarray:
    return model.apply(
        {"params": params},
        tokens,
        deterministic=True,
        use_kv_cache=False,
    )


def export_to_tflite(
    model: GiantGPT,
    params,
    batch_size: int,
    seq_len: int,
    export_dir: Path = EXPORT_DIR,
    tflite_path: Path = TFLITE_PATH,
) -> Path:
    """Convert the JAX forward pass to TF and then to a TFLite flatbuffer."""
    if export_dir.exists():
        shutil.rmtree(export_dir)

    params_f32 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), params)

    def jax_predict(input_ids: jnp.ndarray):
        return model.apply(
            {"params": params_f32},
            input_ids,
            deterministic=True,
            use_kv_cache=False,
        )

    tf_predict = jax2tf.convert(
        jax_predict,
        enable_xla=False,
        with_gradient=False,
    )

    class ExportModule(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(batch_size, seq_len), dtype=tf.int32)
            ]
        )
        def __call__(self, input_ids):
            return tf_predict(input_ids)

    export_module = ExportModule()
    tf.saved_model.save(export_module, export_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(export_dir))
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)
    return tflite_path


def run_tflite(tflite_path: Path, tokens: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(input_details["index"], tokens.astype(np.int32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details["index"])


def main():
    cfg = load_configs()
    vocab_size = 64
    batch_size = 1
    seq_len = min(8, int(cfg.model.context_length))

    # Build model + params
    model = build_model(cfg, vocab_size)
    params = init_params(model, batch_size, seq_len, seed=0)

    # Random tokens for the parity test
    tokens = jax.random.randint(
        jax.random.PRNGKey(42),
        (batch_size, seq_len),
        minval=0,
        maxval=vocab_size,
        dtype=jnp.int32,
    )
    tokens_np = np.asarray(tokens)
    logits_jax = run_jax(model, params, tokens)
    logits_jax_np = np.asarray(logits_jax, dtype=np.float32)
    pred_jax = logits_jax_np.argmax(axis=-1)

    # Export JAX -> TF SavedModel -> TFLite
    tflite_path = export_to_tflite(model, params, batch_size, seq_len)
    logits_tflite = run_tflite(tflite_path, tokens_np)
    pred_tflite = logits_tflite.argmax(axis=-1)

    max_diff = np.max(np.abs(logits_jax_np - logits_tflite))
    print("Random input tokens:", tokens_np)
    print("JAX argmax tokens:", pred_jax)
    print("TFLite argmax tokens:", pred_tflite)
    print("Max logits abs diff:", float(max_diff))

    np.testing.assert_allclose(
        logits_jax_np,
        logits_tflite,
        rtol=1e-5,
        atol=1e-5,
    )
    print("✅ JAX and TFLite outputs match within tolerance.")


if __name__ == "__main__":
    try:
        tf.config.experimental.set_visible_devices([], "GPU")
    except Exception:
        pass
    main()
