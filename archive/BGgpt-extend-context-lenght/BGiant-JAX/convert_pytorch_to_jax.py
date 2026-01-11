"""
Convert PyTorch BgGPT checkpoint to JAX/FLAX format.
Downloads the model from HuggingFace and saves parameters in JAX-compatible format.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any
import yaml

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

# Import torch only when needed
import torch


def load_global_config():
    """Load Global_Config.yml to get data_root."""
    config_path = Path(__file__).resolve().parent.parent / "Global_Config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def convert_pytorch_to_jax_params(pytorch_state_dict: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert PyTorch state dict to FLAX parameter tree with remapping."""
    jax_params = {}
    
    print("Converting PyTorch parameters to JAX with remapping...")
    
    # Helper to set nested dict
    def set_nested(d, keys, value):
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    # Get dimensions for compressor initialization
    head_dim = config.get('head_dim', 256)
    # Default compression ratio 1.0 means latent_dim = head_dim
    latent_dim = head_dim 
    
    # Create identity matrix for compressors (float16)
    identity = np.eye(head_dim, dtype=np.float16)
    
    # Track which layers we've seen to add compressors later
    seen_layers = set()

    # Process each parameter
    for name, param in tqdm(pytorch_state_dict.items()):
        # Skip lm_head if it shares weights with embed_tokens (Gemma2 usually does)
        if name == "lm_head.weight":
            continue

        # Convert to numpy first (handle bfloat16)
        if hasattr(param, 'cpu'):
            if param.dtype == torch.bfloat16:
                param_np = param.cpu().to(torch.float16).numpy()
            elif param.dtype == torch.float32:
                param_np = param.cpu().to(torch.float16).numpy()
            else:
                param_np = param.cpu().numpy()
        else:
            param_np = np.array(param)
            
        # --- REMAPPING LOGIC ---
        new_keys = []
        should_transpose = False
        
        if name == 'model.embed_tokens.weight':
            new_keys = ['embed_tokens', 'embedding']
            
        elif name == 'model.norm.weight':
            new_keys = ['norm', 'scale']
            
        elif name.startswith('model.layers.'):
            parts = name.split('.')
            layer_idx = parts[2]
            layer_name = f'layer_{layer_idx}'
            seen_layers.add(layer_name)
            
            if 'input_layernorm' in name:
                new_keys = [layer_name, 'input_layernorm', 'scale']
            elif 'post_attention_layernorm' in name:
                new_keys = [layer_name, 'post_attention_layernorm', 'scale']
            elif 'pre_feedforward_layernorm' in name:
                new_keys = [layer_name, 'pre_feedforward_layernorm', 'scale']
            elif 'post_feedforward_layernorm' in name:
                new_keys = [layer_name, 'post_feedforward_layernorm', 'scale']
            elif 'self_attn' in name:
                # q_proj, k_proj, v_proj, o_proj
                proj_name = parts[4]
                new_keys = [layer_name, 'self_attn', proj_name, 'kernel']
                should_transpose = True
            elif 'mlp' in name:
                # gate_proj, up_proj, down_proj
                # In FLAX model, these are direct children of the layer
                proj_name = parts[4]
                new_keys = [layer_name, proj_name, 'kernel']
                should_transpose = True
        
        if not new_keys:
            print(f"Warning: Unmapped parameter {name}")
            continue
            
        # Transpose weights for linear layers (PyTorch [out, in] -> JAX [in, out])
        if should_transpose and len(param_np.shape) == 2:
            param_np = param_np.T
            
        # Store as numpy float16
        set_nested(jax_params, new_keys, param_np.astype(np.float16))

    # Add identity weights for compressors
    print("Adding identity weights for KV compressors...")
    for layer_name in seen_layers:
        # k_compressor
        set_nested(jax_params, [layer_name, 'self_attn', 'k_compressor', 'compress', 'kernel'], identity)
        set_nested(jax_params, [layer_name, 'self_attn', 'k_compressor', 'decompress', 'kernel'], identity)
        # v_compressor
        set_nested(jax_params, [layer_name, 'self_attn', 'v_compressor', 'compress', 'kernel'], identity)
        set_nested(jax_params, [layer_name, 'self_attn', 'v_compressor', 'decompress', 'kernel'], identity)
    
    return jax_params


def download_and_convert_bggpt(
    model_name: str = "INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0",
    output_dir: Path = None,
    use_int8: bool = False,
):
    """Download BgGPT model from HuggingFace and convert to JAX."""
    
    print(f"Loading model from HuggingFace: {model_name}")
    print("Note: This requires PyTorch and transformers installed")
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: PyTorch and transformers are required for conversion")
        print("Install with: pip install torch transformers")
        return
    
    # Load the model
    print("Downloading model (this may take a while)...")
    if use_int8:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            attn_implementation="eager",
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_default_system_prompt=False,
    )
    
    # Get state dict
    print("Extracting state dict...")
    state_dict = model.state_dict()
    
    # Convert to JAX
    model_config = {
        'head_dim': model.config.head_dim,
    }
    jax_params = convert_pytorch_to_jax_params(state_dict, model_config)
    
    # Save parameters
    if output_dir is None:
        config = load_global_config()
        data_root = Path(config['paths']['data_root'])
        output_dir = data_root / "bggpt_jax_params"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving JAX parameters to {output_dir}")
    
    # Save as numpy npz for easy loading
    params_file = output_dir / "bggpt_params.npz"
    
    # Flatten the nested dict for npz
    flat_params = {}
    
    def flatten_dict(d, prefix=''):
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flatten_dict(v, new_key)
            else:
                flat_params[new_key] = np.array(v)
    
    flatten_dict(jax_params)
    
    print(f"Saving {len(flat_params)} parameters...")
    np.savez(params_file, **flat_params)
    
    # Save tokenizer
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    
    # Save model config
    config_file = output_dir / "model_config.yaml"
    model_config = {
        'vocab_size': model.config.vocab_size,
        'hidden_size': model.config.hidden_size,
        'num_layers': model.config.num_hidden_layers,
        'num_heads': model.config.num_attention_heads,
        'num_kv_heads': model.config.num_key_value_heads,
        'intermediate_size': model.config.intermediate_size,
        'head_dim': model.config.head_dim,
        'sliding_window': model.config.sliding_window,
        'rope_theta': model.config.rope_theta,
        'attn_logit_softcapping': getattr(model.config, 'attn_logit_softcapping', None),
        'final_logit_softcapping': getattr(model.config, 'final_logit_softcapping', None),
        'query_pre_attn_scalar': getattr(model.config, 'query_pre_attn_scalar', None),
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(model_config, f)
    
    print(f"\nConversion complete!")
    print(f"Parameters saved to: {params_file}")
    print(f"Tokenizer saved to: {tokenizer_dir}")
    print(f"Model config saved to: {config_file}")
    
    return output_dir


def load_jax_params(params_file: Path) -> Dict[str, Any]:
    """Load JAX parameters from npz file."""
    print(f"Loading JAX parameters from {params_file}")
    
    flat_params = np.load(params_file)
    
    # Unflatten the dict
    params = {}
    for key, value in flat_params.items():
        parts = key.split('.')
        current = params
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = jnp.asarray(value)
    
    return params


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch BgGPT to JAX format")
    parser.add_argument(
        "--model_name",
        type=str,
        default="INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to {data_root}/bggpt_jax_params)"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Load model in int8 quantization (requires bitsandbytes)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    download_and_convert_bggpt(
        model_name=args.model_name,
        output_dir=output_dir,
        use_int8=args.int8,
    )


if __name__ == "__main__":
    main()
