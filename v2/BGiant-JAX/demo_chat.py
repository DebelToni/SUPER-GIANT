"""
JAX implementation of the chat demo, matching run_bggpt_compressed.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from bggpt_compressed_kv_model_jax import create_bggpt_compressed_model
from jit_inference_bggpt import make_generate_fn
from demo_generate import load_global_config, load_model_config, load_jax_params

def main():
    # Default parameters matching PyTorch script
    model_name = "INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0"
    
    # Load config and params
    config = load_global_config()
    data_root = Path(config['paths']['data_root'])
    params_dir = data_root / "bggpt_jax_params"
    
    print(f"Loading model config from {params_dir}")
    model_config = load_model_config(params_dir)
    
    print("Creating model...")
    model = create_bggpt_compressed_model(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        num_kv_heads=model_config['num_kv_heads'],
        d_ff=model_config['intermediate_size'],
        head_dim=model_config.get('head_dim', 256),
        sliding_window=model_config.get('sliding_window', 4096),
        kv_compression_ratio=1.0,
        rope_factor=1.0,
        attn_logit_softcapping=model_config.get('attn_logit_softcapping', 50.0),
        final_logit_softcapping=model_config.get('final_logit_softcapping', 30.0),
        query_pre_attn_scalar=model_config.get('query_pre_attn_scalar', 256),
    )
    
    params_file = params_dir / "bggpt_params.npz"
    params = load_jax_params(params_file)
    
    print(f"Loading tokenizer from {params_dir / 'tokenizer'}")
    tokenizer = AutoTokenizer.from_pretrained(params_dir / "tokenizer")
    
    # ---------------------------------------------------------
    # Match PyTorch run_bggpt_compressed.py logic exactly
    # ---------------------------------------------------------
    messages = [
        {
            "role": "user",
            "content": "Кога е основана българия и от кой?",
        },
    ]
    
    print(f"\nMessages: {messages}")
    
    # Use apply_chat_template
    # Note: return_tensors="np" for JAX
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="np",
        add_generation_prompt=True,
    )
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids}")
    
    # Generation parameters from PyTorch script
    max_new_tokens = 10
    temperature = 0.4
    top_k = 40
    
    print("\nCompiling generation function...")
    generate_fn = make_generate_fn(model, params)
    
    print("Generating...")
    start_time = time.time()
    
    # Run generation
    # Note: JAX random seed handling
    rng = jax.random.PRNGKey(42)
    
    output_ids = generate_fn(
        input_ids, 
        max_new_tokens, 
        temperature, 
        top_k, 
        42 # seed
    )
    
    end_time = time.time()
    
    # Decode
    # output_ids contains [input_ids, generated_ids]
    # We want to decode the whole thing to match PyTorch script which decodes generated_ids
    # But wait, PyTorch script does:
    # generated_ids = model.generate(input_ids=input_ids, ...)
    # generated_ids = generated_ids[0].tolist()
    # generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Our generate_fn returns the full sequence (input + output)
    output_list = np.array(output_ids[0]).tolist()
    generated_text = tokenizer.decode(output_list, skip_special_tokens=True)
    
    print("\n--- Generated with BgGPT + compressed wrapper (JAX) ---")
    print(generated_text)
    print("-----------------------------------------------------------------------")
    print(f"Time: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    main()
