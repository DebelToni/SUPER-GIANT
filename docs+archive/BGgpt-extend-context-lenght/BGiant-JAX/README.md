# BGiant-JAX: JAX/FLAX Implementation of Compressed BgGPT

This folder contains a JAX/FLAX implementation of the compressed KV-cache BgGPT model, converted from the PyTorch implementation in `../BGiant/`.

## Overview

The implementation follows the architecture patterns established in:
- `../model/Transformer_block.py` - JAX transformer blocks with RoPE
- `../model/GiantGPT.py` - JAX GPT architecture
- `../model/Generate_faster.py` - Fast generation with KV cache
- `../model/jit_inference.py` - JIT-compiled inference with lax.scan

## Files

### Core Model
- **`bggpt_compressed_kv_model_jax.py`**: Main model implementation
  - `KVCompressor`: Linear compression/decompression for KV cache
  - `CompressedKVGemma2Attention`: Attention with compressed KV and RoPE scaling
  - `CompressedGemma2Layer`: Decoder layer with compressed attention + SwiGLU MLP
  - `CompressedBgGPTForCausalLM`: Full causal language model

### Parameter Conversion
- **`convert_pytorch_to_jax.py`**: Downloads BgGPT from HuggingFace and converts PyTorch weights to JAX format
  - Handles weight transposition (PyTorch uses [out, in], JAX uses [in, out])
  - Saves parameters as `.npz` file in the global data directory
  - Also saves tokenizer and model config

### Inference Scripts
- **`run_bggpt_compressed_jax.py`**: Basic inference script
  - Simple autoregressive generation
  - Supports temperature and top-k sampling
  - Good for debugging and understanding the model
  
- **`jit_inference_bggpt.py`**: Optimized JIT inference
  - Uses `jax.lax.scan` for efficient decoding loop
  - Separates prefill and decode phases
  - Much faster for production use
  
- **`demo_generate.py`**: Simple demo script
  - Easy-to-use interface
  - Includes benchmarking mode
  - Good for quick tests

## Usage

### 1. Convert PyTorch Model to JAX

First, download and convert the BgGPT model from HuggingFace:

```bash
cd /workspace/app/SUPER-GIANT/v2/BGiant-JAX
python convert_pytorch_to_jax.py
```

This will:
1. Download `INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0` from HuggingFace
2. Convert PyTorch parameters to JAX format
3. Save to `{data_root}/bggpt_jax_params/` (from `Global_Config.yml`)

Options:
- `--model_name`: Specify different HuggingFace model
- `--output_dir`: Custom output directory
- `--int8`: Load model in int8 quantization (requires bitsandbytes)

### 2. Run Inference

#### Quick Demo
```bash
python demo_generate.py --prompt "The meaning of life is" --max_tokens 50
```

#### With Custom Settings
```bash
python demo_generate.py \
  --prompt "Explain quantum computing:" \
  --max_tokens 100 \
  --temperature 0.7 \
  --top_k 40 \
  --kv_compression 0.5 \
  --rope_factor 2.0
```

#### Benchmark Mode
```bash
python demo_generate.py --benchmark --max_tokens 64
```

#### Basic Inference Script
```bash
python run_bggpt_compressed_jax.py \
  --prompt "Once upon a time" \
  --max_new_tokens 64 \
  --temperature 0.7
```

### 3. Parameters

#### Model Configuration
- `kv_compression_ratio`: KV cache compression (default: 1.0 = no compression)
  - Try 0.5, 0.25 for memory savings
- `rope_factor`: RoPE position scaling for context extension
  - 1.0 = 8k context, 2.0 = 16k, 8.0 = 64k

#### Generation Settings
- `temperature`: Sampling temperature (0.0 = greedy, higher = more random)
- `top_k`: Top-k sampling cutoff (0 = disabled)
- `max_new_tokens`: Number of tokens to generate
- `seed`: Random seed for reproducibility

## Architecture Details

### Key Features

1. **Compressed KV Cache**
   - Compresses key/value tensors to a lower dimension
   - Saves memory during inference
   - Decompresses before attention computation

2. **RoPE Scaling**
   - Extends context length beyond training window
   - Linear position scaling via `rope_factor`

3. **Gemma2 Features**
   - Alternating sliding window and global attention
   - Logit soft-capping for stability
   - Query pre-attention scaling
   - RMSNorm instead of LayerNorm
   - SwiGLU activation in MLP

4. **JAX Optimizations**
   - JIT compilation for speed
   - `lax.scan` for efficient loops
   - Separated prefill/decode phases
   - XLA optimization

### Model Architecture

```
CompressedBgGPTForCausalLM
├── embed_tokens (vocab_size → hidden_size)
├── layers (x num_layers)
│   ├── input_layernorm (RMSNorm)
│   ├── self_attn (CompressedKVGemma2Attention)
│   │   ├── q_proj, k_proj, v_proj
│   │   ├── k_compressor, v_compressor
│   │   ├── RoPE embeddings
│   │   └── o_proj
│   ├── post_attention_layernorm (RMSNorm)
│   ├── pre_feedforward_layernorm (RMSNorm)
│   ├── mlp (SwiGLU)
│   │   ├── gate_proj
│   │   ├── up_proj
│   │   └── down_proj
│   └── post_feedforward_layernorm (RMSNorm)
├── norm (RMSNorm)
└── lm_head (tied with embeddings)
```

## Comparison with PyTorch Version

| Feature | PyTorch (`../BGiant/`) | JAX (this folder) |
|---------|------------------------|-------------------|
| Framework | PyTorch + transformers | JAX + FLAX |
| Attention | torch.matmul | jnp.matmul |
| Modules | nn.Module | nn.Module (FLAX) |
| Compilation | torch.compile | jax.jit |
| Loops | Python for-loop | lax.scan |
| Cache | List of tuples | List of tuples |
| Weights | [out, in] | [in, out] |

## Performance Tips

1. **First run is slow** due to JIT compilation - this is normal
2. **Use benchmark mode** to measure actual steady-state performance
3. **Try KV compression** (0.25-0.5) to reduce memory and increase speed
4. **Lower precision** (bfloat16) is already used for speed
5. **Batch inference** for multiple prompts simultaneously

## Dependencies

Required:
- JAX (with GPU support recommended)
- FLAX
- transformers (for tokenizer)
- PyYAML
- numpy
- tqdm

For conversion from PyTorch:
- torch
- transformers
- (optional) bitsandbytes for int8 quantization

Install:
```bash
pip install jax[cuda12] flax transformers pyyaml numpy tqdm
# For conversion:
pip install torch transformers
```

## Troubleshooting

### "Parameter directory not found"
Run `convert_pytorch_to_jax.py` first to download and convert the model.

### "Out of memory"
- Try lower `kv_compression_ratio` (e.g., 0.5 or 0.25)
- Reduce batch size or max sequence length
- Use smaller model variant

### "Slow generation"
- First run includes JIT compilation time (normal)
- Use `--benchmark` to measure steady-state performance
- Ensure JAX is using GPU: check `jax.devices()`

### "Incorrect output"
- Check that parameters were converted correctly
- Verify tokenizer matches the model
- Try greedy decoding first (`--temperature 0.0`)

## References

- Original PyTorch implementation: `../BGiant/bggpt_compressed_kv_model.py`
- JAX transformer reference: `../model/Transformer_block.py`
- BgGPT model: [INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0](https://huggingface.co/INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0)
- Gemma2 paper: [Google Gemma 2](https://ai.google.dev/gemma)
