import torch

from bggpt_compressed_kv_model import (
    load_bggpt_compressed,
    load_bggpt_compressed_int8,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        # INT8 quantized weights on GPU
        model, tokenizer = load_bggpt_compressed_int8(
            model_name="INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0",
            kv_compression_ratio=1.0,   # test identity first
            rope_factor=1.0,
            device=device,
        )
    else:
        # Old bf16/float32 path for CPU
        model, tokenizer = load_bggpt_compressed(
            model_name="INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0",
            kv_compression_ratio=1.0,
            rope_factor=1.0,
            dtype=torch.bfloat16,
            device=device,
        )

    messages = [
        {
            "role": "user",
            "content": "How are you? Answer in english.",
        },
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)

    print(f"Input IDs shape: {input_ids.shape}")

    # First forward pass (prompt)
    with torch.no_grad():
        logits, past_kv = model(input_ids, past_key_values=None, use_cache=True)

    print(f"Logits shape after prompt: {logits.shape}")
    if past_kv is not None and len(past_kv) > 0:
        k_comp, v_comp = past_kv[0]
        print(
            f"Compressed KV (layer 0): "
            f"K={k_comp.shape}, V={v_comp.shape} (latent dim = {k_comp.shape[-1]})"
        )

    # Autoregressive generation
    max_new_tokens = 10
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.4,
            top_k=40,
        )

    generated_ids = generated_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n--- Generated with BgGPT + compressed wrapper (identity settings) ---")
    print(generated_text)
    print("-----------------------------------------------------------------------")

if __name__ == "__main__":
    main()
