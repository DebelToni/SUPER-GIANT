
# run_bggpt_baseline.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        attn_implementation="eager",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_default_system_prompt=False,
    )

    messages = [
        {
            "role": "user",
            "content": "Обясни ми с няколко изречения какво прави този модел.",
        },
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=128,
            temperature=0.4,
            top_k=40,
        )

    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
