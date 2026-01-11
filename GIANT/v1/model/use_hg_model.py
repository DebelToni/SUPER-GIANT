#!/usr/bin/env python3
"""
Simple CLI chat script using Hugging Face Transformers and Microsoft Phi-1_5 model.

Usage:
    python chat_phi1_5_cli.py [--max-new-tokens N] [--temperature T]

Then type your prompts interactively.
"""
import argparse
import torch
from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForCausalLM

def detect_device():
    # Prefer GPU (CUDA), then Apple MPS, then CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Chat with Microsoft Phi-1_5 via Transformers pipeline"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Maximum number of new tokens to generate (default: 256)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    device = detect_device()
    print(f"Using device: {device}")

    # Load tokenizer and model
    print("Loading model microsoft/phi-1_5 ...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-1_5",
        device_map="auto" if device.type != "cpu" else None,
        torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
        trust_remote_code=True
    )

    chat_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device.index if hasattr(device, 'index') else -1,
        truncation=True
    )
    print("Model loaded. Start chatting! (type 'exit' or Ctrl+C to quit)")

    try:
        while True:
            prompt = input("\nYou: ")
            if prompt.strip().lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            # Include a basic conversational template
            full_prompt = f"Human: {prompt}\nAssistant:"
            outputs = chat_pipe(
                full_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1
            )

            generated = outputs[0]["generated_text"]
            # Extract reply text after the template
            reply = generated[len(full_prompt):].split('\n')[0].strip()
            print(f"Phi-1_5: {reply}")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Goodbye!")


if __name__ == "__main__":
    main()

