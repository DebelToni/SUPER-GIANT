

# run_inference.py
import torch

from compressed_kv_transformer import ModelConfig, CompressedKVTransformerLM


# -------------------------
# Tiny toy tokenizer
# -------------------------

class ToyCharTokenizer:
    """
    Super simple tokenizer:
    - maps characters in a fixed alphabet to integer IDs
    - everything else -> 0 (unk)
    - vocab_size should be >= len(alphabet)
    """

    def __init__(self, alphabet=None):
        if alphabet is None:
            # printable ascii-ish subset
            alphabet = (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
                " .,!?;:'\"-_/\\()[]{}"
            )
        self.alphabet = alphabet
        self.char2id = {ch: i + 1 for i, ch in enumerate(alphabet)}  # reserve 0 for unk
        self.id2char = {i + 1: ch for i, ch in enumerate(alphabet)}
        self.unk_id = 0

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet) + 1  # + unk

    def encode(self, text: str):
        ids = [self.char2id.get(ch, self.unk_id) for ch in text]
        return ids

    def decode(self, ids):
        chars = []
        for i in ids:
            if i == self.unk_id:
                chars.append("�")
            else:
                chars.append(self.id2char.get(int(i), "�"))
        return "".join(chars)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------------
    # Build tokenizer + config
    # -------------------------
    tokenizer = ToyCharTokenizer()
    vocab_size = tokenizer.vocab_size

    # You can make d_model / n_layers bigger if you have VRAM.
    config = ModelConfig(
        vocab_size=vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=8,
        d_ff=2048,
        max_seq_len=65536,      # 64k context
        orig_max_seq_len=8192,  # "trained" context window for RoPE scaling
        kv_compression_ratio=0.25,  # store only 1/4 of head_dim in KV cache
        dropout=0.0,
        rope_base=10000.0,
    )

    # -------------------------
    # Build model
    # -------------------------
    model = CompressedKVTransformerLM(config).to(device)
    model.eval()

    # Optional: cast to float16 to reduce VRAM (if your hardware supports it)
    # model = model.half()

    # -------------------------
    # Example prompt
    # -------------------------
    prompt = "Hello, compressed KV + 64k RoPE!"
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device,
    )  # (1, seq_len)

    print(f"Prompt: {prompt}")
    print(f"Input IDs shape: {input_ids.shape}")

    # -------------------------
    # Run generation
    # -------------------------
    with torch.no_grad():
        # first call: passes full prompt, builds full causal mask inside
        logits, past_kv = model(input_ids, past_key_values=None, use_cache=True)
        print(f"Logits shape after prompt: {logits.shape}")
        if past_kv is not None:
            k_comp, v_comp = past_kv[0]
            print(
                f"Compressed KV shape (layer 0): "
                f"K={k_comp.shape}, V={v_comp.shape} (latent dim = {k_comp.shape[-1]})"
            )

        # now let's also try the convenience generate() method
        max_new_tokens = 64
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=20,
        )

    generated_ids = generated[0].tolist()
    generated_text = tokenizer.decode(generated_ids)

    print("\n--- Generated (random-weights gibberish, but pipeline works) ---")
    print(generated_text)
    print("----------------------------------------------------------------")

    # -------------------------
    # (Optional) Rough long-context smoke test
    # -------------------------
    # This just pushes position_ids near 64k to make sure nothing crashes.
    # NOTE: This will be slow and memory-hungry if you go too big; adjust length.
    long_length = 2048  # you can increase if you want
    long_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, long_length),
        device=device,
        dtype=torch.long,
    )
    with torch.no_grad():
        logits_long, past_kv_long = model(long_ids, past_key_values=None, use_cache=True)
    print(f"\nLong run (len={long_length}) completed. Logits shape: {logits_long.shape}")


if __name__ == "__main__":
    main()

