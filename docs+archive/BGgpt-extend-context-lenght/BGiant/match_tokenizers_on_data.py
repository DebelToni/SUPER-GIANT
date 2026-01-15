import argparse
from collections import Counter
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Inspect a Hugging Face tokenizer and optionally report token usage on a text file."
    )
    parser.add_argument("tokenizer", nargs="?", default="bert-base-uncased",
                        help="Tokenizer name or path (default: bert-base-uncased)")
    parser.add_argument("--test-text", "-t", type=str,
                        help="Path to a text file to inspect token usage.")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab = set(tok.get_vocab().keys())

    print(f"Tokenizer: {args.tokenizer}")
    print(f"  Vocab size: {len(vocab)}")

    if args.test_text:
        with open(args.test_text, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = tok.tokenize(text)
        used = set(tokens)
        unk_token = tok.unk_token
        unk_count = tokens.count(unk_token) if unk_token else 0
        pct_vocab_used = (len(used) / len(vocab) * 100.0) if vocab else 0.0
        freq = Counter(tokens)
        tokens_once = [t for t, c in freq.items() if c == 1]
        tokens_le_5 = [t for t, c in freq.items() if c <= 5]
        sample_once = ", ".join(sorted(tokens_once)[:10])

        print("\n=== Token usage on test text ===")
        print(f"File: {args.test_text}")
        print(f"Total tokens: {len(tokens)}")
        print(f"Distinct tokens: {len(used)}")
        print(f"Vocab coverage: {pct_vocab_used:.2f}% of tokenizer vocab")
        if unk_token:
            print(f"Unknown token count ({unk_token}): {unk_count}")
        print("\nLow-frequency tokens (based on this text):")
        print(f"  Tokens appearing exactly once: {len(tokens_once)}")
        if tokens_once:
            print(f"    Sample (up to 10): {sample_once}")
        print(f"  Tokens appearing ≤5 times: {len(tokens_le_5)}")

if __name__ == "__main__":
    main()
