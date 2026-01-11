import sys
import argparse
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Compare vocab overlap between two Hugging Face tokenizers."
    )
    parser.add_argument("tok1", nargs="?", default="bert-base-uncased",
                        help="First tokenizer name (default: bert-base-uncased)")
    parser.add_argument("tok2", nargs="?", default="roberta-base",
                        help="Second tokenizer name (default: roberta-base)")
    parser.add_argument("--test-text", "-t", type=str,
                        help="Path to a text file to test token overlap on actual text.")
    args = parser.parse_args()

    tok1 = AutoTokenizer.from_pretrained(args.tok1)
    tok2 = AutoTokenizer.from_pretrained(args.tok2)

    vocab1 = set(tok1.get_vocab().keys())
    vocab2 = set(tok2.get_vocab().keys())

    overlap_vocab = vocab1 & vocab2
    n_overlap_vocab = len(overlap_vocab)

    pct1_vocab = 100.0 * n_overlap_vocab / len(vocab1)
    pct2_vocab = 100.0 * n_overlap_vocab / len(vocab2)

    print(f"Tokenizer 1: {args.tok1}")
    print(f"  Vocab size: {len(vocab1)}")
    print(f"Tokenizer 2: {args.tok2}")
    print(f"  Vocab size: {len(vocab2)}")
    print(f"Shared vocab tokens: {n_overlap_vocab}")
    print(f"Overlap as % of {args.tok1}: {pct1_vocab:.2f}%")
    print(f"Overlap as % of {args.tok2}: {pct2_vocab:.2f}%")

    if args.test_text:
        with open(args.test_text, "r", encoding="utf-8") as f:
            text = f.read()

        # tokenize into *string tokens* and compare exact matches
        tokens1 = tok1.tokenize(text)
        tokens2 = tok2.tokenize(text)

        used1 = set(tokens1)
        used2 = set(tokens2)
        overlap_used = used1 & used2

        n_used1 = len(used1)
        n_used2 = len(used2)
        n_overlap_used = len(overlap_used)

        pct1_used = 100.0 * n_overlap_used / n_used1 if n_used1 else 0.0
        pct2_used = 100.0 * n_overlap_used / n_used2 if n_used2 else 0.0

        print("\n=== Overlap on test text ===")
        print(f"File: {args.test_text}")
        print(f"Distinct tokens in text ({args.tok1}): {n_used1}")
        print(f"Distinct tokens in text ({args.tok2}): {n_used2}")
        print(f"Shared distinct tokens in text: {n_overlap_used}")
        print(f"Shared-as-% of {args.tok1} text tokens: {pct1_used:.2f}%")
        print(f"Shared-as-% of {args.tok2} text tokens: {pct2_used:.2f}%")

if __name__ == "__main__":
    main()

