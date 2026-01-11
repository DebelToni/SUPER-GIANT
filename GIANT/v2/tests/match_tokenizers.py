from transformers import AutoTokenizer
import sys

# Usage:
#   python compare_vocab.py bert-base-uncased roberta-base
# If no args are given, defaults are used.
name1 = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"
name2 = sys.argv[2] if len(sys.argv) > 2 else "roberta-base"

tok1 = AutoTokenizer.from_pretrained(name1)
tok2 = AutoTokenizer.from_pretrained(name2)

vocab1 = set(tok1.get_vocab().keys())
vocab2 = set(tok2.get_vocab().keys())

overlap = vocab1 & vocab2
n_overlap = len(overlap)

pct1 = 100.0 * n_overlap / len(vocab1)
pct2 = 100.0 * n_overlap / len(vocab2)

print(f"Tokenizer 1: {name1}")
print(f"  Vocab size: {len(vocab1)}")
print(f"Tokenizer 2: {name2}")
print(f"  Vocab size: {len(vocab2)}")
print(f"Shared tokens: {n_overlap}")
print(f"Overlap as % of {name1}: {pct1:.2f}%")
print(f"Overlap as % of {name2}: {pct2:.2f}%")

