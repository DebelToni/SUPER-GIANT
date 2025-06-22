# math_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

def build_math_tokenizer():
    # 1. Define special and core tokens
    pad_token = "[PAD]"
    eos_token = "[EOS]"
    unk_token = "[UNK]"                  # required by WordLevel model
    # Numeric tokens 000–121
    number_tokens = [f"{i:03d}" for i in range(122)]
    # Basic math operators
    math_tokens = ["*", "+", "-", "="]

    # Assemble full vocab list (specials first)
    tokens = [unk_token, pad_token, eos_token] + number_tokens + math_tokens

    # Map tokens to integer IDs
    vocab = {tok: idx for idx, tok in enumerate(tokens)}

    # 2. Instantiate a WordLevel tokenizer with our vocab
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=unk_token))

    # 3. Simple whitespace splitting
    tokenizer.pre_tokenizer = Whitespace()

    # 4. Append [EOS] after every sequence
    tokenizer.post_processor = TemplateProcessing(
        single=f"$A {eos_token}",
        special_tokens=[(eos_token, vocab[eos_token])],
    )

    # 5. Wrap in a HuggingFace PreTrainedTokenizerFast for full compatibility
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=unk_token,
        pad_token=pad_token,
        eos_token=eos_token,
    )

    return hf_tok

if __name__ == "__main__":
    tok = build_math_tokenizer()
    # Example usage
    # print(tok.encode("002 + 017 = 019").tokens)
    print(tok.tokenize("002 + 017 = 019"))
    print(tok.encode("002 + 017 = 019"))
    for token in tok.get_vocab():
        print(f"{token}: {tok.get_vocab()[token]}")
    print("Pad token ID:", tok.pad_token_id)
    print("EOS token ID:", tok.eos_token_id)

    # Save to disk for later loading via .from_pretrained()
    tok.save_pretrained("math_tokenizer_data")

