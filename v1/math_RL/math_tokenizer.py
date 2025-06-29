from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

def build_math_tokenizer():
    pad_token = "[PAD]"
    eos_token = "[EOS]"
    unk_token = "[UNK]"
    number_tokens = [f"{i:03d}" for i in range(122)]
    math_tokens = ["*", "+", "-", "="]

    tokens = [unk_token, pad_token, eos_token] + number_tokens + math_tokens

    vocab = {tok: idx for idx, tok in enumerate(tokens)}

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=unk_token))

    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.post_processor = TemplateProcessing(
        single=f"$A {eos_token}",
        special_tokens=[(eos_token, vocab[eos_token])],
    )

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=unk_token,
        pad_token=pad_token,
        eos_token=eos_token,
    )

    return hf_tok

if __name__ == "__main__":
    tok = build_math_tokenizer()
    print(tok.tokenize("002 + 017 * 002 = 036"))
    print(tok.encode("002 + 017 * 002 = 036"))
    print(tok.tokenize("2 + 17 * 2 = 36"))
    print(tok.encode("2 + 17 * 2 = 36"))
    print(tok.tokenize("002+017*002=036"))
    print(tok.encode("002+017*002=036"))

    tok.save_pretrained("math_tokenizer_data")

