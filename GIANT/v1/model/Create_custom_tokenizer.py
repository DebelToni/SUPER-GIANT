from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", use_fast=True)

specials = {}
if tok.pad_token is None:
    specials["pad_token"] = "<|pad|>"
if tok.bos_token_id == tok.eos_token_id:
    specials.setdefault("additional_special_tokens", []).append("<|bos|>")
if specials:
    tok.add_special_tokens(specials)
if "<|bos|>" in tok.get_vocab():
    tok.bos_token = "<|bos|>"

tok._tokenizer.post_processor = TemplateProcessing(
    single="<|bos|> $A <|endoftext|>",
    pair="<|bos|> $A <|endoftext|> $B:1 <|endoftext|>:1",
    special_tokens=[
        ("<|bos|>", tok.bos_token_id),
        ("<|endoftext|>", tok.eos_token_id),
    ],
)

tok.save_pretrained("neo-english-cust")

print("PAD token:", tok.pad_token, "ID:", tok.pad_token_id)
print("BOS token:", tok.bos_token, "ID:", tok.bos_token_id)
print("EOS token:", tok.eos_token, "ID:", tok.eos_token_id)
print("Tokenizer size:", len(tok))


tok = AutoTokenizer.from_pretrained("neo-english-cust", use_fast=True)

ids = tok.encode("Hello world!     ")
print(ids[:3], "...", ids[-3:])
print(ids)
decoded = tok.decode(ids)
print("Decoded text:", decoded)
assert ids[0]  == tok.bos_token_id
assert ids[-1] == tok.eos_token_id


tok = AutoTokenizer.from_pretrained("neo-english-cust", use_fast=True)

encoded = tok(
    "Hello world!",                 # any short text
    padding="max_length",
    max_length=16,                  # force extra room so you *see* the pad ids
    return_tensors="np"             # or "pt", "jax"
)

print("input_ids :", encoded["input_ids"][0])
print("attn_mask :", encoded["attention_mask"][0])

# Assertions
assert encoded["input_ids"][0, 0]  == tok.bos_token_id      # <|bos|>
assert encoded["input_ids"][0, -1] == tok.pad_token_id      # <|pad|>
assert encoded["attention_mask"][0, -1] == 0                # mask is 0

