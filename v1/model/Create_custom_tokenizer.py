from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", use_fast=True)

specials = {}
if tok.pad_token is None:
    specials["pad_token"] = "<|pad|>"
if tok.bos_token is None:
    specials["bos_token"] = "<|bos|>"
if specials:
    tok.add_special_tokens(specials)

tok.add_bos_token = True      # make .encode() insert it
tok.add_eos_token = True      # optional, GPT‑Neo already has one
tok.save_pretrained("neo-english-cust")

print("pad :", tok.pad_token, tok.pad_token_id)
print("bos :", tok.bos_token, tok.bos_token_id)
print("eos :", tok.eos_token, tok.eos_token_id)
print("vocab size :", tok.vocab_size)
