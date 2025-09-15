from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", use_fast=True)

print("vocab size :", tok.vocab_size)
print("eos_token  :", tok.eos_token, tok.eos_token_id)
print("bos_token  :", tok.bos_token, tok.bos_token_id)
print("pad_token  :", tok.pad_token, tok.pad_token_id)
print(tok.special_tokens_map)

