# Based on your original script (the one you pasted above).
from pathlib import Path

from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing
from omegaconf import OmegaConf

DATA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_DIR.parent

cfg = OmegaConf.merge(
    OmegaConf.load(PROJECT_ROOT / "Global_Config.yml"),
    OmegaConf.load(DATA_DIR / "Config.yml"),
)

teacher_cfg = cfg.teacher if "teacher" in cfg else cfg
tokenizer_id = getattr(teacher_cfg, "model_tokenizer", cfg.tokenizer.name)

# Load the base tokenizer
tok = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)

# Prepare special tokens to add.
# We keep your existing logic for pad / optional <|bos|> and also add [USER] and [AI].
specials = {}

# Add pad token if missing
if tok.pad_token is None:
    specials["pad_token"] = "<|pad|>"

# If tokenizer uses same id for bos/eos (common for some GPT tokenizers),
# add an explicit <|bos|> token (you had this logic before).
if tok.bos_token_id == tok.eos_token_id:
    specials.setdefault("additional_special_tokens", []).append("<|bos|>")

# Add the two chat markers you asked for
# (you can change the strings "[USER]" / "[AI]" to any other token text if desired)
specials.setdefault("additional_special_tokens", []).extend(["[USER]", "[AI]"])

# Add the collected special tokens to tokenizer (this resizes the vocab)
if specials:
    tok.add_special_tokens(specials)

# If we added a literal "<|bos|>" token into vocab, set tok.bos_token to that string
if "<|bos|>" in tok.get_vocab():
    tok.bos_token = "<|bos|>"

# Now build a post-processor template that inserts the new tokens.
# We map each special token string to its id so the TemplateProcessing knows them.
# Fetch the ids (convert_tokens_to_ids returns int id for the token string)
bos_id = tok.bos_token_id
eos_id = tok.eos_token_id
user_id = tok.convert_tokens_to_ids("[USER]")
ai_id   = tok.convert_tokens_to_ids("[AI]")

# Sanity: if any of these are None or -1, something went wrong
assert bos_id is not None and eos_id is not None
assert user_id not in (None, -1) and ai_id not in (None, -1), "Failed to add [USER] or [AI] tokens"

tok._tokenizer.post_processor = TemplateProcessing(
    # single sequence: <|bos|> [USER] <text> <|endoftext|>
    single="<|bos|> [USER] $A <|endoftext|>",
    # pair sequence: <|bos|> [USER] <A> <|endoftext|> [AI] <B> <|endoftext|>
    pair="<|bos|> [USER] $A <|endoftext|> [AI] $B:1 <|endoftext|>:1",
    special_tokens=[
        ("<|bos|>", bos_id),
        ("<|endoftext|>", eos_id),
        ("[USER]", user_id),
        ("[AI]", ai_id),
    ],
)

# Save and reload (like your original flow)
tok.save_pretrained("neo-english-cust")
print("Saved tokenizer to 'neo-english-cust'")

# Show the tokens and IDs for verification
print("PAD token:", tok.pad_token, "ID:", tok.pad_token_id)
print("BOS token:", tok.bos_token, "ID:", tok.bos_token_id)
print("EOS token:", tok.eos_token, "ID:", tok.eos_token_id)
print("[USER] ID:", user_id)
print("[AI]   ID:", ai_id)
print("Tokenizer size:", len(tok))

# Reload to ensure saved config is correct
tok = AutoTokenizer.from_pretrained("neo-english-cust", use_fast=True)

# Quick encode test (single)
ids = tok.encode("Hello world!     ")
print("Encoded ids (single) sample:", ids[:6], "...", ids[-6:])
decoded = tok.decode(ids)
print("Decoded text:", decoded)

# Verify the post-processor placed <|bos|> then [USER] at the start, and <|endoftext|> at the end
assert ids[0]  == tok.bos_token_id, "First token should be BOS"
# after reloading, fetch the IDs again
user_id_reload = tok.convert_tokens_to_ids("[USER]")
eos_id_reload  = tok.eos_token_id
assert ids[1] == user_id_reload, "Second token should be [USER]"
assert ids[-1] == eos_id_reload, "Last token should be EOS (<|endoftext|>)"

# Padding example (same as your original)
encoded = tok(
    "Hello world!",                 # any short text
    padding="max_length",
    max_length=16,                  # force extra room so you *see* the pad ids
    return_tensors="np"             # or "pt", "jax"
)

print("input_ids :", encoded["input_ids"][0])
print("attn_mask :", encoded["attention_mask"][0])

# Assertions similar to your original ones but now also check the [USER] token
assert encoded["input_ids"][0, 0]  == tok.bos_token_id      # <|bos|>
assert encoded["input_ids"][0, 1]  == tok.convert_tokens_to_ids("[USER]")  # [USER]
assert encoded["input_ids"][0, -1] == tok.pad_token_id      # <|pad|>
assert encoded["attention_mask"][0, -1] == 0                # mask is 0

print("All checks passed. [USER] and [AI] tokens are added and used in the post-processor.")
