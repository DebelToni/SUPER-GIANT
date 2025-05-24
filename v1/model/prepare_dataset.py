from datasets      import load_dataset
from transformers  import AutoTokenizer
import numpy as np, Config

_TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"

def get_data():
    print("Entering get_data()")
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
    dataset   = load_dataset("roneneldan/TinyStories", split="all")
    print(f"Loaded dataset with {len(dataset)} examples")

    dataset = dataset.select(range(int(1 * len(dataset))))  # use only 10% for testing

    # Encode every story, then chunk into context_length tokens
    ctx = Config.context_length
    sequences = []
    for ex in dataset["text"]:
        # print(f"Encoding story: {ex[:50]}...")  # print first 50 chars for context
        ids = tokenizer.encode(ex, add_special_tokens=False)
        # pad or chunk to fixed length
        # print(f"Encoding story of length {len(ids)} tokens")
        for i in range(0, len(ids), ctx):
            chunk = ids[i:i+ctx]
            if len(chunk) < ctx:
                chunk += [tokenizer.eos_token_id]*(ctx-len(chunk))
            sequences.append(chunk)

    print(f"Encoded {len(sequences)} sequences of length {ctx} tokens")

    sequences = np.array(sequences, dtype=np.int32)
    np.random.shuffle(sequences)
    split = int(0.97*len(sequences))     # 97 % train / 3 % val
    return sequences[:split], sequences[split:], tokenizer

# from datasets import load_dataset
# from transformers import AutoTokenizer
# import numpy as np
# import Config
# import os
#
# _TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"
#
# def get_data():
#     # Force HuggingFace datasets cache to /content/hf-cache (local, not Google Drive)
#     os.environ["HF_DATASETS_CACHE"] = "/content/hf-cache"
#     os.environ["TRANSFORMERS_CACHE"] = "/content/hf-cache"
#     tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
#
#     # Use explicit train/validation splits, not 'all'
#     train_ds = load_dataset("roneneldan/TinyStories", split="train")
#     val_ds   = load_dataset("roneneldan/TinyStories", split="validation")
#
#     ctx = Config.context_length
#
#     def encode_split(ds):
#         sequences = []
#         for ex in ds["text"]:
#             ids = tokenizer.encode(ex, add_special_tokens=False)
#             for i in range(0, len(ids), ctx):
#                 chunk = ids[i:i+ctx]
#                 if len(chunk) < ctx:
#                     chunk += [tokenizer.eos_token_id]*(ctx-len(chunk))
#                 sequences.append(chunk)
#         return np.array(sequences, dtype=np.int32)
#
#     train_tokens = encode_split(train_ds)
#     val_tokens   = encode_split(val_ds)
#     return train_tokens, val_tokens, tokenizer
