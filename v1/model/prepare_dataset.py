from datasets      import load_dataset
from transformers  import AutoTokenizer
import numpy as np, Config

_TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"

def get_data():
    print("Entering get_data()")
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
    dataset   = load_dataset("roneneldan/TinyStories", split="all")
    print(f"Loaded dataset with {len(dataset)} examples")

    # Encode every story, then chunk into context_length tokens
    ctx = Config.context_length
    sequences = []
    for ex in dataset["text"]:
        ids = tokenizer.encode(ex, add_special_tokens=False)
        # pad or chunk to fixed length
        print(f"Encoding story of length {len(ids)} tokens")
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

