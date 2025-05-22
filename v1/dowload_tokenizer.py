# 1. Install the 🤗 datasets library if you haven't already:
#    pip install datasets transformers numpy

from datasets import load_dataset
from transformers import AutoTokenizer

# 2. Load the TinyStories dataset
#    We’ll use the “tatsu-lab/tiny-stories” split “all” which contains all stories.
dataset = load_dataset("roneneldan/TinyStories",
                    cache_dir="/Users/antonhristov/.cache/huggingface/datasets/roneneldan___tiny_stories/default/0.0.0/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64", split="all"
)

# 3. Pull out one story (e.g. the first one) and inspect it
story = dataset[0]["text"]
print("=== Raw story ===")
print(story)

# 4. Load the GPT-Neo-125M tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# 5. Tokenize the story
#    return_tensors="np" gives you a NumPy array of token IDs;
#    you can also use "pt" for PyTorch tensors or omit for Python list.
tokens = tokenizer(story, return_tensors="np")

# 6. Examine the token IDs
input_ids = tokens["input_ids"][0]   # shape is (1, seq_len), so take [0]
print("\n=== Token IDs ===")
print(input_ids)
print(f"Number of tokens: {len(input_ids)}")

# 7. (Optional) Decode back to text to verify round-trip
decoded = tokenizer.decode(input_ids, clean_up_tokenization_spaces=True)
print("\n=== Decoded ===")
print(decoded)

