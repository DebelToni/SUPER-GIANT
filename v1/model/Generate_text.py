# def generate_text(params, prompt, max_new_tokens=50, temperature=1.0):
#     input_ids = tokenizer.encode(prompt, return_tensors="np")[0]  # shape [prompt_len]
#     generated = list(input_ids)
#     for _ in range(max_new_tokens):
#         # take last 256 tokens as context if prompt is long (for efficiency)
#         context_ids = np.array([generated[-256:]], dtype=np.int32)
#         logits = model.apply({'params': params}, context_ids, deterministic=True)
#         # take the last position's logits
#         next_token_logits = logits[0, -1, :]  # shape [vocab_size]
#         # Optionally apply temperature
#         if temperature != 1.0:
#             next_token_logits = next_token_logits / temperature
#         # Sample from the distribution
#         next_token_id = int(np.argmax(next_token_logits))  # greedy; or use random sampling
#         # If you want stochastic sampling:
#         # probs = jax.nn.softmax(next_token_logits)
#         # next_token_id = int(np.random.choice(len(probs), p=np.array(probs)))
#         if next_token_id == tokenizer.eos_token_id:
#             break
#         generated.append(next_token_id)
#     return tokenizer.decode(generated)
#

# Generate_text.py
import pickle, jax, jax.numpy as jnp, numpy as np
from Transformer_block import TinyTransformerLM
import Config, sys

def load_everything():
    with open("model_params.pkl", "rb") as f:
        params = pickle.load(f)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    model = TinyTransformerLM(
        vocab_size = Config.vocab_size,
        max_len    = Config.context_length,
        d_model    = Config.embedding_size,
        n_heads    = Config.num_heads,
        d_ff       = Config.feed_forward_size,
        n_layers   = Config.num_layers,
    )
    return params, tokenizer, model

def generate(prompt, max_new_tokens=128, temperature=1.0):
    params, tokenizer, model = load_everything()
    ids = tokenizer.encode(prompt)
    while len(ids) < Config.context_length:
        ctx = np.array([ids[-Config.context_length:]], dtype=np.int32)
        logits = model.apply({"params": params}, ctx, deterministic=True)
        next_logits = logits[0, -1] / temperature
        next_id = int(np.argmax(next_logits))
        if next_id == tokenizer.eos_token_id: break
        ids.append(next_id)
        if len(ids) >= len(prompt)+max_new_tokens: break
    return tokenizer.decode(ids)

if __name__ == "__main__":
    print(generate(sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"))

