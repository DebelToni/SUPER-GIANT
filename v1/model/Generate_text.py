def generate_text(params, prompt, max_new_tokens=50, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors="np")[0]  # shape [prompt_len]
    generated = list(input_ids)
    for _ in range(max_new_tokens):
        # take last 256 tokens as context if prompt is long (for efficiency)
        context_ids = np.array([generated[-256:]], dtype=np.int32)
        logits = model.apply({'params': params}, context_ids, deterministic=True)
        # take the last position's logits
        next_token_logits = logits[0, -1, :]  # shape [vocab_size]
        # Optionally apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        # Sample from the distribution
        next_token_id = int(np.argmax(next_token_logits))  # greedy; or use random sampling
        # If you want stochastic sampling:
        # probs = jax.nn.softmax(next_token_logits)
        # next_token_id = int(np.random.choice(len(probs), p=np.array(probs)))
        if next_token_id == tokenizer.eos_token_id:
            break
        generated.append(next_token_id)
    return tokenizer.decode(generated)

