# SUPER GIANT version 0.1.0

This is my initial start of the project. The idea is to train the simplest possible LLM from alwready available examples, to get more familiar with the process of training.

---

Setup:
- JAX
- TinyStories
- gpt-neo-125M tokenizer 
- 512 tokens context length

---

Architecture:
Token + Positional embeddings → [Transformer Block] × N → Linear output → Softmax

Pseudo code:
```python
tokens = [t1, t2, ..., tN]            # input token IDs
embeddings = EmbeddingLookup(tokens)  # shape [N, d_model]
pos_embs = PositionalEmbeddings[0:N]  # shape [N, d_model]
h = embeddings + pos_embs            # add position information

for block in range(N):
    # Self-attention sub-layer
    attn_out = SelfAttentionMask(h)   # attends to past positions
    attn_out = Dropout(attn_out)
    h = LayerNorm(h + attn_out)       # add & normalize (residual)
    # Feed-forward sub-layer
    ff_out = FeedForward(h)           # typically Linear->ReLU->Linear
    ff_out = Dropout(ff_out)
    h = LayerNorm(h + ff_out)         # add & normalize (residual)

# Now h is the final hidden states for each position
logits = Dense(h)  # project each h (d_model) to vocab size V
# (If tied embeddings, Dense uses Embedding matrix transpose)
probs = softmax(logits)  # convert to probabilities
```
