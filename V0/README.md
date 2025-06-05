# SUPER GIANT version 0.0.1

This is my initial start of the project. The idea is to train the simplest possible LLM from alwready available examples, to get more familiar with the process of training.

Version 0.0.1 is the simplest possible **working** implementation of the small LLM.

---

Setup:
- JAX
- 10% TinyStories dataset
- gpt-neo-125M tokenizer 

256 embeddings and context length

---

Architecture:
Token + Positional embeddings → [Transformer Block] × 2 → Linear output → Softmax

Transformer Block:
- LayerNorm → Multi-Head Self-Attention → Residual → LayerNorm → MLP → Residual
 

