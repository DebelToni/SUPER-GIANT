# SUPER GIANT version 0.1.0

This is my initial start of the project. The idea is to train the simplest possible LLM from alwready available examples, to get more familiar with the process of training.

---

Setup:
- JAX
- TinyStories
- gpt-neo-125M tokenizer 
<!-- - 512 tokens context length -->
config

---

Architecture:
Token + Positional embeddings → [Transformer Block] × N → Linear output → Softmax

Transformer Block:
- LayerNorm → Multi-Head Self-Attention (Flash Attention 2, JAX cuDNN, Ampere optimized) → Residual → LayerNorm → MLP → Residual
 

