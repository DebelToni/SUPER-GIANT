# SUPER GIANT version 0.1.0

> Historical version. Current active GIANT work is in [GIANT/v3](../v3/).

Architecture:
Token + Positional embeddings → [Transformer Block] × N → Linear output → Softmax

Transformer Block:
- LayerNorm → Multi-Head Self-Attention (Flash Attention 2, JAX cuDNN, Ampere optimized) → Residual → LayerNorm → MLP → Residual
- KV cache
 

