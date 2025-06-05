# SUPER GIANT version 0.1.0

Architecture:
Token + Positional embeddings → [Transformer Block] × N → Linear output → Softmax

Transformer Block:
- LayerNorm → Multi-Head Self-Attention (Flash Attention 2, JAX cuDNN, Ampere optimized) → Residual → LayerNorm → MLP → Residual
- KV cache
 

