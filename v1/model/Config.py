# Config.py

import jax.numpy as jnp
from transformers import AutoTokenizer

# -----------------------------
dtype = jnp.bfloat16
compute_dtype = dtype
param_dtype = jnp.float32
# -----------------------------

# Model Hyperparameters
embedding_size = 256
context_length = 256
num_heads = 2
num_layers = 2
feed_forward_size = num_layers * embedding_size
vocab_size = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M").vocab_size
dropout_rate = 0.1

# Training Hyperparameters
learning_rate = 2e-4
weight_decay = 1e-2
batch_size = 8
num_epochs = 1
acc_steps = 2

# Other Settings
use_remat = True
dataset_percent = 10
chunk_percent = 10

deafult_device = "cuda"
