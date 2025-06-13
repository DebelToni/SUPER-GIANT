
# Config.py

import jax.numpy as jnp
from transformers import AutoTokenizer

# -----------------------------
dtype = jnp.bfloat16
compute_dtype = jnp.bfloat16
param_dtype = jnp.float32
# -----------------------------

# Model Hyperparameters
embedding_size = 256 # 768
context_length = 257 # 1024
num_heads =  2 # 12
num_layers = 2 # 12
feed_forward_size = 4 * embedding_size # 4 is standard in transformer models
tokenizer_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
dropout_rate = 0.1

vocab_size = tokenizer.vocab_size

# Training Hyperparameters
learning_rate = 2e-4 # 1e-4
weight_decay = 1e-2
batch_size = 8 # 16
num_epochs = 1 # 5
acc_steps = 2 # 4

# Other Settings
use_remat = True
dataset_percent = 10
chunk_percent = 10

deafult_device = "cuda"

