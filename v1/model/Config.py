dtype = "float32"  # JAX default dtype
# context_length  = 512
# num_heads       = 4
# num_layers      = 4
embedding_size  = 256
# embedding_size  = 128 
context_length  = 256
num_heads       = 2 
num_layers      = 2
feed_forward_size = num_layers * embedding_size

from transformers import AutoTokenizer
vocab_size      = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M").vocab_size

learning_rate   = 2e-4
weight_decay    = 1e-2
batch_size      = 8 
num_epochs      = 1
use_remat       = True         # true means save memory
acc_steps = 2

dropout_rate    = 0.1

# DTYPE_ACT  = jnp.float16          
# DTYPE_NORM = jnp.float32         

# dataset_percent = 30
dataset_percent = 100
chunk_percent = 10 # how much to save on disk
