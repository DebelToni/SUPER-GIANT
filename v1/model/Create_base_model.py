import Config as Config
import Transformer_Block as Transformer

import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import jax
import jax.numpy as jnp

import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
batch_size = 1
seq_length = 256
dummy_input = jnp.zeros((batch_size, seq_length), dtype=jnp.int32)  # a batch of zeros
model = Transformer.TinyTransformerLM(
    vocab_size=Config.vocab_size,
    max_len=Config.context_length,
    d_model=Config.embedding_size,
    n_heads=Config.num_heads,
    d_ff=Config.feed_forward_size,
    n_layers=Config.num_layers
)
params = model.init(key, dummy_input, deterministic=True)['params']

# print(params)

# from flax.serialization import to_bytes, from_bytes
#
# params_bytes = to_bytes(params)
#
# with open('model_params.msgpack', 'wb') as f:
#     f.write(params_bytes)

""" Importing the model
from flax.serialization import from_bytes

# Load from a file
with open('model_params.msgpack', 'rb') as f:
    param_bytes = f.read()

# Deserialize parameters
params = from_bytes(model.init(key, dummy_input), param_bytes)
"""

import optax
optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-2)
opt_state = optimizer.init(params)

print("Optimizer:", opt_state)



