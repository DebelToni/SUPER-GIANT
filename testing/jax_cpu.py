import jax
from jax import numpy as jnp

# Force CPU backend
jax.config.update("jax_platform_name", "cpu")

# Example operation
x = jnp.array([1.0, 2.0, 3.0])
print(jax.devices())  # should show only CPU device

