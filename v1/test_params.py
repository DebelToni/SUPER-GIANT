import pickle, jax.numpy as jnp
tree = pickle.load(open("model_params.pkl", "rb"))

# Print top-level keys to explore model structure
print("Top-level keys:", tree.keys())
# Print all top-level keys for inspection
for key in tree.keys():
    print(f"Top-level key: {key}")
# Try to print keys of the first block if it exists
first_block = next(iter(tree.keys()))
print(f"Keys in {first_block}:", tree[first_block].keys())
print("mean of first block kernel", jnp.mean(tree[first_block]["kernel"]))
