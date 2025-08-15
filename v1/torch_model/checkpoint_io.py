# checkpoint_io.py
"""
Tiny utility to (de)serialize Flax/JAX parameter PyTrees
to a single *.npz* file.  Works for both CPU and GPU tensors.

Why not pickle?
---------------
✓ portable between Python versions  
✓ inspectable with 'np.load' if needed  
✓ no security worries when sharing checkpoints
"""
from typing import Dict, Tuple

import numpy as np
import jax
from flax.traverse_util import flatten_dict, unflatten_dict


def _as_numpy(x):
    return np.asarray(x, dtype=x.dtype)


# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #
def save_npz(params: Dict, path):
    """
    Save *params* (a PyTree/FrozenDict) to **path** with names like
    'Embed_0/embedding', 'Block_3/kv_proj/kernel' …

    """
    flat: Dict[Tuple[str, ...], np.ndarray] = flatten_dict(
        jax.device_get(params)  # bring to host
    )
    np.savez_compressed(
        path,
        **{"/".join(k): _as_numpy(v) for k, v in flat.items()},
    )


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_npz(path) -> Dict:
    """
    Load params back as a *nested* dict of NumPy arrays.
    You can `jax.device_put` afterwards if you like.
    """
    flat = {tuple(k.split("/")): v for k, v in np.load(path).items()}
    return unflatten_dict(flat)

