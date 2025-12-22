"""
Tiny utility to (de)serialize Flax/JAX parameter PyTrees
to a single *.npz* file.
"""

from typing import Dict, Tuple

import numpy as np
import jax
from flax.traverse_util import flatten_dict, unflatten_dict


def _as_numpy(x):
    return np.asarray(x, dtype=x.dtype)


def save_npz(params: Dict, path):
    flat: Dict[Tuple[str, ...], np.ndarray] = flatten_dict(jax.device_get(params))
    np.savez_compressed(path, **{"/".join(k): _as_numpy(v) for k, v in flat.items()})


def load_npz(path) -> Dict:
    flat = {tuple(k.split("/")): v for k, v in np.load(path).items()}
    return unflatten_dict(flat)

