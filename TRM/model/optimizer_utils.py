from typing import Iterable

from flax import traverse_util
from flax.core import freeze


def create_weight_decay_mask(params, exclusions: Iterable[str]):
    """Return a mask tree where True applies weight decay."""
    exclusion_set = {str(name).lower() for name in exclusions}
    flat_params = traverse_util.flatten_dict(params)
    flat_mask = {}
    for path in flat_params:
        last = path[-1].lower()
        flat_mask[path] = last not in exclusion_set
    return freeze(traverse_util.unflatten_dict(flat_mask))

