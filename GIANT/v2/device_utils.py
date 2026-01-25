from __future__ import annotations

import jax


def select_default_device() -> jax.Device:
    try:
        gpus = jax.devices("gpu")
        if gpus:
            return gpus[0]
    except RuntimeError:
        pass
    return jax.devices("cpu")[0]
