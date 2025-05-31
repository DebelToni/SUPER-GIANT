# Custom Dockerfile

This dockerfile is designed to build a custom Docker image for faster deployment for both inferencing and training.


Contents:
- ubuntu 22.04
- cuda12.4.1
- cuDNN wheels
- Python 3.11

Pip packages:
- jax[cuda12]
- flax
- transfromers
- datasets
(venv always activated)
