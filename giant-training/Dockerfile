FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget curl ca-certificates vim pkg-config \
        python3.11 python3.11-venv python3-pip \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev libffi-dev liblzma-dev tk-dev uuid-dev \
    && rm -rf /var/lib/apt/lists/*

ENV VENV_DIR=/opt/venv
RUN python3.11 -m venv $VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH" \
    VIRTUAL_ENV=$VENV_DIR

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      jax[cuda12] \
      flax \
      transformers \
      datasets \
      optax

WORKDIR /workspace/app
RUN mkdir -p /workspace/app
CMD ["tail", "-f", "/dev/null"]

# FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04
# # py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
#
# ENV DEBIAN_FRONTEND=noninteractive
#
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         build-essential git wget curl ca-certificates vim pkg-config \
#         python3.11 python3.11-venv python3-pip \
#         libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
#         libsqlite3-dev libffi-dev liblzma-dev tk-dev uuid-dev \
#     && rm -rf /var/lib/apt/lists/*
#
# ENV VENV_DIR=/opt/venv
# RUN python3.11 -m venv $VENV_DIR
# ENV PATH="$VENV_DIR/bin:$PATH" \
#     VIRTUAL_ENV=$VENV_DIR
#
# # ENV XLA_FLAGS="--xla_gpu_enable_cublaslt=false"
#
# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir \
#       "jax[cuda12]" \
#       flax \
#       transformers \
#       datasets \
# 	  optax
#
# WORKDIR /workspace/app
# RUN mkdir -p /workspace/app
# # CMD ["bash"]
# CMD ["tail", "-f", "/dev/null"]
