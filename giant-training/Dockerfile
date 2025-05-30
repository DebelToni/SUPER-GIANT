# ---------- base: Ubuntu 22.04 + CUDA 12 + cuDNN ----------
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

# --- 1. System dependencies ---------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget curl ca-certificates \
        vim pkg-config \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev libffi-dev liblzma-dev tk-dev uuid-dev \
    && rm -rf /var/lib/apt/lists/*

# --- 2. Build the latest CPython from source ----------------------------------
ARG PYTHON_VERSION=3.12.3
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j"$(nproc)" && make altinstall && \
    cd / && rm -rf /tmp/Python-${PYTHON_VERSION}*      # keep image lean

ENV PATH="/usr/local/bin:${PATH}"

# --- 3. Lightweight virtual-env with JAX (CUDA 12) stack ----------------------
ENV VENV_DIR=/opt/venv
RUN python3.12 -m venv ${VENV_DIR} && \
    ${VENV_DIR}/bin/pip install --upgrade pip && \
    ${VENV_DIR}/bin/pip install --no-cache-dir \
        "jax[cuda12]" jaxlib flax transformers databases datasets

ENV VIRTUAL_ENV=${VENV_DIR}
ENV PATH="${VENV_DIR}/bin:${PATH}"

# --- 4. Prepare a workspace for manual git clones ---------------------------
WORKDIR /workspace/app
RUN mkdir -p /workspace/app

# By default, drop you into a bash shell in /workspace/app
# CMD ["bash"]
CMD ["tail", "-f", "/dev/null"]
