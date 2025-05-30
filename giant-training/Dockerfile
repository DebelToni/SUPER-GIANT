# ---------- base: Ubuntu 22.04 + CUDA 12.9 + cuDNN 8-devel ----------
FROM nvidia/cuda:12.9.0-cudnn8-devel-ubuntu22.04

# noninteractive apt
ENV DEBIAN_FRONTEND=noninteractive

# ─── 1. System deps + Python 3.11 ────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget curl ca-certificates vim pkg-config \
        python3.11 python3.11-venv python3-pip \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev libffi-dev liblzma-dev tk-dev uuid-dev \
    && rm -rf /var/lib/apt/lists/*

# ─── 2. Create & activate venv ───────────────────────────────────────
ENV VENV_DIR=/opt/venv
RUN python3.11 -m venv $VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH" \
    VIRTUAL_ENV=$VENV_DIR

# ─── 3. Disable cuBLAS-Lt BF16-only kernels (optional but recommended) ─
#    This avoids “unsupported parameter” errors on non-A100 GPUs.
ENV XLA_FLAGS="--xla_gpu_enable_cublaslt=false"

# ─── 4. Python deps (single jax install) ─────────────────────────────
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      "jax[cuda12]>=0.4.17" \
      flax \
      transformers \
      datasets

# ─── 5. Workspace & default CMD ─────────────────────────────────────
WORKDIR /workspace/app
RUN mkdir -p /workspace/app
CMD ["bash"]

