# ────────────────────────────────────────────────────────────────────────────────
# CUDA 12.9 + cuDNN + Python 3.11 + JAX stack
# with OpenSSH server pre-configured for root / key-only login
# ────────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Base packages + SSH daemon
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget curl ca-certificates vim pkg-config \
        python3.11 python3.11-venv python3-pip                       \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev             \
        libsqlite3-dev libffi-dev liblzma-dev tk-dev uuid-dev        \
        openssh-server                                               \
    && rm -rf /var/lib/apt/lists/*

# Python virtual-env
ENV VENV_DIR=/opt/venv
RUN python3.11 -m venv $VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH" \
    VIRTUAL_ENV=$VENV_DIR

# Deep-learning stack
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        "jax[cuda12]" \
        flax transformers datasets optax

# ── SSH server configuration ───────────────────────────────────────────────────
#  * runs as root          (PermitRootLogin yes)
#  * key-based only        (PasswordAuthentication no)
#  * empty /root/.ssh that you’ll fill via a bind-mount or later COPY
RUN mkdir /var/run/sshd && \
    mkdir -p /root/.ssh && chmod 700 /root/.ssh && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/'  /etc/ssh/sshd_config

# If you want to bake your key straight into the image, uncomment:
# COPY id_rsa.pub /root/.ssh/authorized_keys
# RUN chmod 600 /root/.ssh/authorized_keys

EXPOSE 22

# Workspace
WORKDIR /workspace/app
RUN mkdir -p /workspace/app

# Start SSH daemon in the foreground
CMD ["/usr/sbin/sshd","-D"]

