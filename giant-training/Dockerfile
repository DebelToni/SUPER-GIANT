# CUDA 12.9 + cuDNN + Python 3.11 + JAX stack
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

# Python virtual env
ENV VENV_DIR=/opt/venv
RUN python3.11 -m venv "$VENV_DIR"
# Make venv-default tools used during build
ENV PATH="$VENV_DIR/bin:$PATH" VIRTUAL_ENV="$VENV_DIR"

# Deep-learning stack
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        "jax[cuda12]" \
        flax transformers datasets optax omegaconf

# Install latest Node.js (includes npm) via NodeSource
RUN curl -fsSL https://deb.nodesource.com/setup_current.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Ensure **latest** npm
RUN npm install -g npm@latest

# ── SSH server: password-only root login ────────────────────────────────────────
RUN mkdir -p /var/run/sshd && \
    # Set root password
    echo 'root:dev' | chpasswd && \
    # Remove any keys and disable key auth
    rm -rf /root/.ssh && mkdir -p /root && \
    printf '%s\n' \
      'PermitRootLogin yes' \
      'PasswordAuthentication yes' \
      'PubkeyAuthentication no' \
      'ChallengeResponseAuthentication no' \
      'UsePAM yes' >> /etc/ssh/sshd_config

# ── Ensure venv is always active (interactive and non-interactive) ─────────────
# 1) For non-interactive and SSH-command sessions: set via PAM's /etc/environment
RUN printf '%s\n' \
      'VIRTUAL_ENV=/opt/venv' \
      'PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' \
      >> /etc/environment
# 2) For interactive shells: source the activate scrip
RUN printf '%s\n' 'source /opt/venv/bin/activate || true' > /etc/profile.d/activate-venv.sh

EXPOSE 22

# Workspace
WORKDIR /workspace/app
RUN mkdir -p /workspace/app

# Start SSH daemon in the foreground 
CMD ["/usr/sbin/sshd","-D"]

