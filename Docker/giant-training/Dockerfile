
# ---- build s5cmd (static, CGO disabled) --------------------------------------
FROM --platform=$BUILDPLATFORM golang:1.22-alpine AS s5cmd-builder
ARG TARGETOS
ARG TARGETARCH
ARG S5VER=v2.3.0
ENV CGO_ENABLED=0 GOFLAGS="-buildvcs=false"
RUN apk add --no-cache git
RUN GOOS=${TARGETOS} GOARCH=${TARGETARCH} go install -ldflags="-s -w" github.com/peak/s5cmd/v2@${S5VER}

# ---- main image --------------------------------------------------------------
# CUDA 12.9 + cuDNN + Python 3.13.0 + JAX stack
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Base packages + build deps for CPython + SSH daemon
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget curl ca-certificates vim tmux pkg-config \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev             \
        libsqlite3-dev libffi-dev liblzma-dev tk-dev uuid-dev        \
        libncurses5-dev libgdbm-dev libnss3-dev                      \
        openssh-server                                               \
    && rm -rf /var/lib/apt/lists/*

# Build and install Python 3.13.0 from source (keeps system python3 intact)
ARG PYTHON_VERSION=3.13.0
RUN set -eux; \
    cd /tmp; \
    wget "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"; \
    tar -xf "Python-${PYTHON_VERSION}.tgz"; \
    cd "Python-${PYTHON_VERSION}"; \
    ./configure --enable-optimizations --with-lto --prefix=/usr/local; \
    make -j"$(nproc)"; \
    make altinstall; \
    cd /; \
    rm -rf /tmp/Python-*; \
    python3.13 -m ensurepip --upgrade

# Python virtual env (using 3.13.0)
ENV VENV_DIR=/opt/venv
RUN python3.13 -m venv "$VENV_DIR"
# Make venv-default tools used during build
ENV PATH="$VENV_DIR/bin:$PATH" VIRTUAL_ENV="$VENV_DIR"

# Deep-learning stack
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        "jax[cuda12]" \
        flax transformers datasets optax omegaconf kagglehub huggingface-hub

# Node.js + latest npm (pin major; no corepack)
ARG NODE_MAJOR=22
RUN set -eux; \
    curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    npm install -g npm@latest && \
    node -v && npm -v && \
    rm -rf /var/lib/apt/lists/*

# ── SSH server: password-only root login ───────────────────────────────────────
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

# ── Ensure venv is always active (interactive and non-interactive) ────────────
# 1) For non-interactive and SSH-command sessions: set via PAM's /etc/environment
RUN printf '%s\n' \
      'VIRTUAL_ENV=/opt/venv' \
      'PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' \
      >> /etc/environment
# 2) For interactive shells: source the activate script
RUN printf '%s\n' 'source /opt/venv/bin/activate || true' > /etc/profile.d/activate-venv.sh

# ── s5cmd from builder (static binary) ─────────────────────────────────────────
COPY --from=s5cmd-builder /go/bin/s5cmd /usr/local/bin/s5cmd
RUN s5cmd version

EXPOSE 22

# Workspace
WORKDIR /workspace/app
RUN mkdir -p /workspace/app

# Start SSH daemon in the foreground
CMD ["/usr/sbin/sshd","-D"]
