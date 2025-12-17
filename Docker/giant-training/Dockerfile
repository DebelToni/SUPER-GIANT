# ---- build s5cmd (static, CGO disabled) --------------------------------------
FROM --platform=$BUILDPLATFORM golang:1.22-alpine AS s5cmd-builder
ARG TARGETOS
ARG TARGETARCH
ARG S5VER=v2.3.0
ENV CGO_ENABLED=0 GOFLAGS="-buildvcs=false"
RUN apk add --no-cache git
RUN GOOS=${TARGETOS} GOARCH=${TARGETARCH} go install -ldflags="-s -w" github.com/peak/s5cmd/v2@${S5VER}

# ---- main image --------------------------------------------------------------
# CUDA 12.9 + cuDNN + Python 3.13 + JAX stack + SSH + Tailscale
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Base packages + SSH daemon + Python 3.13 from deadsnakes
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gnupg \
        build-essential git wget curl ca-certificates vim tmux pkg-config \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev             \
        libsqlite3-dev libffi-dev liblzma-dev tk-dev uuid-dev        \
        openssh-server iproute2 procps                               \
    && add-apt-repository ppa:deadsnakes/ppa -y                      \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 python3.13-venv python3-pip                       \
    && rm -rf /var/lib/apt/lists/*

# Python virtual env (Python 3.13)
ENV VENV_DIR=/opt/venv
RUN python3.13 -m venv "$VENV_DIR" && \
    "$VENV_DIR/bin/python" -m pip install --upgrade pip
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

# ── SSH server: password-only root login ─────────────────────────────────────
RUN mkdir -p /var/run/sshd && \
    echo 'root:dev' | chpasswd && \
    rm -rf /root/.ssh && mkdir -p /root && \
    printf '%s\n' \
      'PermitRootLogin yes' \
      'PasswordAuthentication yes' \
      'PubkeyAuthentication no' \
      'ChallengeResponseAuthentication no' \
      'UsePAM yes' >> /etc/ssh/sshd_config

# ── Ensure venv is always active (interactive and non-interactive) ──────────
RUN printf '%s\n' \
      'VIRTUAL_ENV=/opt/venv' \
      'PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' \
      >> /etc/environment
RUN printf '%s\n' 'source /opt/venv/bin/activate || true' > /etc/profile.d/activate-venv.sh

# ── s5cmd from builder (static binary) ──────────────────────────────────────
COPY --from=s5cmd-builder /go/bin/s5cmd /usr/local/bin/s5cmd
RUN s5cmd version

# ── Tailscale ──────────────────────────────────────────────────────────────
RUN curl -fsSL https://tailscale.com/install.sh | sh

ENV TS_AUTHKEY="" \
    TS_HOSTNAME="" \
    TS_EXTRA_ARGS="" \
    TS_STATE_DIR="/var/lib/tailscale"

RUN mkdir -p /var/lib/tailscale

COPY --chown=root:root <<'BASH' /usr/local/bin/entrypoint.sh
#!/usr/bin/env bash
set -euo pipefail

mkdir -p /var/run/sshd "$TS_STATE_DIR" /var/run/tailscale

/usr/sbin/tailscaled \
  --state="${TS_STATE_DIR}/tailscaled.state" \
  --socket=/var/run/tailscale/tailscaled.sock \
  &

for i in {1..50}; do
  [[ -S /var/run/tailscale/tailscaled.sock ]] && break
  sleep 0.1
done

if [[ -n "${TS_AUTHKEY:-}" ]]; then
  hn="${TS_HOSTNAME:-gpu-$(hostname)}"
  tailscale up \
    --authkey="${TS_AUTHKEY}" \
    --hostname="${hn}" \
    --accept-dns=false \
    ${TS_EXTRA_ARGS} || true
fi

exec /usr/sbin/sshd -D
BASH

RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 22

WORKDIR /workspace/app
RUN mkdir -p /workspace/app

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []
