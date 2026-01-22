# ---- build s5cmd (static, CGO disabled) --------------------------------------
FROM --platform=$BUILDPLATFORM golang:1.22-alpine AS s5cmd-builder
ARG TARGETOS
ARG TARGETARCH
ARG S5VER=v2.3.0
ENV CGO_ENABLED=0 GOFLAGS="-buildvcs=false"
RUN apk add --no-cache git
RUN GOOS=${TARGETOS} GOARCH=${TARGETARCH} go install -ldflags="-s -w" github.com/peak/s5cmd/v2@${S5VER}

# ---- main image --------------------------------------------------------------
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Base packages + Python 3.13 from deadsnakes (NO openssh-server)
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gnupg \
        build-essential git wget curl ca-certificates vim tmux pkg-config \
		ripgrep \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev libffi-dev liblzma-dev tk-dev uuid-dev \
        iproute2 procps \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 python3.13-venv python3-pip \
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

# Ensure venv is always active (interactive and non-interactive)
RUN printf '%s\n' \
      'VIRTUAL_ENV=/opt/venv' \
      'PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' \
      >> /etc/environment
RUN printf '%s\n' 'source /opt/venv/bin/activate || true' > /etc/profile.d/activate-venv.sh
RUN printf '%s\n' 'set -o vi' > /etc/profile.d/vi-mode.sh

# s5cmd from builder (static binary)
COPY --from=s5cmd-builder /go/bin/s5cmd /usr/local/bin/s5cmd
RUN s5cmd version

# Tailscale
RUN curl -fsSL https://tailscale.com/install.sh | sh

# Defaults (can be overridden at runtime)
ENV TS_AUTHKEY="" \
    TS_HOSTNAME="" \
    TS_TAGS="tag:gpu" \
    TS_STATE_DIR="/var/lib/tailscale" \
    TS_ACCEPT_DNS="false" \
    TS_ENABLE_SSH="true" \
    TS_EXTRA_ARGS=""

RUN mkdir -p /var/lib/tailscale

COPY --chown=root:root <<'BASH' /usr/local/bin/entrypoint.sh
#!/usr/bin/env bash
set -euo pipefail

APP_ROOT=/proj
DATA_DIR="${APP_ROOT}/giant-data"
REPO_DIR="${APP_ROOT}/SUPER-GIANT"
REPO_URL="https://github.com/DebelToni/SUPER-GIANT"

# mkdir -p "$DATA_DIR"
# if [[ ! -d "$REPO_DIR" ]]; then
#   git clone "$REPO_URL" "$REPO_DIR"
# fi

mkdir -p "$DATA_DIR"

SYNC_MARKER="${DATA_DIR}/.r2_sync_done"
if [[ ! -f "$SYNC_MARKER" ]]; then
  s5cmd sync --size-only "s3://giant-data/*" "$DATA_DIR/"
  touch "$SYNC_MARKER"
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  # Fresh clone (also handles "dir missing" or "dir exists but not a git repo")
  rm -rf "$REPO_DIR" 2>/dev/null || true
  git clone "$REPO_URL" "$REPO_DIR"
else
  # Repo exists: update it
  git -C "$REPO_DIR" fetch --all --prune

  # If you want to keep local changes, use pull --rebase:
  git -C "$REPO_DIR" pull --rebase --autostash

  # If you prefer "always exactly match remote" (discard local changes), use instead:
  # git -C "$REPO_DIR" reset --hard origin/HEAD
  # git -C "$REPO_DIR" clean -fd
fi

if [[ -f "$REPO_DIR/pyproject.toml" || -f "$REPO_DIR/setup.py" ]]; then
  /opt/venv/bin/pip install -e "$REPO_DIR"
fi


mkdir -p "$TS_STATE_DIR" /var/run/tailscale

# Start daemon
nohup /usr/sbin/tailscaled \
  --tun=userspace-networking \
  --state="${TS_STATE_DIR}/tailscaled.state" \
  --socket=/var/run/tailscale/tailscaled.sock \
  >/tmp/tailscaled.log 2>&1 &

# Wait for socket
for _ in {1..100}; do
  [[ -S /var/run/tailscale/tailscaled.sock ]] && break
  sleep 0.1
done

hn="${TS_HOSTNAME:-gpu-box}"

# Build tailscale up args
UP_ARGS=(--hostname="$hn" --accept-dns="$TS_ACCEPT_DNS" --advertise-tags="$TS_TAGS")

if [[ "${TS_ENABLE_SSH:-true}" == "true" ]]; then
  UP_ARGS+=(--ssh)
fi

if [[ -n "${TS_AUTHKEY:-}" ]]; then
  UP_ARGS+=(--authkey="$TS_AUTHKEY")
fi

# Split extra args by spaces (simple case)
EXTRA=()
if [[ -n "${TS_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA <<<"${TS_EXTRA_ARGS}"
fi

# Keep retrying until it succeeds.
# If no authkey is provided, tailscale up will print a login URL you can open to authorize.
for _ in {1..60}; do
  if tailscale up "${UP_ARGS[@]}" "${EXTRA[@]}"; then
    break
  fi
  sleep 2
done

tailscale status || true

# Run user command, or just keep container alive for SSH sessions
if [[ $# -gt 0 ]]; then
  exec "$@"
else
  exec sleep infinity
fi
BASH

RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /proj
RUN mkdir -p /proj

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []
