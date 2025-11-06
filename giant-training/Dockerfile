# CUDA 12.9 + cuDNN + Python 3.11 + JAX stack + SSH + Tailscale + rclone + s5cmd for S3 I/O
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Base packages + SSH daemon + fuse3 (for rclone mount)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget curl ca-certificates vim pkg-config \
        python3.11 python3.11-venv python3-pip                       \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev             \
        libsqlite3-dev libffi-dev liblzma-dev tk-dev uuid-dev        \
        openssh-server iproute2 procps fuse3 unzip gnupg lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Python virtual env
ENV VENV_DIR=/opt/venv
RUN python3.11 -m venv "$VENV_DIR"
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

# ── SSH server: password-only root login (your original setup) ────────────────
RUN mkdir -p /var/run/sshd && \
    echo 'root:dev' | chpasswd && \
    rm -rf /root/.ssh && mkdir -p /root && \
    printf '%s\n' \
      'PermitRootLogin yes' \
      'PasswordAuthentication yes' \
      'PubkeyAuthentication no' \
      'ChallengeResponseAuthentication no' \
      'UsePAM yes' >> /etc/ssh/sshd_config

# Make venv default (your original)
RUN printf '%s\n' \
      'VIRTUAL_ENV=/opt/venv' \
      'PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' \
      >> /etc/environment
RUN printf '%s\n' 'source /opt/venv/bin/activate || true' > /etc/profile.d/activate-venv.sh

# ── Tools we need for networking & S3 I/O ─────────────────────────────────────
# 1) Tailscale
RUN curl -fsSL https://tailscale.com/install.sh | sh

# 2) rclone (pin version + arch-aware; provides rclone mount/bisync)
ARG RCLONE_VERSION=1.67.0
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
      amd64) rarch="amd64" ;; \
      arm64) rarch="arm64" ;; \
      *) echo "Unsupported arch: $arch"; exit 1 ;; \
    esac; \
    curl -fsSLo /tmp/rclone.zip "https://downloads.rclone.org/v${RCLONE_VERSION}/rclone-v${RCLONE_VERSION}-linux-${rarch}.zip"; \
    unzip -q /tmp/rclone.zip -d /tmp; \
    install -m 0755 "/tmp/rclone-v${RCLONE_VERSION}-linux-${rarch}/rclone" /usr/local/bin/; \
    rm -rf /tmp/rclone.zip "/tmp/rclone-v${RCLONE_VERSION}-linux-${rarch}"; \
    rclone version

# 3) s5cmd (fast parallel S3 copies) — pin to existing release + arch-aware
ARG S5VER=2.3.0
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
      amd64)  s5arch="Linux-64bit" ;; \
      arm64)  s5arch="Linux-arm64" ;; \
      i386)   s5arch="Linux-32bit" ;; \
      ppc64el|ppc64le) s5arch="Linux-ppc64le" ;; \
      armhf|armv7l)    s5arch="Linux-armv6" ;; \
      *) echo "Unsupported arch: $arch"; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/s5cmd.tgz \
      "https://github.com/peak/s5cmd/releases/download/v${S5VER}/s5cmd_${S5VER}_${s5arch}.tar.gz"; \
    tar -xzf /tmp/s5cmd.tgz -C /usr/local/bin s5cmd; \
    rm -f /tmp/s5cmd.tgz; \
    s5cmd --version

# ── Runtime env (filled via pod console) ──────────────────────────────────────
ENV TS_AUTHKEY="" \
    S3_ENDPOINT="" \
    S3_BUCKET="ml-home" \
    S3_ACCESS_KEY_ID="" \
    S3_SECRET_ACCESS_KEY="" \
    MOUNT_BUCKET="1" \
    RCLONE_VFS_CACHE_MODE="full" \
    RCLONE_VFS_WRITE_BACK="2s" \
    RCLONE_CONCURRENCY="64" \
    HF_HOME="/root/ml-home/hf" \
    TRANSFORMERS_CACHE="/root/ml-home/hf/transformers" \
    AWS_EC2_METADATA_DISABLED="true"

# ── Entrypoint: tailscale up → rclone remote → mount or bisync → sshd ────────
COPY --chown=root:root <<'BASH' /usr/local/bin/entrypoint.sh
#!/usr/bin/env bash
set -euo pipefail

MOUNTPOINT="${MOUNTPOINT:-/root/ml-home}"
CACHE_DIR="${RCLONE_CACHE_DIR:-/tmp/rclone-cache}"
CONCURRENCY="${RCLONE_CONCURRENCY:-64}"
WRITE_BACK="${RCLONE_VFS_WRITE_BACK:-2s}"

mkdir -p "$MOUNTPOINT" "$CACHE_DIR" /var/run/sshd

# 1) Tailscale (kernel TUN preferred; else userspace + SOCKS5)
if [[ -n "${TS_AUTHKEY:-}" ]]; then
  if [[ -e /dev/net/tun ]]; then
    /usr/sbin/tailscaled --state=/tmp/tailscaled.state --tun=autodetect &
    sleep 1
    tailscale up --authkey="${TS_AUTHKEY}" --hostname="gpu-$(hostname)" --accept-dns=false || true
  else
    /usr/sbin/tailscaled --state=/tmp/tailscaled.state --tun=userspace-networking --socks5-server=localhost:1055 &
    sleep 1
    tailscale up --authkey="${TS_AUTHKEY}" --hostname="gpu-$(hostname)" --accept-dns=false || true
    export ALL_PROXY="socks5://127.0.0.1:1055"
    export HTTPS_PROXY="$ALL_PROXY"; export HTTP_PROXY="$ALL_PROXY"
    export NO_PROXY="localhost,127.0.0.1"
  fi
fi

# 2) Wait for S3 endpoint if TS is used (best effort)
if [[ -n "${S3_ENDPOINT:-}" ]]; then
  for i in {1..30}; do
    curl -fsS "${S3_ENDPOINT}/minio/health/ready" >/dev/null && break || sleep 1
  done || true
fi

# 3) rclone remote (idempotent)
if ! rclone listremotes 2>/dev/null | grep -q '^mlstore:'; then
  rclone config create mlstore s3 provider=Minio \
    access_key_id="${S3_ACCESS_KEY_ID}" secret_access_key="${S3_SECRET_ACCESS_KEY}" \
    endpoint="${S3_ENDPOINT}" >/dev/null
fi

# 4) Prefer mount if allowed; otherwise fall back to bisync
if [[ "${MOUNT_BUCKET:-1}" = "1" && -e /dev/fuse ]]; then
  echo "[entrypoint] mounting ${S3_BUCKET} at ${MOUNTPOINT}"
  umount -f "${MOUNTPOINT}" 2>/dev/null || true
  rclone mount mlstore:${S3_BUCKET} "${MOUNTPOINT}" \
    --daemon \
    --umask 022 \
    --vfs-cache-mode "${RCLONE_VFS_CACHE_MODE:-full}" \
    --vfs-write-back "${WRITE_BACK}" \
    --cache-dir "${CACHE_DIR}" \
    --transfers "${CONCURRENCY}" \
    --checkers "${CONCURRENCY}" \
    --dir-cache-time 1h \
    --poll-interval 30s \
    --log-file /tmp/rclone-mount.log || true

  sleep 1
  if mountpoint -q "${MOUNTPOINT}"; then
    echo "[entrypoint] mount OK → ${MOUNTPOINT}"
  else
    echo "[entrypoint] mount failed, starting sync mode"
    MOUNT_BUCKET="0"
  fi
fi

if [[ "${MOUNT_BUCKET:-0}" = "0" || ! -e /dev/fuse ]]; then
  echo "[entrypoint] sync mode: initial pull to ${MOUNTPOINT}"
  rclone sync mlstore:${S3_BUCKET} "${MOUNTPOINT}" \
    --create-empty-src-dirs --transfers "${CONCURRENCY}" --checkers "${CONCURRENCY}" --fast-list

  if [[ ! -f "${MOUNTPOINT}/.rclonestate" ]]; then
    rclone bisync "${MOUNTPOINT}" mlstore:${S3_BUCKET} --resync --max-delete 100000 || true
    touch "${MOUNTPOINT}/.rclonestate"
  fi

  cat > /usr/local/bin/mlhome-bisync-loop.sh <<'LOOP'
#!/usr/bin/env bash
set -euo pipefail
SRC="${MOUNTPOINT:-/root/ml-home}"
DST="mlstore:${S3_BUCKET:-ml-home}"
while true; do
  rclone bisync "${SRC}" "${DST}" --max-delete 100000 --create-empty-src-dirs || true
  sleep 8
done
LOOP
  chmod +x /usr/local/bin/mlhome-bisync-loop.sh
  ( S3_BUCKET="${S3_BUCKET}" MOUNTPOINT="${MOUNTPOINT}" nohup /usr/local/bin/mlhome-bisync-loop.sh >/tmp/mlhome-bisync.log 2>&1 & ) || true
fi

# 5) Start SSHD in the foreground
exec /usr/sbin/sshd -D
BASH
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 22
WORKDIR /workspace/app
RUN mkdir -p /workspace/app

CMD ["/usr/local/bin/entrypoint.sh"]
