# ---- build s5cmd (static, CGO disabled) --------------------------------------
FROM --platform=$BUILDPLATFORM golang:1.22-alpine AS s5cmd-builder
ARG TARGETOS
ARG TARGETARCH
ARG S5VER=v2.3.0
ENV CGO_ENABLED=0 GOFLAGS="-buildvcs=false"
RUN apk add --no-cache git
RUN GOOS=${TARGETOS} GOARCH=${TARGETARCH} go install -ldflags="-s -w" github.com/peak/s5cmd/v2@${S5VER}

# ---- main image --------------------------------------------------------------
# CUDA 12.9 + cuDNN + Python 3.11 + JAX + SSH + Tailscale + rclone + s5cmd
FROM --platform=linux/amd64 nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Guard: bail out if not amd64 (we build/push only x86_64)
RUN dpkg --print-architecture | grep -qx amd64 || (echo "This image targets amd64/x86_64 only." && exit 1)

# Base packages + SSH daemon + fuse3 (for rclone mount)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget curl ca-certificates vim tmux pkg-config \
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

# 2) rclone (amd64-only + retries)
ARG RCVER=1.67.0
RUN set -eux; \
    arch="amd64"; \
    base="https://downloads.rclone.org"; \
    url="${base}/v${RCVER}/rclone-v${RCVER}-linux-${arch}.zip"; \
    curl -fSL --retry 5 --retry-all-errors --retry-delay 2 -o /tmp/rclone.zip "$url"; \
    unzip -q /tmp/rclone.zip -d /tmp; \
    install -m 0755 "/tmp/rclone-v${RCVER}-linux-${arch}/rclone" /usr/local/bin/rclone; \
    rm -rf /tmp/rclone.zip "/tmp/rclone-v${RCVER}-linux-${arch}"; \
    env -u RCLONE_VERSION rclone --version

# 3) s5cmd from builder
COPY --from=s5cmd-builder /go/bin/s5cmd /usr/local/bin/s5cmd
RUN s5cmd version

# ── Runtime env (filled via env-file or flags) ────────────────────────────────
ENV TS_AUTHKEY="" \
    S3_ENDPOINT="" \
    S3_BUCKET="ml-home" \
    S3_ACCESS_KEY_ID="" \
    S3_SECRET_ACCESS_KEY="" \
    AWS_REGION="us-east-1" \
    AWS_EC2_METADATA_DISABLED="true" \
    MOUNT_BUCKET="1" \
    RCLONE_VFS_CACHE_MODE="full" \
    RCLONE_VFS_WRITE_BACK="2s" \
    RCLONE_CONCURRENCY="64" \
    HF_HOME="/root/ml-home/hf" \
    TRANSFORMERS_CACHE="/root/ml-home/hf/transformers" \
    RCLONE_CONFIG="/root/.config/rclone/rclone.conf"

# ── Hardened entrypoint: tailscale up → rclone remote → mount/bisync → sshd ──
COPY --chown=root:root <<'BASH' /usr/local/bin/entrypoint.sh
#!/usr/bin/env bash
set -euo pipefail

echo "[cuda-ts-s3] start"
echo "S3_ENDPOINT=${S3_ENDPOINT:-<empty>}"
echo "S3_BUCKET=${S3_BUCKET:-<empty>}"
[[ -n "${S3_ACCESS_KEY_ID:-}" ]] && echo "S3_ACCESS_KEY_ID=<set>" || echo "S3_ACCESS_KEY_ID=<empty>"
[[ -n "${S3_SECRET_ACCESS_KEY:-}" ]] && echo "S3_SECRET_ACCESS_KEY=<set>" || echo "S3_SECRET_ACCESS_KEY=<empty>"

# Map S3_* -> AWS_* (for s5cmd/AWS SDKs)
export AWS_ACCESS_KEY_ID="${S3_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}"
export AWS_SECRET_ACCESS_KEY="${S3_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_S3_FORCE_PATH_STYLE="true"
export S3_ENDPOINT_URL="${S3_ENDPOINT:-}"

MOUNTPOINT="${MOUNTPOINT:-/root/ml-home}"
CACHE_DIR="${RCLONE_CACHE_DIR:-/tmp/rclone-cache}"
CONCURRENCY="${RCLONE_CONCURRENCY:-64}"
WRITE_BACK="${RCLONE_VFS_WRITE_BACK:-2s}"
CONF="${RCLONE_CONFIG:-/root/.config/rclone/rclone.conf}"

mkdir -p "$MOUNTPOINT" "$CACHE_DIR" /var/run/sshd "$(dirname "$CONF")"
: > "$CONF"
chmod 600 "$CONF"

# 1) Tailscale: prefer kernel TUN if available; else userspace + SOCKS5
if [[ -n "${TS_AUTHKEY:-}" ]]; then
  if [[ -e /dev/net/tun ]]; then
    /usr/sbin/tailscaled --state=/tmp/tailscaled.state --tun=autodetect &
    sleep 1
    tailscale up --authkey="${TS_AUTHKEY}" --hostname="gpu-$(hostname)" --accept-dns=false || true
    echo "[cuda-ts-s3] tailscale up (kernel TUN)"
  else
    /usr/sbin/tailscaled --state=/tmp/tailscaled.state --tun=userspace-networking --socks5-server=localhost:1055 &
    sleep 1
    tailscale up --authkey="${TS_AUTHKEY}" --hostname="gpu-$(hostname)" --accept-dns=false || true
    export ALL_PROXY="socks5://127.0.0.1:1055"
    export HTTPS_PROXY="$ALL_PROXY"; export HTTP_PROXY="$ALL_PROXY"
    export NO_PROXY="localhost,127.0.0.1"
    echo "[cuda-ts-s3] tailscale up (userspace) SOCKS5=$ALL_PROXY"
  fi
fi

# 2) Wait for S3 /minio/health/ready (best effort, if endpoint set)
if [[ -n "${S3_ENDPOINT:-}" ]]; then
  for i in {1..30}; do
    curl -fsS "${S3_ENDPOINT%/}/minio/health/ready" >/dev/null && break || sleep 1
  done || true
  echo "[cuda-ts-s3] health check done"
fi

# 3) rclone remote (create/overwrite, explicit config, MinIO provider)
if [[ -n "${S3_ENDPOINT:-}" ]]; then
  echo "[cuda-ts-s3] creating rclone remote 'mlstore' -> ${S3_ENDPOINT}"
  rclone config delete mlstore --config "$CONF" >/dev/null 2>&1 || true
  rclone config create mlstore s3 \
    provider Minio \
    env_auth false \
    access_key_id "${AWS_ACCESS_KEY_ID:-}" \
    secret_access_key "${AWS_SECRET_ACCESS_KEY:-}" \
    endpoint "${S3_ENDPOINT}" \
    region "${AWS_REGION}" \
    no_check_bucket true \
    force_path_style true \
    --config "$CONF" --non-interactive 1>/dev/null
  rclone listremotes --config "$CONF" || true
fi

# 4) Prefer mount if allowed; otherwise fall back to bisync
if [[ "${MOUNT_BUCKET:-1}" = "1" && -e /dev/fuse ]]; then
  echo "[cuda-ts-s3] mounting ${S3_BUCKET} at ${MOUNTPOINT}"
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
    --log-file /tmp/rclone-mount.log \
    --config "$CONF" || true

  sleep 1
  if mountpoint -q "${MOUNTPOINT}"; then
    echo "[cuda-ts-s3] mount OK → ${MOUNTPOINT}"
  else
    echo "[cuda-ts-s3] mount failed, switching to sync mode"
    MOUNT_BUCKET="0"
  fi
fi

if [[ "${MOUNT_BUCKET:-0}" = "0" || ! -e /dev/fuse ]]; then
  echo "[cuda-ts-s3] sync mode: initial pull to ${MOUNTPOINT}"
  rclone sync mlstore:${S3_BUCKET} "${MOUNTPOINT}" \
    --create-empty-src-dirs --transfers "${CONCURRENCY}" --checkers "${CONCURRENCY}" --fast-list --config "$CONF"

  if [[ ! -f "${MOUNTPOINT}/.rclonestate" ]]; then
    rclone bisync "${MOUNTPOINT}" mlstore:${S3_BUCKET} --resync --max-delete 100000 --config "$CONF" || true
    touch "${MOUNTPOINT}/.rclonestate"
  fi

  cat > /usr/local/bin/mlhome-bisync-loop.sh <<'LOOP'
#!/usr/bin/env bash
set -euo pipefail
SRC="${MOUNTPOINT:-/root/ml-home}"
DST="mlstore:${S3_BUCKET:-ml-home}"
CONF="${RCLONE_CONFIG:-/root/.config/rclone/rclone.conf}"
while true; do
  rclone bisync "${SRC}" "${DST}" --max-delete 100000 --create-empty-src-dirs --config "$CONF" || true
  sleep 8
done
LOOP
  chmod +x /usr/local/bin/mlhome-bisync-loop.sh
  ( S3_BUCKET="${S3_BUCKET}" MOUNTPOINT="${MOUNTPOINT}" RCLONE_CONFIG="${CONF}" nohup /usr/local/bin/mlhome-bisync-loop.sh >/tmp/mlhome-bisync.log 2>&1 & ) || true
fi

# 5) Optional selftest then SSHD in foreground
if [[ "${1:-}" == "selftest" ]]; then
  if [[ -z "${S3_ENDPOINT:-}" || -z "${S3_BUCKET:-}" ]]; then
    echo "[cuda-ts-s3] selftest: set S3_ENDPOINT and S3_BUCKET"; exit 2; fi
  echo "[cuda-ts-s3] curl health: ${S3_ENDPOINT%/}/minio/health/ready"
  curl -sSf "${S3_ENDPOINT%/}/minio/health/ready" >/dev/null && echo "[ok] minio health ready"
  echo "[cuda-ts-s3] s5cmd ls s3://${S3_BUCKET}/"
  s5cmd --endpoint-url="${S3_ENDPOINT}" ls "s3://${S3_BUCKET}/" || { echo "[fail] s5cmd"; exit 3; }
  echo "[cuda-ts-s3] rclone ls mlstore:${S3_BUCKET}"
  rclone --config "$CONF" ls "mlstore:${S3_BUCKET}" || { echo "[fail] rclone"; exit 4; }
  echo "[cuda-ts-s3] selftest: SUCCESS"
  exit 0
fi

exec /usr/sbin/sshd -D
BASH
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 22
WORKDIR /workspace/app
RUN mkdir -p /workspace/app

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []

