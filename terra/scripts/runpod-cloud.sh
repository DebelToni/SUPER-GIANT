#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:-usage}
shift || true

STATE_DIR="${HOME}/.runpod-cloud"
API_URL="https://api.runpod.io/v1"
mkdir -p "$STATE_DIR"

die()  { echo "❌ $*" >&2; exit 1; }
json() { jq -nc "$@"; }

require_key() { [[ -z "${RUNPOD_API_KEY:-}" ]] && \
  die "RUNPOD_API_KEY env var not set (run 'export RUNPOD_API_KEY=<key>')" ; }

case "$ACTION" in
  create)
    NAME=${1:-rp-$(date +%s)}
    IMAGE=${2:-${RUNPOD_IMAGE_NAME:-}}
    GPU_TYPE=${3:-${RUNPOD_GPU_TYPE:-}}
    DISK=${4:-10}            # container disk GB
    VOLUME=${5:-10}          # persistent volume GB
    COST=${6:-0.5}           # max $/hr

    [[ -z "$IMAGE" || -z "$GPU_TYPE" ]] && \
      die "Usage: $0 create <name> <image> <gpuType> [diskGB] [volumeGB] [maxCost]"

    OUT=$(runpodctl create pods \
            --name "$NAME" \
            --gpuType "$GPU_TYPE" \
            --imageName "$IMAGE" \
            --containerDiskSize "$DISK" \
            --volumeSize "$VOLUME" \
            --cost "$COST" \
            --communityCloud)

    POD_ID=$(sed -n 's/^pod "\([^"]\+\)".*/\1/p' <<<"$OUT")
    [[ -z "$POD_ID" ]] && die "Could not parse pod ID from output: $OUT"

    echo "$POD_ID" >"$STATE_DIR/$NAME"
    json --arg id "$POD_ID" '{id:$id}'
    ;;

  ip)
    NAME=${1:?missing pod name}
    POD_ID=$(cat "$STATE_DIR/$NAME" 2>/dev/null) || die "No state for $NAME"
    require_key
    PUBLIC_IP=$(curl -fsSL \
      -H "Authorization: Bearer $RUNPOD_API_KEY" \
      "$API_URL/pods/$POD_ID" | jq -r '.publicIp')

    [[ "$PUBLIC_IP" == "null" || -z "$PUBLIC_IP" ]] && \
      die "Pod $POD_ID has no public IP yet"

    json --arg ip "$PUBLIC_IP" '{ip:$ip}'
    ;;

  start|stop|destroy)
    NAME=${1:?missing pod name}
    POD_ID=$(cat "$STATE_DIR/$NAME" 2>/dev/null) || die "No state for $NAME"
    case "$ACTION" in
      start)   runpodctl start  pod "$POD_ID" ;;
      stop)    runpodctl stop   pod "$POD_ID" ;;
      destroy) runpodctl remove pod "$POD_ID"; rm -f "$STATE_DIR/$NAME" ;;
    esac
    ;;

  *)
    echo "Usage: $0 {create|ip|start|stop|destroy} <name>" >&2
    exit 1 ;;
esac

