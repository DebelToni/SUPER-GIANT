#!/usr/bin/env bash
set -euo pipefail
ACTION=${1:-usage}

# Where we persist pod-ids for easy lookup
STATE_DIR="${HOME}/.runpod-cloud"
mkdir -p "$STATE_DIR"

json() { jq -nc "$1"; }

case "$ACTION" in
  create)
    NAME=${2:-rp-$(date +%s)}
    IMAGE=${3:-"runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"}
    GPU_TYPE=${4:-"NVIDIA GeForce RTX 4090"}
    # Ask runpodctl to make the pod and capture ID
    POD_JSON=$(runpodctl create pods \
      --name "$NAME" --gpuType "$GPU_TYPE" --imageName "$IMAGE" \
      --containerDiskSize 10 --volumeSize 20 --json)
    POD_ID=$(echo "$POD_JSON" | jq -r '.id')
    echo "$POD_ID" > "$STATE_DIR/$NAME"
    json --arg id "$POD_ID" '{id:$id}'
    ;;

  ip)
    POD_ID=$(cat "$STATE_DIR/$2")
    # Using runpodctl but you could hit the REST endpoint directly
    PUBLIC_IP=$(runpodctl get pod "$POD_ID" --json | jq -r '.publicIp')
    json --arg ip "$PUBLIC_IP" '{ip:$ip}'
    ;;

  start) POD_ID=$(cat "$STATE_DIR/$2"); runpodctl start pod "$POD_ID" ;;
  stop)  POD_ID=$(cat "$STATE_DIR/$2"); runpodctl stop  pod "$POD_ID" ;;
  destroy)
    POD_ID=$(cat "$STATE_DIR/$2"); runpodctl remove pod "$POD_ID"
    rm -f "$STATE_DIR/$2"
    ;;

  *)
    echo "Usage: $0 {create|ip|start|stop|destroy} ..." >&2; exit 1 ;;
esac

