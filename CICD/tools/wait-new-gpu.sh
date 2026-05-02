#!/usr/bin/env bash
set -euo pipefail

interval="${1:-5}"
max_seconds="${2:-300}"
baseline="$( (tailscale status | rg gpu || true) | wc -l | tr -d ' ' )"
start="$(date +%s)"

if ssh -o BatchMode=yes -o ConnectTimeout=5 root@gpu-box "echo ok" >/dev/null 2>&1; then
  echo "gpu ssh reachable"
  exit 0
fi

while true; do
  current="$( (tailscale status | rg gpu || true) | wc -l | tr -d ' ' )"
  if [ "$current" -gt "$baseline" ]; then
    echo "new gpu detected"
    exit 0
  fi
  now="$(date +%s)"
  if [ $((now - start)) -ge "$max_seconds" ]; then
    echo "timed out waiting for gpu after ${max_seconds}s" >&2
    exit 124
  fi
  sleep "$interval"
done
