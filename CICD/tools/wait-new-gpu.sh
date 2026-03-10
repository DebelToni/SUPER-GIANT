#!/usr/bin/env bash
set -euo pipefail

interval="${1:-5}"
baseline="$( (tailscale status | rg gpu || true) | wc -l | tr -d ' ' )"

while true; do
  current="$( (tailscale status | rg gpu || true) | wc -l | tr -d ' ' )"
  if [ "$current" -gt "$baseline" ]; then
    echo "new gpu detected"
    exit 0
  fi
  sleep "$interval"
done
