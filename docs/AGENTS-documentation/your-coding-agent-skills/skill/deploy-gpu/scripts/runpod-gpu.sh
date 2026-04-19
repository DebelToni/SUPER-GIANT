#!/usr/bin/env bash
set -euo pipefail

DEFAULT_TEMPLATE_ID="bg2jwnb3zk"
DEFAULT_TEMPLATE_NAME="GIANT-container"
DEFAULT_GPU_FALLBACK=(
  "NVIDIA RTX A6000"
  "NVIDIA RTX A5000"
  "NVIDIA RTX A4500"
  "NVIDIA RTX A4000"
  "NVIDIA A40"
  "NVIDIA GeForce RTX 5090"
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
WAIT_SCRIPT="$REPO_ROOT/CICD/tools/wait-new-gpu.sh"
RUNPOD_CONFIG="$HOME/.runpod/config.toml"

usage() {
  cat <<'EOF'
Usage:
  runpod-gpu.sh create [--gpu GPU_NAME|auto] [--name NAME] [--template-name NAME] [--template-id ID] [--spot] [--wait] [--id-only]
  runpod-gpu.sh list
  runpod-gpu.sh stop POD_ID
  runpod-gpu.sh remove POD_ID

Defaults:
  - template defaults to GIANT-container: bg2jwnb3zk
  - cloud type is always SECURE
  - `create --gpu auto` tries this order:
      NVIDIA RTX A6000 -> NVIDIA RTX A5000 -> NVIDIA RTX A4500 ->
      NVIDIA RTX A4000 -> NVIDIA A40 -> NVIDIA GeForce RTX 5090

Examples:
  runpod-gpu.sh create --gpu auto --name longdsl-baseline --wait
  runpod-gpu.sh create --gpu "NVIDIA GeForce RTX 5090" --template-name TUESGPT-container --wait
  runpod-gpu.sh create --gpu "NVIDIA RTX A40" --spot --id-only
  runpod-gpu.sh list
  runpod-gpu.sh stop 123abc
  runpod-gpu.sh remove 123abc
EOF
}

die() {
  printf '[runpod-gpu] %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

load_api_key() {
  if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
    printf '%s\n' "$RUNPOD_API_KEY"
    return 0
  fi
  if [[ -f "$RUNPOD_CONFIG" ]]; then
    python3 - <<'PY'
from pathlib import Path
import re

cfg = Path.home() / ".runpod" / "config.toml"
text = cfg.read_text(encoding="utf-8")
match = re.search(r'apikey\s*=\s*"([^"]+)"', text)
print(match.group(1) if match else "")
PY
    return 0
  fi
  printf '\n'
}

json_get() {
  local key_expr="$1"
  python3 -c '
import json
import sys

expr = sys.argv[1]
data = json.load(sys.stdin)
cur = data
for part in expr.split("."):
    if not part:
        continue
    if isinstance(cur, dict):
        cur = cur.get(part)
    else:
        cur = None
        break
if cur is None:
    print("")
elif isinstance(cur, (dict, list)):
    print(json.dumps(cur))
else:
    print(cur)
' "$key_expr"
}

print_summary_json() {
  local response="$1"
  python3 -c '
import json
import sys

data = json.load(sys.stdin)
gpu = data.get("machine", {}).get("gpuTypeId") or data.get("gpuTypeId") or ""
status = data.get("desiredStatus") or data.get("status") or ""
cost = data.get("costPerHr")
name = data.get("name") or ""
pod_id = data.get("id") or ""
dc = data.get("machine", {}).get("dataCenterId") or ""
parts = [
    f"pod_id={pod_id}",
    f"name={name}",
    f"gpu={gpu}",
    f"status={status}",
]
if cost is not None:
    parts.append(f"cost_per_hr={cost}")
if dc:
    parts.append(f"datacenter={dc}")
print("[runpod-gpu] " + " ".join(parts))
' <<<"$response"
}

lookup_pod_id_by_name() {
  local target_name="$1"
  runpodctl get pod | python3 -c '
import sys

target = sys.argv[1]
lines = [line.rstrip("\n") for line in sys.stdin if line.strip()]
if len(lines) <= 1:
    print("")
    raise SystemExit(0)
best = ""
for line in lines[1:]:
    cols = [col.strip() for col in line.split("\t") if col.strip()]
    if len(cols) < 2:
        cols = line.split()
    if len(cols) >= 2 and cols[1] == target:
        best = cols[0]
print(best)
' "$target_name"
}

resolve_template_id() {
  local api_key="$1"
  local template_name="$2"
  local template_id="$3"

  if [[ -n "$template_id" ]]; then
    printf '%s\n' "$template_id"
    return 0
  fi

  if [[ -z "$template_name" ]]; then
    template_name="$DEFAULT_TEMPLATE_NAME"
  fi

  local response
  response=$(curl --silent --show-error \
    --request GET \
    --url https://rest.runpod.io/v1/templates \
    --header "Authorization: Bearer $api_key")

  python3 -c '
import json
import sys

target = sys.argv[1]
data = json.load(sys.stdin)
matches = [item for item in data if item.get("name") == target]
if not matches:
    print("")
    raise SystemExit(0)
if len(matches) > 1:
    raise SystemExit(f"multiple templates found for name: {target}")
print(matches[0].get("id", ""))
' "$template_name" <<<"$response"
}

try_cli_create() {
  local gpu="$1"
  local name="$2"
  local template_id="$3"
  local output
  set +e
  output=$(runpodctl create pod --secureCloud --gpuType "$gpu" --templateId "$template_id" --name "$name" 2>&1)
  local status=$?
  set -e
  printf '%s' "$output"
  return "$status"
}

rest_create() {
  local api_key="$1"
  local gpu="$2"
  local name="$3"
  local interruptible="$4"
  local template_id="$5"
  local payload
  payload=$(python3 - "$gpu" "$name" "$interruptible" "$template_id" <<'PY'
import json
import sys

gpu = sys.argv[1]
name = sys.argv[2]
interruptible = sys.argv[3].lower() == "true"
template_id = sys.argv[4]
payload = {
    "cloudType": "SECURE",
    "computeType": "GPU",
    "gpuCount": 1,
    "gpuTypeIds": [gpu],
    "templateId": template_id,
    "name": name,
}
if interruptible:
    payload["interruptible"] = True
print(json.dumps(payload, separators=(",", ":")))
PY
)
  curl --silent --show-error \
    --request POST \
    --url https://rest.runpod.io/v1/pods \
    --header "Authorization: Bearer $api_key" \
    --header "Content-Type: application/json" \
    --data "$payload"
}

wait_for_gpu() {
  [[ -x "$WAIT_SCRIPT" ]] || die "wait script not found: $WAIT_SCRIPT"
  "$WAIT_SCRIPT"
}

create_cmd() {
  require_cmd runpodctl
  require_cmd curl
  require_cmd python3

  local gpu="auto"
  local name="giant-$(date +%m%d-%H%M%S)"
  local template_name="$DEFAULT_TEMPLATE_NAME"
  local template_id=""
  local spot=0
  local wait_flag=0
  local id_only=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpu)
        gpu="${2:-}"
        shift 2
        ;;
      --name)
        name="${2:-}"
        shift 2
        ;;
      --template-name)
        template_name="${2:-}"
        shift 2
        ;;
      --template-id)
        template_id="${2:-}"
        shift 2
        ;;
      --spot)
        spot=1
        shift
        ;;
      --wait)
        wait_flag=1
        shift
        ;;
      --id-only)
        id_only=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown create arg: $1"
        ;;
    esac
  done

  local api_key=""
  api_key="$(load_api_key)"
  [[ -n "$api_key" ]] || die "RUNPOD_API_KEY not set and ~/.runpod/config.toml missing apikey"

  template_id="$(resolve_template_id "$api_key" "$template_name" "$template_id")"
  [[ -n "$template_id" ]] || die "template not found: ${template_name:-$DEFAULT_TEMPLATE_NAME}"

  local -a gpu_candidates
  if [[ "$gpu" == "auto" ]]; then
    gpu_candidates=("${DEFAULT_GPU_FALLBACK[@]}")
  else
    gpu_candidates=("$gpu")
  fi

  local chosen_gpu=""
  local response=""
  local pod_id=""

  for candidate in "${gpu_candidates[@]}"; do
    printf '[runpod-gpu] trying gpu=%s template=%s template_id=%s\n' "$candidate" "$template_name" "$template_id" >&2
    chosen_gpu="$candidate"

    if [[ $spot -eq 0 ]]; then
      local cli_out
      if cli_out=$(try_cli_create "$candidate" "$name" "$template_id"); then
        for _ in 1 2 3 4 5; do
          pod_id="$(lookup_pod_id_by_name "$name")"
          [[ -n "$pod_id" ]] && break
          sleep 1
        done
        if [[ -n "$pod_id" ]]; then
          if [[ $id_only -eq 1 ]]; then
            printf '%s\n' "$pod_id"
          else
            printf '[runpod-gpu] pod_id=%s name=%s gpu=%s status=CREATED via=cli\n' "$pod_id" "$name" "$candidate"
          fi
          [[ $wait_flag -eq 1 ]] && wait_for_gpu
          return 0
        fi
        if [[ $id_only -eq 1 ]]; then
          printf '\n'
        else
          printf '[runpod-gpu] create succeeded via cli but pod id lookup failed; name=%s gpu=%s\n' "$name" "$candidate"
        fi
        [[ $wait_flag -eq 1 ]] && wait_for_gpu
        return 0
      fi
      if [[ "$cli_out" != *'required flag(s) "imageName" not set'* ]]; then
        printf '[runpod-gpu] cli create failed for gpu=%s; falling back to REST\n' "$candidate" >&2
      else
        printf '[runpod-gpu] cli create needs imageName; using REST fallback\n' >&2
      fi
    fi

    response="$(rest_create "$api_key" "$candidate" "$name" "$([[ $spot -eq 1 ]] && printf true || printf false)" "$template_id")"
    pod_id="$(json_get 'id' <<<"$response")"
    if [[ -n "$pod_id" ]]; then
      if [[ $id_only -eq 1 ]]; then
        printf '%s\n' "$pod_id"
      else
        print_summary_json "$response"
      fi
      [[ $wait_flag -eq 1 ]] && wait_for_gpu
      return 0
    fi
    printf '[runpod-gpu] create failed for gpu=%s response=%s\n' "$candidate" "$response" >&2
  done

  die "unable to create pod after trying all requested GPUs"
}

list_cmd() {
  require_cmd runpodctl
  runpodctl get pod
}

stop_cmd() {
  require_cmd runpodctl
  [[ $# -eq 1 ]] || die "stop requires POD_ID"
  runpodctl stop pod "$1"
}

remove_cmd() {
  require_cmd runpodctl
  [[ $# -eq 1 ]] || die "remove requires POD_ID"
  runpodctl remove pod "$1"
}

main() {
  [[ $# -ge 1 ]] || { usage; exit 1; }
  local cmd="$1"
  shift
  case "$cmd" in
    create)
      create_cmd "$@"
      ;;
    list)
      list_cmd "$@"
      ;;
    stop)
      stop_cmd "$@"
      ;;
    remove)
      remove_cmd "$@"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      die "unknown command: $cmd"
      ;;
  esac
}

main "$@"
