#!/usr/bin/env bash
set -e

# Clone once, pull on subsequent starts
if [ ! -d "${PROJECT_DIR}/.git" ]; then
  git clone --depth 1 "${REPO_URL}" "${PROJECT_DIR}"
else
  git -C "${PROJECT_DIR}" pull --ff-only
fi

exec "$@"
