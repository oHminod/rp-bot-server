#!/usr/bin/env bash

set -euo pipefail
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
unset DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH DYLD_FRAMEWORK_PATH DYLD_FALLBACK_FRAMEWORK_PATH DYLD_INSERT_LIBRARIES LD_LIBRARY_PATH LD_PRELOAD

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Python PuLID introuvable. Exécutez d'abord : ./install_macos.sh" >&2
  exit 1
fi
"${VENV_PYTHON}" -I "${PROJECT_DIR}/scripts/check_environment.py"

cd "${PROJECT_DIR}"
exec "${VENV_PYTHON}" -I frontend/server.py --host 127.0.0.1 --port 8888 "$@"
