#!/usr/bin/env bash

set -euo pipefail
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
unset DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH DYLD_FRAMEWORK_PATH DYLD_FALLBACK_FRAMEWORK_PATH DYLD_INSERT_LIBRARIES LD_LIBRARY_PATH LD_PRELOAD

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_DIR}/scripts/prepare_runtime_macos.sh"
cd "${PROJECT_DIR}"
unset PYTHONHOME PYTHONPATH
"${VENV_PYTHON}" -I "${PROJECT_DIR}/scripts/check_environment.py"
exec "${VENV_PYTHON}" -I frontend/server.py --host 127.0.0.1 --port 8888 "$@"
