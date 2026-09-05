#!/usr/bin/env bash

set -euo pipefail
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
unset DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH DYLD_FRAMEWORK_PATH DYLD_FALLBACK_FRAMEWORK_PATH DYLD_INSERT_LIBRARIES LD_LIBRARY_PATH LD_PRELOAD

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PULID_PROJECT_ROOT="${PROJECT_DIR}"
VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Python absent ou dossier déplacé : ${VENV_PYTHON}. Relancez ./install_macos.sh." >&2
  exit 1
fi
cd "${PROJECT_DIR}"
unset PYTHONHOME PYTHONPATH
if ! "${VENV_PYTHON}" -I "${PROJECT_DIR}/scripts/check_environment.py"; then
  echo "Relancez ./install_macos.sh pour recréer l'environnement." >&2
  exit 1
fi
exec "${VENV_PYTHON}" -I -m pulid_app.server \
  --host 127.0.0.1 \
  --port 12693 \
  --device mps \
  --cors-origin http://localhost:8800 \
  "$@"
