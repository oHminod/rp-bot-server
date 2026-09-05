#!/usr/bin/env bash
# Sourced by both launchers, after sanitizing PATH and library search variables.
VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"
RUNTIME_PYTHON="${VENV_PYTHON}"
if [[ -f "${PROJECT_DIR}/.venv/pulid-python-path" ]]; then
  IFS= read -r RUNTIME_PYTHON < "${PROJECT_DIR}/.venv/pulid-python-path"
  if [[ "${RUNTIME_PYTHON}" != /* ]]; then
    RUNTIME_PYTHON="${PROJECT_DIR}/${RUNTIME_PYTHON}"
  fi
fi
if [[ ! -x "${RUNTIME_PYTHON}" ]]; then
  echo "Python géré absent : ${RUNTIME_PYTHON}. Déplacez le dossier complet, ou relancez ./install_macos.sh." >&2
  exit 1
fi
"${RUNTIME_PYTHON}" -I "${PROJECT_DIR}/scripts/check_environment.py" --prepare
