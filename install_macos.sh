#!/usr/bin/env bash

set -Eeuo pipefail
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
unset DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH DYLD_FRAMEWORK_PATH DYLD_FALLBACK_FRAMEWORK_PATH DYLD_INSERT_LIBRARIES LD_LIBRARY_PATH LD_PRELOAD

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PULID_PROJECT_ROOT="${PROJECT_DIR}"
VENV_DIR="${PROJECT_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
PULID_INSTALL_PROFILE="${PULID_INSTALL_PROFILE:-development}"

while (($#)); do
  case "$1" in
    --production)
      PULID_INSTALL_PROFILE="production"
      ;;
    --development)
      PULID_INSTALL_PROFILE="development"
      ;;
    --help|-h)
      echo "Usage : ./install_macos.sh [--production|--development]"
      exit 0
      ;;
    *)
      echo "[ERREUR] Option d'installation inconnue : $1" >&2
      exit 2
      ;;
  esac
  shift
done

case "${PULID_INSTALL_PROFILE}" in
  production|development) ;;
  *)
    echo "[ERREUR] Profil PULID_INSTALL_PROFILE inconnu : ${PULID_INSTALL_PROFILE}" >&2
    exit 2
    ;;
esac

normalize_models_root() {
  local selected="${1%/}"
  if [[ "${selected}" == "~" ]]; then
    selected="${HOME}"
  elif [[ "${selected}" == "~/"* ]]; then
    selected="${HOME}/${selected:2}"
  elif [[ "${selected}" != /* ]]; then
    selected="${PROJECT_DIR}/${selected}"
  fi
  if [[ "$(basename -- "${selected}" | tr '[:upper:]' '[:lower:]')" == "pulid_models" ]]; then
    printf '%s\n' "${selected}"
  else
    printf '%s\n' "${selected}/PuLID_models"
  fi
}

read_configured_models_root() {
  local config_file="${PROJECT_DIR}/config/local.yaml"
  local line=""
  local value=""
  [[ -f "${config_file}" ]] || return 0
  while IFS= read -r line; do
    if [[ "${line}" == models_root:* ]]; then
      value="${line#models_root:}"
      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%"${value##*[![:space:]]}"}"
      if [[ ${#value} -ge 2 ]]; then
        if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]] ||
          [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
          value="${value:1:${#value}-2}"
        fi
      fi
      if [[ -n "${value}" && "${value}" != /* ]]; then
        value="${PROJECT_DIR}/${value}"
      fi
      printf '%s\n' "${value%/}"
      return 0
    fi
  done < "${config_file}"
}

DEFAULT_MODELS_ROOT="${PROJECT_DIR}/PuLID_models"
REQUESTED_MODELS_ROOT="${PULID_MODELS_ROOT:-}"
PULID_MODELS_ROOT=""

if [[ -n "${REQUESTED_MODELS_ROOT}" ]]; then
  REQUESTED_MODELS_ROOT="$(normalize_models_root "${REQUESTED_MODELS_ROOT}")"
  if [[ -d "${REQUESTED_MODELS_ROOT}" ]]; then
    PULID_MODELS_ROOT="${REQUESTED_MODELS_ROOT}"
  fi
fi

if [[ -z "${PULID_MODELS_ROOT}" ]]; then
  CONFIGURED_MODELS_ROOT="$(read_configured_models_root)"
  if [[ -n "${CONFIGURED_MODELS_ROOT}" && -d "${CONFIGURED_MODELS_ROOT}" ]]; then
    PULID_MODELS_ROOT="${CONFIGURED_MODELS_ROOT}"
  fi
fi

if [[ -z "${PULID_MODELS_ROOT}" && -d "${DEFAULT_MODELS_ROOT}" ]]; then
  PULID_MODELS_ROOT="${DEFAULT_MODELS_ROOT}"
fi

if [[ -n "${PULID_MODELS_ROOT}" ]]; then
  echo "Installation existante détectée : ${PULID_MODELS_ROOT}"
else
  while true; do
    read -r -p "Utiliser l'emplacement par défaut ${DEFAULT_MODELS_ROOT} ? [O/n] " USE_DEFAULT
    case "${USE_DEFAULT:-o}" in
      o|O|oui|OUI|y|Y|yes|YES)
        PULID_MODELS_ROOT="${DEFAULT_MODELS_ROOT}"
        break
        ;;
      n|N|non|NON|no|NO)
        read -r -p "Chemin du dossier parent (ou d'un dossier déjà nommé PuLID_models) : " CUSTOM_MODELS_ROOT
        if [[ -n "${CUSTOM_MODELS_ROOT}" ]]; then
          PULID_MODELS_ROOT="$(normalize_models_root "${CUSTOM_MODELS_ROOT}")"
          break
        fi
        ;;
      *) echo "Répondez oui ou non." ;;
    esac
  done
fi

mkdir -p "${PULID_MODELS_ROOT}"

export PULID_MODELS_ROOT
export HF_HOME="${PULID_MODELS_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${PULID_MODELS_ROOT}/huggingface/hub"
export TRANSFORMERS_CACHE="${PULID_MODELS_ROOT}/huggingface/transformers"
export TORCH_HOME="${PULID_MODELS_ROOT}/torch"
export XDG_CACHE_HOME="${PULID_MODELS_ROOT}/other"
export MPLCONFIGDIR="${PULID_MODELS_ROOT}/other/matplotlib"
for pulid_uv_variable in ${!UV_@}; do
  unset "${pulid_uv_variable}"
done
export UV_CACHE_DIR="${PULID_MODELS_ROOT}/other/uv-macos"
export UV_PYTHON_INSTALL_DIR="${PULID_MODELS_ROOT}/other/uv-python-macos"
export UV_LINK_MODE=copy
export NO_ALBUMENTATIONS_UPDATE=1

on_error() {
  local exit_code=$?
  echo >&2
  echo "[ERREUR] Installation macOS interrompue à la ligne ${BASH_LINENO[0]}." >&2
  echo "Corrigez l'erreur affichée ci-dessus puis relancez install_macos.sh." >&2
  exit "${exit_code}"
}
trap on_error ERR

cd "${PROJECT_DIR}"

mkdir -p \
  "${HF_HOME}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${MPLCONFIGDIR}" \
  "${UV_CACHE_DIR}" \
  "${UV_PYTHON_INSTALL_DIR}"

UV_VERSION="$(cat "${PROJECT_DIR}/.uv-version")"
PYTHON_VERSION="$(cat "${PROJECT_DIR}/.python-version")"
UV_UNMANAGED_INSTALL="${PULID_MODELS_ROOT}/other/uv-macos-bin"
export UV_UNMANAGED_INSTALL
UV_EXE="${UV_UNMANAGED_INSTALL}/uv"
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "[ERREUR] L'installation macOS requiert Apple Silicon (arm64)." >&2
  exit 1
fi
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX
export UV_PYTHON_PREFERENCE=only-managed
if [[ ! -x "${UV_EXE}" ]] || [[ "$("${UV_EXE}" --version | awk '{print $2}')" != "${UV_VERSION}" ]]; then
  mkdir -p "${UV_UNMANAGED_INSTALL}"
  curl --proto '=https' --tlsv1.2 -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
fi
[[ "$("${UV_EXE}" --version | awk '{print $2}')" == "${UV_VERSION}" ]]
"${UV_EXE}" python install "cpython-${PYTHON_VERSION}-macos-aarch64-none" --no-bin --no-registry --no-config
MANAGED_PYTHON="${UV_PYTHON_INSTALL_DIR}/cpython-${PYTHON_VERSION}-macos-aarch64-none/bin/python3.11"
"${MANAGED_PYTHON}" -I "${PROJECT_DIR}/scripts/install_environment.py" \
  --uv "${UV_EXE}" --models-root "${PULID_MODELS_ROOT}" --profile "${PULID_INSTALL_PROFILE}"
PYTHON_ARCH="arm64"

echo
echo "Installation ou réparation des modèles et configurations..."
"${VENV_PYTHON}" -I -m pulid_app.installer \
  --models-root "${PULID_MODELS_ROOT}" \
  --sdxl ask

echo "Vérification des composants Python..."
"${VENV_PYTHON}" -I -c "import diffusers, fastapi, llama_cpp, torch, transformers; info = llama_cpp.llama_print_system_info().decode(); assert 'MTL' in info, 'Backend Metal absent de llama-cpp-python'; print('Python', '${PYTHON_VERSION}', '-', '${PYTHON_ARCH}'); print('PyTorch', torch.__version__, '- MPS disponible :', torch.backends.mps.is_available()); print('llama-cpp-python', llama_cpp.__version__, '- Metal OK')"
"${VENV_PYTHON}" -I -m pulid_app.cli --version
"${VENV_PYTHON}" -I -c "from pulid_app.config import load_config; config = load_config(); embedding = config.text_embedding; assert embedding is not None; assert embedding.checkpoint.is_file(), embedding.checkpoint; print('GGUF configuré :', embedding.checkpoint)"
"${VENV_PYTHON}" -I scripts/verify_text_embedding.py --device mps
"${VENV_PYTHON}" -I -m pulid_app.cli doctor --allow-missing-sdxl
"${VENV_PYTHON}" -I scripts/inspect_models.py \
  --show-cache-env \
  --fail-on-internal-cache \
  --allow-missing-sdxl

trap - ERR
echo
echo "Installation macOS terminée."
echo "Activez l'environnement avec : source .venv/bin/activate"
echo "Tests unitaires : .venv/bin/python -m pytest"
echo "Serveur local : ./start_pulid_server.sh"
