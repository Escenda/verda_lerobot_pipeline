#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-train}"  # train | resume_hub

BASE_DIR="${HOME}/lerobot_run"
VENV_DIR="${BASE_DIR}/.venv"

mkdir -p "${BASE_DIR}"
cd "${BASE_DIR}"

# Load .env if present (all variables exported)
if [ -f ".env" ]; then
  set -a
  . ".env"
  set +a
fi

LOG_FILE="${BASE_DIR}/setup_env_${MODE}.log"
echo "[INFO] Using BASE_DIR=${BASE_DIR}" | tee -a "${LOG_FILE}"
echo "[INFO] Using MODE=${MODE}" | tee -a "${LOG_FILE}"

# Prefer wheels to avoid unnecessary source builds
export PIP_PREFER_BINARY=1

# Ensure system build deps for LeRobot are present (ffmpeg/libav/cmake/gcc/venv, etc.)
echo "[INFO] Installing system build deps for LeRobot (cmake, ffmpeg, libav*, ...)" | tee -a "${LOG_FILE}"
export DEBIAN_FRONTEND=noninteractive
apt-get update >> "${LOG_FILE}" 2>&1
apt-get install -y \
  python3-venv python3-dev python3.12-venv \
  build-essential cmake pkg-config \
  ffmpeg \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libswscale-dev libswresample-dev libavfilter-dev >> "${LOG_FILE}" 2>&1

# Create or reuse venv
if [ ! -d "${VENV_DIR}" ]; then
  echo "[INFO] Creating Python venv at ${VENV_DIR}" | tee -a "${LOG_FILE}"
  python3 -m venv "${VENV_DIR}"
else
  echo "[INFO] Reusing existing venv at ${VENV_DIR}" | tee -a "${LOG_FILE}"
fi

# Activate venv
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# Install / upgrade dependencies
echo "[INFO] Installing Python dependencies..." | tee -a "${LOG_FILE}"
pip install --upgrade pip setuptools wheel 2>&1 | tee -a "${LOG_FILE}"
pip install --upgrade \
  "lerobot[smolvla]" \
  "huggingface_hub[cli]" \
  "wandb" 2>&1 | tee -a "${LOG_FILE}"

# Optional: B200(sm_100) など最新GPU向けに nightly PyTorch/cu128 を上書きする
if [ "${LEROBOT_TORCH_NIGHTLY:-}" = "1" ]; then
  echo "[INFO] Installing PyTorch nightly (cu128) for sm_100-class GPUs..." | tee -a "${LOG_FILE}"
  pip uninstall -y torchaudio 2>&1 | tee -a "${LOG_FILE}" || true
  pip install --upgrade --pre torch torchvision torchcodec\
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 2>&1 | tee -a "${LOG_FILE}"
fi

export PYTHONUNBUFFERED=1

echo "[INFO] Starting train_lerobot_entry.py (mode=${MODE})" | tee -a "${LOG_FILE}"
python train_lerobot_entry.py --mode "${MODE}"
