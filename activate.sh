#!/bin/bash
#===============================================================================
# activate.sh -- Source this to activate JAX GPU environment
#
# Usage:
#   source /path/to/jax-cluster-setup/activate.sh
#
# What it does:
#   1. Loads Python module
#   2. Activates the virtual environment
#   3. Forces CUDA 12.0.1 ptxas (matches driver 525's CUDA 12.0)
#   4. Sets XLA_FLAGS, CUDA_HOME, LD_LIBRARY_PATH
#   5. Verifies ptxas version and JAX GPU detection
#
# Why CUDA 12.0.1?
#   Driver 525 only supports CUDA 12.0. Using ptxas from CUDA 12.8 or pip's
#   ptxas 12.9 causes XLA to disable parallel compilation -> 10-100x slower.
#===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# --- Step 1: Load Python ---
if command -v module &>/dev/null; then
    module load "${PYTHON_MODULE}" 2>/dev/null
fi

# --- Step 2: Activate venv ---
if [ ! -d "${VENV_DIR}" ]; then
    echo "ERROR: Venv not found at ${VENV_DIR}. Run install.sh first."
    return 1 2>/dev/null || exit 1
fi
source "${VENV_DIR}/bin/activate"

# --- Step 3: Force CUDA 12.0.1 for XLA ---
# ptxas version MUST match driver's CUDA capability (12.0)
export CUDA_HOME="${CUDA_HOME}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# --- Step 4: Runtime libraries ---
# Pip nvidia-* wheels provide runtime libs (cuBLAS, cuSPARSE, etc.)
# These are minor-version compatible with driver 525.
CUPTI_PATH="${CUDA_HOME}/extras/CUPTI/lib64"

SITE_PACKAGES=$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)
NVIDIA_PATH="${SITE_PACKAGES}/nvidia"
NVIDIA_LIBS=""
if [ -d "${NVIDIA_PATH}" ]; then
    for pkg in cusparse cusolver cufft cublas cudnn cuda_runtime cuda_nvrtc nvjitlink nccl cuda_cupti curand; do
        if [ -d "${NVIDIA_PATH}/${pkg}/lib" ]; then
            NVIDIA_LIBS="${NVIDIA_PATH}/${pkg}/lib:${NVIDIA_LIBS}"
        fi
    done
fi

export LD_LIBRARY_PATH="${NVIDIA_LIBS}${CUDNN_PATH}/lib:${CUPTI_PATH}:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# --- Step 5: Verify ---
PTXAS_ACTUAL=$(which ptxas 2>/dev/null)
PTXAS_VER=$("${PTXAS_ACTUAL}" --version 2>/dev/null | grep -oP 'release \K[0-9.]+')

echo "============================================"
echo "  JAX GPU Environment Activated"
echo "============================================"
echo "  CUDA_HOME:  ${CUDA_HOME}"
echo "  ptxas:      ${PTXAS_ACTUAL} (v${PTXAS_VER})"
echo "  cuDNN:      ${CUDNN_PATH}"
echo "  Python:     $(python3 --version 2>&1)"
echo "  Venv:       ${VENV_DIR}"
if [ "${PTXAS_VER}" = "12.0" ]; then
    echo "  [OK] ptxas 12.0 matches driver CUDA 12.0"
else
    echo "  [WARN] ptxas ${PTXAS_VER} -- may cause slow compilation!"
    echo "         Expected 12.0. Check PATH or re-run install.sh."
fi
echo "============================================"

python3 -c "
import jax
print(f'JAX {jax.__version__}')
print(f'Devices: {jax.devices()}')
backend = jax.default_backend()
if backend == 'gpu':
    print('[OK] GPU backend active')
else:
    print(f'[WARN] Backend: {backend} (expected gpu)')
" 2>&1
