#!/bin/bash

echo "Activating JAX GPU environment, we run this in an existing venv..."

#===============================================================================
# JAX GPU Activation Script — source this to activate JAX on the cluster
#
#   source ~/activate_jax.sh
#
# PROBLEM: Driver 525 = CUDA 12.0. Any ptxas newer than 12.0 causes XLA
#          to disable parallel compilation → very slow JIT.
#
# SOLUTION: Force CUDA 12.0.1 for ptxas/libdevice. Override everything.
#===============================================================================

# --- Step 1: Load correct Python ---
module load devel/python/3.11.13 2>/dev/null

# --- Step 2: FORCE CUDA 12.0.1 for XLA (ptxas 12.0 matches driver 525) ---
# We MUST override whatever the cuda-sdk module set
export CUDA_HOME="/softs/nvidia/sdk/12.0.1"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/softs/nvidia/sdk/12.0.1"

# Put 12.0.1 ptxas FIRST in PATH, before any 12.8.1 from modules
export PATH="/softs/nvidia/sdk/12.0.1/bin:${PATH}"

# --- Step 3: Runtime libs from pip wheels + system cuDNN ---
CUDNN_ROOT="/softs/nvidia/cudnn/9.10.1.4_cuda12"
CUPTI_PATH="/softs/nvidia/sdk/12.0.1/extras/CUPTI/lib64"

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

export LD_LIBRARY_PATH="${NVIDIA_LIBS}${CUDNN_ROOT}/lib:${CUPTI_PATH}:/softs/nvidia/sdk/12.0.1/lib64:${LD_LIBRARY_PATH}"

# --- Verify ---
PTXAS_ACTUAL=$(which ptxas 2>/dev/null)
PTXAS_VER=$("${PTXAS_ACTUAL}" --version 2>/dev/null | grep -oP 'release \K[0-9.]+')

echo "============================================"
echo "  JAX GPU Environment Activated"
echo "============================================"
echo "  CUDA_HOME:    $CUDA_HOME"
echo "  XLA_FLAGS:    $XLA_FLAGS"
echo "  ptxas:        ${PTXAS_ACTUAL} (v${PTXAS_VER})"
echo "  cuDNN:        ${CUDNN_ROOT}"
echo "  Python:       $(python3 --version 2>&1)"
if [ "$PTXAS_VER" = "12.0" ]; then
    echo "  ✅ ptxas 12.0 matches driver CUDA 12.0"
else
    echo "  ⚠️  ptxas ${PTXAS_VER} — may cause slow compilation!"
fi
echo "============================================"

python3 -c "
import jax
print(f'JAX {jax.__version__}')
print(f'Devices: {jax.devices()}')
if jax.default_backend() == 'gpu':
    print('✅ GPU backend active')
else:
    print(f'⚠️  Backend: {jax.default_backend()}')
" 2>&1
