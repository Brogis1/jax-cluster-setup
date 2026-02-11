#!/bin/bash
#===============================================================================
# config.sh -- Cluster-specific configuration for JAX GPU setup
#
# Edit these values to match your cluster. Every other script sources this file.
#===============================================================================

# --- CUDA Configuration ---
# Must match your NVIDIA driver's max supported CUDA version.
# Driver 525 -> CUDA 12.0 max -> use CUDA 12.0.1 toolkit
CUDA_VERSION="12.0.1"
CUDA_SDK_BASE="/softs/nvidia/sdk"
CUDA_HOME="${CUDA_SDK_BASE}/${CUDA_VERSION}"

# cuDNN path (must be CUDA 12 compatible)
CUDNN_PATH="/softs/nvidia/cudnn/9.10.1.4_cuda12"

# --- Python ---
PYTHON_MODULE="devel/python/3.11.13"

# --- JAX Versions ---
# 0.4.29 is the last version with standalone CUDA wheels.
# 0.4.30+ requires bundled nvidia-* pip packages that conflict with driver 525.
JAX_VERSION="0.4.29"
JAXLIB_WHEEL="jaxlib==0.4.29+cuda12.cudnn91"
JAX_RELEASES_URL="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"

# --- NVIDIA Runtime Libraries (pip wheels) ---
# The jaxlib standalone wheel does NOT bundle CUDA runtime libs.
# We need these pip packages to provide cuBLAS 12.6, cuSPARSE 12.5, etc.
# (CUDA 12.x minor-version compatible with driver 525).
# CUDA 12.0.1 system libs are too old (jaxlib needs >= 12.1).
# NOTE: nvidia-cuda-nvcc-cu12 bundles ptxas 12.9 which we must disable after install.
NVIDIA_RUNTIME_DEPS="\
nvidia-cublas-cu12==12.6.4.1 \
nvidia-cuda-cupti-cu12==12.6.80 \
nvidia-cuda-nvcc-cu12==12.9.86 \
nvidia-cuda-nvrtc-cu12==12.6.77 \
nvidia-cuda-runtime-cu12==12.6.77 \
nvidia-cudnn-cu12==9.5.1.17 \
nvidia-cufft-cu12==11.3.0.4 \
nvidia-curand-cu12==10.3.7.77 \
nvidia-cusolver-cu12==11.7.1.2 \
nvidia-cusparse-cu12==12.5.4.2 \
nvidia-nccl-cu12==2.23.4 \
nvidia-nvjitlink-cu12==12.6.85"

# --- Virtual Environment ---
# Override with JAX_VENV_DIR env var before running install.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${JAX_VENV_DIR:-${SCRIPT_DIR}/.venv}"

# --- Test Dependencies ---
# flax 0.8.5 is the last version compatible with JAX 0.4.29.
# flax >=0.9 pulls in newer JAX and breaks everything.
TEST_DEPS="pytest flax==0.8.5"
