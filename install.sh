#!/bin/bash
#===============================================================================
# install.sh -- One-time JAX GPU installation
#
# Usage:
#   ssh node07
#   ./install.sh              # fresh install (venv at .venv/)
#   ./install.sh --reuse      # reuse existing venv
#   JAX_VENV_DIR=/path ./install.sh  # custom venv location
#
# This script:
#   1. Checks for GPU access (must run on GPU node)
#   2. Creates a virtual environment
#   3. Installs JAX 0.4.29 with CUDA 12 support
#   4. Disables pip-bundled ptxas (prevents slow compilation)
#   5. Installs test dependencies (pytest, flax)
#   6. Verifies the installation
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# --- Parse arguments ---
REUSE_VENV=false
while [[ "$1" ]]; do
    case "$1" in
        --reuse) REUSE_VENV=true ;;
        --help|-h)
            echo "Usage: ./install.sh [--reuse]"
            echo "  --reuse   Reuse existing venv instead of creating a new one"
            echo ""
            echo "Set JAX_VENV_DIR to change venv location (default: .venv/ in repo)"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# --- Pre-flight checks ---

check_gpu_node() {
    log_info "Checking for GPU access..."
    if ! command -v nvidia-smi &>/dev/null; then
        log_error "nvidia-smi not found. Are you on a GPU node? (ssh node07)"
        exit 1
    fi
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [ -z "$DRIVER_VERSION" ]; then
        log_error "Cannot query GPU. Are you on a GPU node?"
        exit 1
    fi
    log_success "GPU: ${GPU_NAME} (driver ${DRIVER_VERSION})"
}

check_cuda_toolkit() {
    log_info "Checking CUDA toolkit at ${CUDA_HOME}..."
    if [ ! -d "${CUDA_HOME}" ]; then
        log_error "CUDA toolkit not found at ${CUDA_HOME}"
        log_info "Available toolkits:"
        ls -1 "${CUDA_SDK_BASE}" 2>/dev/null | while read v; do echo "  ${CUDA_SDK_BASE}/${v}"; done
        exit 1
    fi
    if [ ! -f "${CUDA_HOME}/bin/ptxas" ]; then
        log_error "ptxas not found in ${CUDA_HOME}/bin/"
        exit 1
    fi
    PTXAS_VER=$("${CUDA_HOME}/bin/ptxas" --version 2>/dev/null | grep -oP 'release \K[0-9.]+')
    log_success "CUDA ${CUDA_VERSION} toolkit found (ptxas ${PTXAS_VER})"
}

check_cudnn() {
    log_info "Checking cuDNN at ${CUDNN_PATH}..."
    if [ ! -d "${CUDNN_PATH}" ]; then
        log_error "cuDNN not found at ${CUDNN_PATH}"
        exit 1
    fi
    log_success "cuDNN found"
}

load_python() {
    log_info "Loading Python module ${PYTHON_MODULE}..."
    if command -v module &>/dev/null; then
        module load "${PYTHON_MODULE}" 2>/dev/null
    fi
    PYTHON_VER=$(python3 --version 2>&1)
    log_success "${PYTHON_VER}"
}

# --- Installation ---

create_venv() {
    if [ -d "${VENV_DIR}" ]; then
        if [ "${REUSE_VENV}" = true ]; then
            log_info "Reusing existing venv at ${VENV_DIR}"
            source "${VENV_DIR}/bin/activate"
            pip install --upgrade pip -q
            return
        else
            log_error "Venv already exists at ${VENV_DIR}"
            log_info "Use --reuse to keep it, or remove it first:"
            log_info "  rm -rf ${VENV_DIR}"
            exit 1
        fi
    fi
    log_info "Creating virtual environment at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip -q
    log_success "Virtual environment created"
}

install_jax() {
    log_info "Installing JAX ${JAX_VERSION} with CUDA 12 support..."
    pip install "jax==${JAX_VERSION}" "${JAXLIB_WHEEL}" \
        -f "${JAX_RELEASES_URL}"
    log_success "JAX installed"
}

install_nvidia_runtime() {
    # The jaxlib standalone wheel does NOT bundle CUDA runtime libs.
    # CUDA 12.0.1 system libs are too old (jaxlib needs >= 12.1 for cuBLAS, cuSPARSE).
    # These pip packages provide cuBLAS 12.6, cuSPARSE 12.5, etc. which are
    # CUDA 12.x minor-version compatible with driver 525.
    log_info "Installing NVIDIA runtime libraries (pip wheels)..."
    pip install ${NVIDIA_RUNTIME_DEPS} -q
    log_success "NVIDIA runtime libraries installed"
}

neutralize_pip_ptxas() {
    # The nvidia-cuda-nvcc-cu12 package bundles ptxas 12.9 which causes
    # XLA to disable parallel compilation on driver 525 (CUDA 12.0 max).
    # Renaming it forces XLA to use the system ptxas 12.0 instead.
    log_info "Checking for pip-bundled ptxas (must be disabled)..."
    SITE_PACKAGES=$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)
    PIP_PTXAS="${SITE_PACKAGES}/nvidia/cuda_nvcc/bin/ptxas"
    if [ -f "${PIP_PTXAS}" ]; then
        mv "${PIP_PTXAS}" "${PIP_PTXAS}.disabled"
        log_success "Disabled pip-bundled ptxas at ${PIP_PTXAS}"
    else
        log_success "No pip-bundled ptxas found (good)"
    fi
}

install_test_deps() {
    log_info "Installing test dependencies: ${TEST_DEPS}..."
    # Constrain jax so pip doesn't upgrade it when resolving flax dependencies
    pip install ${TEST_DEPS} "jax==${JAX_VERSION}" -q \
        -f "${JAX_RELEASES_URL}"

    # Verify JAX wasn't upgraded
    INSTALLED_JAX=$(python3 -c "import jax; print(jax.__version__)" 2>/dev/null)
    if [ "${INSTALLED_JAX}" != "${JAX_VERSION}" ]; then
        log_error "JAX was upgraded to ${INSTALLED_JAX} by test dependencies!"
        log_error "Expected ${JAX_VERSION}. Fixing..."
        pip install "jax==${JAX_VERSION}" "${JAXLIB_WHEEL}" \
            -f "${JAX_RELEASES_URL}" --force-reinstall --no-deps -q
    fi
    log_success "Test dependencies installed"
}

verify_jax_version() {
    # Final safety check: ensure JAX version is correct
    INSTALLED_JAX=$(python3 -c "import jax; print(jax.__version__)" 2>/dev/null)
    if [ "${INSTALLED_JAX}" != "${JAX_VERSION}" ]; then
        log_error "JAX version is ${INSTALLED_JAX}, expected ${JAX_VERSION}"
        log_error "Something upgraded JAX. Reinstalling..."
        pip install "jax==${JAX_VERSION}" "${JAXLIB_WHEEL}" \
            -f "${JAX_RELEASES_URL}" --force-reinstall --no-deps -q
    fi
}

# --- Verification ---

verify() {
    log_info "Verifying installation..."

    # Source activate to set all env vars
    source "${SCRIPT_DIR}/activate.sh"

    python3 -c "
import jax
import jax.numpy as jnp

print(f'  JAX version:  {jax.__version__}')
devices = jax.devices()
print(f'  Devices:      {devices}')
print(f'  Backend:      {jax.default_backend()}')

gpu_devices = [d for d in devices if d.platform == 'gpu']
assert gpu_devices, 'No GPU detected! Check that you are on a GPU node.'

# Quick computation test
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (500, 500))
result = jnp.dot(x, x).block_until_ready()
print(f'  GPU matmul:   OK ({result.shape})')
"
    log_success "Installation verified -- GPU is working"
}

# --- Main ---

main() {
    echo ""
    echo "========================================================================"
    echo "  JAX Cluster Setup -- Install"
    echo "========================================================================"
    echo "  JAX:    ${JAX_VERSION}"
    echo "  CUDA:   ${CUDA_VERSION} (at ${CUDA_HOME})"
    echo "  cuDNN:  ${CUDNN_PATH}"
    echo "  Venv:   ${VENV_DIR}"
    echo "========================================================================"
    echo ""

    check_gpu_node
    check_cuda_toolkit
    check_cudnn
    load_python
    create_venv
    install_jax
    install_nvidia_runtime
    neutralize_pip_ptxas
    install_test_deps
    verify_jax_version
    neutralize_pip_ptxas   # re-check after test deps (flax may pull nvidia packages)
    verify

    echo ""
    echo "========================================================================"
    echo "  INSTALLATION COMPLETE"
    echo "========================================================================"
    echo ""
    echo "  To activate (every session):"
    echo "    source ${SCRIPT_DIR}/activate.sh"
    echo ""
    echo "  To run tests:"
    echo "    source ${SCRIPT_DIR}/activate.sh"
    echo "    pytest ${SCRIPT_DIR}/tests/"
    echo ""
    echo "  To run quick tests only (skip stress tests):"
    echo "    pytest ${SCRIPT_DIR}/tests/ -m 'not stress'"
    echo ""
}

main
