#!/bin/bash
#===============================================================================
# diagnose.sh -- Troubleshooting diagnostics for JAX GPU setup
#
# Usage:
#   source activate.sh  # activate first
#   ./diagnose.sh        # then run diagnostics
#
# Can also run without activation to diagnose broken environments.
#===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "========================================="
echo "  JAX CUDA Diagnostics"
echo "========================================="
echo ""

# 1. GPU
echo "1. GPU Status:"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null
    MAX_CUDA=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    echo "   Driver's max CUDA: ${MAX_CUDA}"
else
    echo "   nvidia-smi not found -- not on a GPU node?"
fi
echo ""

# 2. ptxas (THE critical check)
echo "2. ptxas Version (CRITICAL):"
PTXAS_PATH=$(which ptxas 2>/dev/null)
if [ -n "$PTXAS_PATH" ]; then
    PTXAS_VER=$(${PTXAS_PATH} --version 2>/dev/null | grep -oP 'release \K[0-9.]+')
    echo "   Path:    ${PTXAS_PATH}"
    echo "   Version: ${PTXAS_VER}"
    if [ "$PTXAS_VER" = "12.0" ]; then
        echo "   [OK] Matches driver CUDA 12.0"
    else
        echo "   [FAIL] Version ${PTXAS_VER} != 12.0 -- XLA will disable parallel compilation!"
        echo "   Fix: Ensure ${CUDA_HOME}/bin is first in PATH"
    fi
else
    echo "   [FAIL] ptxas not found in PATH"
fi
echo ""

# 3. pip-bundled ptxas
echo "3. Pip-bundled ptxas (should be disabled):"
if [ -n "$VIRTUAL_ENV" ]; then
    SITE_PACKAGES=$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)
else
    SITE_PACKAGES=$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null)
fi
PIP_PTXAS="${SITE_PACKAGES}/nvidia/cuda_nvcc/bin/ptxas"
PIP_PTXAS_BAK="${PIP_PTXAS}.disabled"
if [ -f "$PIP_PTXAS" ]; then
    echo "   [FAIL] Active pip ptxas found at ${PIP_PTXAS}"
    echo "   Fix: mv ${PIP_PTXAS} ${PIP_PTXAS}.disabled"
elif [ -f "$PIP_PTXAS_BAK" ]; then
    echo "   [OK] Disabled (${PIP_PTXAS_BAK})"
else
    echo "   [OK] Not installed"
fi
echo ""

# 4. Environment variables
echo "4. Environment Variables:"
echo "   CUDA_HOME:    ${CUDA_HOME:-NOT SET}"
echo "   XLA_FLAGS:    ${XLA_FLAGS:-NOT SET}"
echo "   VIRTUAL_ENV:  ${VIRTUAL_ENV:-NOT SET}"
if [ -n "$CUDA_HOME" ]; then
    if [[ "$CUDA_HOME" == *"12.0"* ]]; then
        echo "   [OK] CUDA_HOME points to 12.0.x"
    else
        echo "   [WARN] CUDA_HOME does not point to 12.0.x"
    fi
fi
echo ""

echo "   LD_LIBRARY_PATH (first 5 entries):"
echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | head -5 | while read p; do echo "     $p"; done
echo ""

# 5. CUDA toolkit
echo "5. CUDA Toolkit:"
echo "   Expected: ${CUDA_HOME}"
if [ -d "${CUDA_HOME}" ]; then
    echo "   [OK] Directory exists"
    if [ -f "${CUDA_HOME}/nvvm/libdevice/libdevice.10.bc" ]; then
        echo "   [OK] libdevice.10.bc found"
    else
        echo "   [FAIL] libdevice.10.bc missing -- XLA will fail"
    fi
else
    echo "   [FAIL] Directory not found"
fi
echo ""

# 6. cuDNN
echo "6. cuDNN:"
if [ -d "${CUDNN_PATH}" ]; then
    echo "   [OK] Found at ${CUDNN_PATH}"
else
    echo "   [FAIL] Not found at ${CUDNN_PATH}"
fi
echo ""

# 7. Python and JAX
echo "7. Python:"
echo "   Path:    $(which python3 2>/dev/null)"
echo "   Version: $(python3 --version 2>&1)"
echo ""

echo "8. JAX Installation:"
python3 -c "
try:
    import jax
    print(f'   JAX version:    {jax.__version__}')
    import jaxlib
    print(f'   jaxlib version: {jaxlib.__version__}')
except ImportError as e:
    print(f'   [FAIL] {e}')
" 2>&1
echo ""

echo "9. JAX Devices:"
python3 -c "
import jax
try:
    devices = jax.devices()
    print(f'   Devices:  {devices}')
    print(f'   Backend:  {jax.default_backend()}')
    gpus = [d for d in devices if d.platform == 'gpu']
    if gpus:
        print(f'   [OK] {len(gpus)} GPU(s) detected')
    else:
        print(f'   [FAIL] No GPUs detected')
except Exception as e:
    print(f'   [FAIL] {e}')
" 2>&1
echo ""

echo "10. Critical Libraries:"
python3 -c "
import ctypes
libs = [
    'libcudart.so.12',
    'libcublas.so.12',
    'libcudnn.so.9',
    'libcupti.so.12',
    'libcusparse.so.12',
    'libcusolver.so.11',
]
for lib in libs:
    try:
        ctypes.CDLL(lib)
        print(f'    [OK]   {lib}')
    except OSError:
        print(f'    [FAIL] {lib} -- not found in LD_LIBRARY_PATH')
" 2>&1
echo ""

echo "11. Installed nvidia-* pip packages:"
pip list 2>/dev/null | grep nvidia || echo "   (none)"
echo ""

echo "========================================="
echo "  Diagnosis complete"
echo "========================================="
