# JAX GPU Setup for HPC Cluster

<div style="flex: 1; text-align: left;">
    <img src="img/image.png" alt="feature-solid" width="420"/>
  </div>


Tested and working configuration for JAX with GPU on a cluster with NVIDIA driver 525 (CUDA 12.0 max).

I made this because running JAX on a cluster with GPUs can be a huge pain. If the hardware is not up to date etc., you will get errors. They can come at imports or during runtime, so it is hard to debug. So I made this that for sure works with CUDA 12.0 and illustrates the pitfalls that even the smallest mismathces in version can fail the whole thing (e.g., it all runs but compilation takes millenea).

The tests should pass on your environment (even if not installed with these scripts). They test differentiation through loops with eigenvolsers, basically very hard cases, they compare the outputs from CPU and GPU runs to be sure both are the same.


## Quick Start

```bash
ssh node07                        # must be on GPU node
cd /path/to/jax-cluster-setup
./install.sh                      # one-time setup
source activate.sh                # every session
pytest tests/                     # verify everything works
pytest tests/ -m "not stress"     # quick verification only
```

You can also just `source activate_jax.sh` in existing environment and run `pytest tests/` or `python test_jax_gpu.py` to verify everything works.

You might want to also see `requirements-jax-cluster-works-gpu.txt` for a list of dependencies that work together on cluster (installing `qex` library for isntance).


### Bashrc Example

I also just put into the bashrc the following to automatically set the correct environment, which will depends where are your modules installed on your cluster:

```bash

########################################################
# FIX for JAX with GPU
########################################################

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
    echo "  GREAT! ptxas 12.0 matches driver CUDA 12.0"
else
    echo "  TROUBLE! ptxas ${PTXAS_VER} — may cause slow compilation!"
fi
echo "============================================"
```

## The Problem

**Driver 525 only supports CUDA 12.0.** Using a newer CUDA toolkit (12.4, 12.8) gives XLA a `ptxas` version newer than what the driver supports. XLA detects this mismatch and **disables parallel JIT compilation**, making compilation **10-100x slower**.

Additionally, `pip install jax[cuda12]` pulls in `nvidia-cuda-nvcc-cu12` which bundles `ptxas 12.9` -- same problem even if the system toolkit is correct.

JAX 0.4.30+ switched to a plugin architecture that bundles nvidia-* pip packages with CUDA 12.6-12.9 libraries. These conflict with driver 525. So **JAX 0.4.29 is the maximum usable version**.

## The Solution

1. Use **CUDA 12.0.1** toolkit for ptxas and libdevice (matches driver)
2. Use **JAX 0.4.29** with **jaxlib 0.4.29+cuda12.cudnn91** (last standalone wheel)
3. **Disable** the pip-bundled ptxas 12.9 (rename it)
4. Runtime libraries (cuBLAS, cuSPARSE, etc.) come from pip nvidia-* wheels -- these are minor-version compatible and work fine

### Before vs After

| Metric | CUDA 12.8 ptxas | CUDA 12.0 ptxas |
|---|---|---|
| Simple JIT compile | Minutes | **~1s** |
| SCF stress test | Minutes | **~5s** |
| Parallel compilation | Disabled | **Enabled** |

## Cluster Configuration

| Component | Version | Path |
|---|---|---|
| NVIDIA Driver | 525.85.12 / 525.147.05 | - |
| Driver's max CUDA | **12.0** | - |
| CUDA Toolkit (for XLA) | **12.0.1** | `/softs/nvidia/sdk/12.0.1` |
| cuDNN | 9.10.1.4 | `/softs/nvidia/cudnn/9.10.1.4_cuda12` |
| Python | 3.11.13 | `module load devel/python/3.11.13` |
| GPU | A100-SXM4-40GB | node07 |
| JAX | **0.4.29** | pip (standalone CUDA wheel) |
| jaxlib | **0.4.29+cuda12.cudnn91** | pip -f jax_cuda_releases |

## Files

| File | Purpose |
|---|---|
| `config.sh` | All cluster-specific paths and versions -- **edit this for your cluster** |
| `install.sh` | One-time: creates venv, installs JAX, disables bad ptxas, installs test deps |
| `activate.sh` | Source each session: sets CUDA_HOME, XLA_FLAGS, PATH, LD_LIBRARY_PATH |
| `activate_jax.sh` | Source each session: sets CUDA_HOME, XLA_FLAGS, PATH, LD_LIBRARY_PATH in an existing environment |
| `diagnose.sh` | Troubleshooting: checks GPU, ptxas, env vars, libraries, JAX |
| `tests/` | pytest test suite (device, basic ops, stress compilation, CPU/GPU consistency) |

## Configuration

Edit `config.sh` to adapt to a different cluster:

```bash
CUDA_VERSION="12.0.1"                    # Must match driver's max CUDA
CUDA_SDK_BASE="/softs/nvidia/sdk"        # Where CUDA toolkits live
CUDNN_PATH="/softs/nvidia/cudnn/9.10.1.4_cuda12"
PYTHON_MODULE="devel/python/3.11.13"     # For `module load`
JAX_VERSION="0.4.29"                     # Don't change unless driver is upgraded
```

Custom venv location:
```bash
JAX_VENV_DIR=/scratch/user/.jax_venv ./install.sh
```

## Why JAX 0.4.29 (and Not Newer)

| JAX Version | Status | Reason |
|---|---|---|
| **0.4.29** | **Works** | Last version with standalone CUDA wheels |
| 0.4.30+ | Fails | Requires nvidia-* pip packages (CUDA 12.6+) that conflict |
| 0.5.x-0.6.x | Fails | Same plugin architecture, same conflict |
| 0.7.x | Fails | Requires CUDA 13 |

To use newer JAX, the cluster needs a **driver upgrade to >=535**.

## Why CUDA 12.0.1 (and Not 12.8.1)

The driver (525) reports max CUDA 12.0. When XLA finds ptxas with a higher version:

```
The NVIDIA driver's CUDA version is 12.0 which is older than the ptxas
CUDA version (12.8.xx). XLA is disabling parallel compilation, which may
slow down compilation.
```

CUDA 12.0.1 has `ptxas 12.0` which matches the driver -> parallel compilation stays enabled -> fast JIT.

The pip-installed runtime libraries (cuBLAS 12.6, cuSPARSE 12.5, etc.) are forward-compatible and work fine with driver 525. Only `ptxas` needs to match.

## Troubleshooting

Run diagnostics:
```bash
source activate.sh
./diagnose.sh
```

### Slow JIT compilation (>20s for simple ops)

**Cause:** ptxas version mismatch. Check:
```bash
ptxas --version    # Should say 12.0
```

**Fix:** Ensure `activate.sh` was sourced (puts CUDA 12.0.1 bin first in PATH). Also check that pip-bundled ptxas is disabled:
```bash
./diagnose.sh    # Check item 2 and 3
```

### `libdevice not found`

**Cause:** XLA can't find CUDA toolkit.

**Fix:** `source activate.sh` (sets `XLA_FLAGS`).

### `Unable to load cuDNN` / `cuPTI`

**Cause:** Missing from LD_LIBRARY_PATH.

**Fix:** `source activate.sh` (sets `LD_LIBRARY_PATH`).

### `Segmentation fault` on import

**Cause:** Conflicting nvidia-* pip packages.

**Fix:**
```bash
pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 \
    nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
    nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 \
    nvidia-nvjitlink-cu12
pip install jax==0.4.29 jaxlib==0.4.29+cuda12.cudnn91 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### `CUDA backend failed to initialize`

**Cause:** Not on a GPU node.

**Fix:** `ssh node07`

### JAX cache corruption

```bash
rm -rf ~/.cache/jax
```

### ptxas got reinstalled after `pip install`

If a `pip install` pulls in `nvidia-cuda-nvcc-cu12` again:
```bash
SITE=$(python3 -c 'import site; print(site.getsitepackages()[0])')
mv ${SITE}/nvidia/cuda_nvcc/bin/ptxas ${SITE}/nvidia/cuda_nvcc/bin/ptxas.disabled
```

## Installing Additional Packages

When installing packages that depend on JAX, use `--no-deps` to avoid overwriting the working JAX version:

```bash
pip install some-package --no-deps
```

Then install missing dependencies manually.

## Performance (A100-SXM4-40GB)

| Benchmark | Result |
|---|---|
| Matrix multiply (4096x4096, 10 iters) | ~104,000 GFLOPS |
| SCF stress test (64x64, 15 iters, grad) | ~5s compile |
| NN+SCF stress test (vmap+grad) | ~10s compile |
| JIT / Autodiff / vmap | All working |

## What Did NOT Work

- **CUDA 12.8.1 toolkit** -- ptxas 12.8 causes slow compilation
- **CUDA 12.4.1 toolkit** -- ptxas 12.4, same problem
- **`pip install jax[cuda12]`** -- pulls nvidia-* packages with ptxas 12.9
- **JAX >= 0.4.30** -- requires bundled CUDA 12.6+ libraries
- **JAX >= 0.7.x** -- requires CUDA 13
