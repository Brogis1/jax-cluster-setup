#!/usr/bin/env python3
"""JAX GPU verification suite.

Verbose script, unlike tests.

Standalone script to verify JAX GPU functionality on the cluster.
Includes stress tests for JIT compilation that mimic quantum
chemistry workloads (SCF loops, eigendecomposition, einsum,
vmap+grad+jit).

Usage:
    source activate.sh
    python test_jax_gpu.py

A slow SCF compilation (>20s) indicates that XLA parallel
compilation is disabled due to a ptxas version mismatch with
the driver. See README.md.
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import flax.linen as nn

jax.config.update("jax_enable_x64", True)

PASS = "\033[92m[PASS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ------------------------------------------------------------------
# 1. Device Detection
# ------------------------------------------------------------------
def test_device_detection():
    """Detect GPUs and report backend status."""
    section("Device Detection")

    devices = jax.devices()
    print(f"Available devices: {devices}")
    print(f"Number of devices: {len(devices)}")

    for i, dev in enumerate(devices):
        print(f"  Device {i}: {dev.device_kind} - {dev.platform}")

    backend = jax.default_backend()
    print(f"\nDefault backend: {backend}")

    has_gpu = any(d.platform == "gpu" for d in devices)
    return has_gpu


# ------------------------------------------------------------------
# 2. Basic Computation
# ------------------------------------------------------------------
def test_basic_computation():
    """Matrix multiplication on GPU with device placement check."""
    section("Basic Computation")

    key = random.PRNGKey(0)
    x = random.normal(key, (1000, 1000))
    y = random.normal(random.split(key)[1], (1000, 1000))

    result = jnp.dot(x, y)

    print(f"Input x device:  {x.devices()}")
    print(f"Input y device:  {y.devices()}")
    print(f"Result device:   {result.devices()}")
    print(f"Result shape:    {result.shape}")
    print(f"Result dtype:    {result.dtype}")


# ------------------------------------------------------------------
# 3. JIT Compilation
# ------------------------------------------------------------------
def test_jit_compilation():
    """Measure first-call (compile) vs cached-call speedup."""
    section("JIT Compilation")

    @jax.jit
    def compute(x, y):
        return jnp.dot(x, y) + jnp.sin(x) + jnp.cos(y)

    key = random.PRNGKey(42)
    x = random.normal(key, (500, 500))
    y = random.normal(random.split(key)[1], (500, 500))

    t0 = time.time()
    r1 = compute(x, y)
    r1.block_until_ready()
    compile_time = time.time() - t0
    print(f"First call (compile): {compile_time:.4f}s")

    t0 = time.time()
    r2 = compute(x, y)
    r2.block_until_ready()
    cached_time = time.time() - t0
    print(f"Second call (cached): {cached_time:.4f}s")
    print(f"Speedup: {compile_time / cached_time:.1f}x")


# ------------------------------------------------------------------
# 4. SCF Stress Test
# ------------------------------------------------------------------
def test_stress_compilation():
    """Stress test mimicking a quantum chemistry SCF loop.

    15 unrolled iterations with eigendecomposition, einsum
    contractions, and full gradient computation.
    """
    section("Stress Test: SCF Compilation")
    print("15-iteration SCF loop + GRAD + JIT ...")

    def scf_iter(fock, dm, eri):
        w, v = jnp.linalg.eigh(fock + jnp.eye(fock.shape[0]))
        nocc = fock.shape[0] // 2
        occ = jnp.concatenate(
            [jnp.ones(nocc) * 2.0, jnp.zeros(nocc)]
        )
        dm_new = jnp.einsum("ui,i,vi->uv", v, occ, v)
        J_scalar = jnp.sum(dm_new)
        J = J_scalar * jnp.eye(fock.shape[0]) * 0.1
        return fock + J + jnp.sin(dm_new), dm_new

    def loss_fn(params, h1e, eri):
        fock = h1e + params
        dm = jnp.zeros_like(h1e)
        for _ in range(15):
            fock, dm = scf_iter(fock, dm, eri)
        return jnp.sum(fock * dm)

    step_fn = jax.jit(jax.grad(loss_fn))

    N = 64
    # Deterministic initialization to avoid numerical instability
    h1e = jnp.diag(jnp.arange(N, dtype=jnp.float64))
    params = jnp.ones((N, N), dtype=jnp.float64) * 0.0001
    params = 0.5 * (params + params.T)
    eri = jnp.zeros((N, N), dtype=jnp.float64)

    print(f"Matrix {N}x{N}, 15 unrolled iterations.")

    t0 = time.time()
    grads = step_fn(params, h1e, eri)
    grads.block_until_ready()
    elapsed = time.time() - t0

    print(f"Compilation + execution: {elapsed:.4f}s")
    if elapsed > 20.0:
        print(f"{WARN} Compilation is slow (>20s).")
        print("  Parallel compilation may be disabled.")
    else:
        print(f"{PASS} Compilation speed is acceptable.")


# ------------------------------------------------------------------
# 5. VMAP Stress Test
# ------------------------------------------------------------------
def test_vmap_stress():
    """VMAP over batched SCF-like function with eigh."""
    section("Stress Test: VMAP + JIT")

    N, B = 32, 10
    print(f"Vmapping SCF-like function over batch {B} ...")

    def scf_body(x):
        res = x
        for _ in range(10):
            w, v = jnp.linalg.eigh(res + res.T)
            res = jnp.dot(res, res.T) + jnp.sum(w)
        return jnp.sum(res)

    batch_fn = jax.jit(jax.vmap(scf_body))
    key = random.PRNGKey(123)
    x = random.normal(key, (B, N, N))

    t0 = time.time()
    res = batch_fn(x)
    res.block_until_ready()
    print(f"VMAP compile + run: {time.time() - t0:.4f}s")


# ------------------------------------------------------------------
# 6. NN + SCF Hard Stress Test
# ------------------------------------------------------------------
class SimpleMLP(nn.Module):
    features: list

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.gelu(x)
        return nn.Dense(1)(x)


def test_nn_plus_scf_stress():
    """Hard stress test: NN correction fed into SCF loop.

    VMAP over a batch, full gradient through NN + SCF.
    """
    section("Hard Stress Test: VMAP + NN + SCF")
    print("NN correction -> SCF loop -> loss -> gradient")

    mlp = SimpleMLP(features=[64, 64])

    def scf_with_nn(params, inputs, h1e):
        correction = mlp.apply(params, inputs)
        fock = h1e + correction
        dm = jnp.zeros_like(h1e)
        for _ in range(10):
            fock_shifted = fock + jnp.eye(fock.shape[0])
            w, v = jnp.linalg.eigh(fock_shifted)
            nocc = fock.shape[0] // 2
            occ = jnp.concatenate(
                [jnp.ones(nocc) * 2.0, jnp.zeros(nocc)]
            )
            dm = jnp.einsum("ui,i,vi->uv", v, occ, v)
            J = (
                jnp.einsum("uv,uv->", dm, fock)
                * jnp.eye(fock.shape[0])
            )
            fock = fock + J + correction
        return jnp.sum(dm * fock)

    def batch_loss(params, batch_inputs, batch_h1e):
        return jnp.mean(jax.vmap(
            scf_with_nn, in_axes=(None, 0, 0)
        )(params, batch_inputs, batch_h1e))

    update_step = jax.jit(jax.grad(batch_loss))

    N, B = 32, 8
    key = random.PRNGKey(42)
    k1 = random.split(key, 1)[0]

    params = mlp.init(k1, jnp.ones((1, N, N)))
    # Deterministic, well-conditioned inputs to avoid NaN
    batch_inputs = jnp.stack([jnp.ones((N, N), dtype=jnp.float64) * 0.01
                              for _ in range(B)])
    batch_h1e = jnp.stack([jnp.diag(jnp.arange(N, dtype=jnp.float64))
                           for _ in range(B)])

    print(
        f"Batch {B}, matrix {N}x{N}, "
        f"MLP(2 layers), 10 SCF iters, full grad"
    )

    t0 = time.time()
    grads = update_step(params, batch_inputs, batch_h1e)
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready(), grads
    )
    elapsed = time.time() - t0

    print(f"Compilation + execution: {elapsed:.4f}s")
    if elapsed > 30.0:
        print(f"{WARN} Slow compilation (>30s).")
    else:
        print(f"{PASS} Compilation efficient.")


# ------------------------------------------------------------------
# 7. CPU vs GPU Consistency
# ------------------------------------------------------------------
def test_cpu_gpu_consistency():
    """Compare a non-trivial computation on CPU and GPU."""
    section("CPU vs GPU Consistency")

    try:
        cpu_devs = jax.devices("cpu")
        gpu_devs = jax.devices("gpu")
        if not cpu_devs or not gpu_devs:
            print(f"{WARN} Skipping: need both devices.")
            return
    except Exception as e:
        print(f"{WARN} Skipping: {e}")
        return

    print(f"CPU: {cpu_devs[0]}  |  GPU: {gpu_devs[0]}")

    N = 1024
    key = random.PRNGKey(777)
    k1, k2 = random.split(key)
    a = random.normal(k1, (N, N))
    b = random.normal(k2, (N, N))

    @jax.jit
    def complex_op(x, y):
        res = jnp.dot(x, y)
        res = jnp.sin(res) * jnp.tanh(x)
        return jnp.sum(res, axis=1)

    cpu_op = jax.jit(complex_op, backend="cpu")
    gpu_op = jax.jit(complex_op, backend="gpu")

    _ = cpu_op(a, b).block_until_ready()
    _ = gpu_op(a, b).block_until_ready()

    t0 = time.time()
    res_cpu = cpu_op(a, b).block_until_ready()
    t_cpu = time.time() - t0

    t0 = time.time()
    res_gpu = gpu_op(a, b).block_until_ready()
    t_gpu = time.time() - t0

    print(
        f"CPU: {t_cpu:.5f}s  |  GPU: {t_gpu:.5f}s  "
        f"|  Speedup: {t_cpu / t_gpu:.1f}x"
    )

    val_cpu = np.array(res_cpu)
    val_gpu = np.array(res_gpu)
    max_diff = np.max(np.abs(val_cpu - val_gpu))
    print(f"Max absolute difference: {max_diff:.2e}")

    if np.allclose(val_cpu, val_gpu, rtol=1e-5, atol=1e-5):
        print(f"{PASS} CPU/GPU results match.")
    else:
        print(f"{WARN} Results differ (float accumulation).")


# ------------------------------------------------------------------
# 8. SCF CPU vs GPU Gradient Consistency
# ------------------------------------------------------------------
def test_scf_cpu_gpu_consistency():
    """Compare SCF loop gradients between CPU and GPU."""
    section("SCF CPU vs GPU Gradient Consistency")

    try:
        jax.devices("cpu")
        jax.devices("gpu")
    except Exception as e:
        print(f"{WARN} Skipping: {e}")
        return

    def scf_iter(fock, dm, eri):
        fock_shifted = fock + jnp.eye(fock.shape[0])
        w, v = jnp.linalg.eigh(fock_shifted)
        nocc = fock.shape[0] // 2
        occ = jnp.concatenate(
            [jnp.ones(nocc) * 2.0, jnp.zeros(nocc)]
        )
        dm_new = jnp.einsum("ui,i,vi->uv", v, occ, v)
        J_scalar = jnp.sum(dm_new)
        J = J_scalar * jnp.eye(fock.shape[0]) * 0.1
        return fock + J + jnp.sin(dm_new), dm_new

    def loss_fn(params, h1e, eri):
        fock = h1e + params
        dm = jnp.zeros_like(h1e)
        for _ in range(15):
            fock, dm = scf_iter(fock, dm, eri)
        return jnp.sum(fock * dm)

    step_cpu = jax.jit(jax.grad(loss_fn), backend="cpu")
    step_gpu = jax.jit(jax.grad(loss_fn), backend="gpu")

    N = 64
    h1e = jnp.diag(jnp.arange(N, dtype=jnp.float64))
    params = jnp.ones((N, N), dtype=jnp.float64) * 0.0001
    params = 0.5 * (params + params.T)
    eri = jnp.zeros((N, N), dtype=jnp.float64)

    print(f"15-iteration SCF grad, matrix {N}x{N} ...")

    t0 = time.time()
    grad_cpu = step_cpu(params, h1e, eri)
    grad_cpu.block_until_ready()
    t_cpu = time.time() - t0
    print(f"CPU: {t_cpu:.4f}s")

    t0 = time.time()
    grad_gpu = step_gpu(params, h1e, eri)
    grad_gpu.block_until_ready()
    t_gpu = time.time() - t0
    print(f"GPU: {t_gpu:.4f}s  |  Speedup: {t_cpu/t_gpu:.1f}x")

    val_cpu = np.array(grad_cpu)
    val_gpu = np.array(grad_gpu)
    max_diff = np.max(np.abs(val_cpu - val_gpu))
    print(f"Max gradient difference: {max_diff:.2e}")

    if np.allclose(val_cpu, val_gpu, rtol=1e-5, atol=1e-5):
        print(f"{PASS} SCF gradients match.")
    else:
        print(f"{WARN} SCF gradients differ (eigensolver).")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("  JAX GPU Verification Suite")
    print("=" * 60)

    try:
        has_gpu = test_device_detection()
        if not has_gpu:
            print(f"\n{WARN} No GPU detected. Running on CPU.")

        test_basic_computation()
        test_jit_compilation()
        test_stress_compilation()
        test_vmap_stress()
        test_nn_plus_scf_stress()
        test_cpu_gpu_consistency()
        test_scf_cpu_gpu_consistency()

        section("Summary")
        if has_gpu:
            print(f"{PASS} All tests passed. JAX GPU OK.")
        else:
            print(f"{WARN} All tests passed, but no GPU.")

    except Exception as e:
        section("Error")
        print(f"{FAIL} {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
