"""CPU vs GPU consistency tests.

Runs the same computation on both backends and compares results
to catch numerical issues or backend misconfigurations.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax import random


@pytest.mark.gpu
def test_cpu_gpu_matmul_consistency():
    """Matrix ops should produce same results on CPU and GPU."""
    try:
        cpu_devs = jax.devices('cpu')
        gpu_devs = jax.devices('gpu')
    except RuntimeError:
        pytest.skip("Need both CPU and GPU devices")

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

    cpu_op = jax.jit(complex_op, backend='cpu')
    gpu_op = jax.jit(complex_op, backend='gpu')

    # Warmup
    _ = cpu_op(a, b).block_until_ready()
    _ = gpu_op(a, b).block_until_ready()

    res_cpu = cpu_op(a, b)
    res_cpu.block_until_ready()
    res_gpu = gpu_op(a, b)
    res_gpu.block_until_ready()

    val_cpu = np.array(res_cpu)
    val_gpu = np.array(res_gpu)

    np.testing.assert_allclose(
        val_cpu, val_gpu, rtol=1e-5, atol=1e-5,
        err_msg="CPU and GPU matmul results diverge beyond tolerance"
    )


@pytest.mark.gpu
def test_scf_cpu_gpu_gradient_consistency():
    """SCF loop gradients should match between CPU and GPU."""
    try:
        jax.devices('cpu')
        jax.devices('gpu')
    except RuntimeError:
        pytest.skip("Need both CPU and GPU devices")

    def scf_iter(fock, dm, eri):
        w, v = jnp.linalg.eigh(fock + jnp.eye(fock.shape[0]))
        nocc = fock.shape[0] // 2
        occ = jnp.concatenate([jnp.ones(nocc) * 2.0, jnp.zeros(nocc)])
        dm_new = jnp.einsum('ui,i,vi->uv', v, occ, v)
        J_scalar = jnp.sum(dm_new)
        J = J_scalar * jnp.eye(fock.shape[0]) * 0.1
        fock_new = fock + J + jnp.sin(dm_new)
        return fock_new, dm_new

    def loss_fn(params, h1e, eri):
        fock = h1e + params
        dm = jnp.zeros_like(h1e)
        for _ in range(15):
            fock, dm = scf_iter(fock, dm, eri)
        return jnp.sum(fock * dm)

    step_fn_cpu = jax.jit(jax.grad(loss_fn), backend='cpu')
    step_fn_gpu = jax.jit(jax.grad(loss_fn), backend='gpu')

    N = 64
    # Deterministic initialization to avoid numerical instability
    h1e = jnp.diag(jnp.arange(N, dtype=jnp.float64))
    params = jnp.ones((N, N), dtype=jnp.float64) * 0.0001
    params = 0.5 * (params + params.T)
    eri = jnp.zeros((N, N), dtype=jnp.float64)

    grad_cpu = step_fn_cpu(params, h1e, eri)
    grad_cpu.block_until_ready()
    grad_gpu = step_fn_gpu(params, h1e, eri)
    grad_gpu.block_until_ready()

    val_cpu = np.array(grad_cpu)
    val_gpu = np.array(grad_gpu)

    # Eigensolver gradients can be sensitive -- use looser tolerance
    np.testing.assert_allclose(
        val_cpu, val_gpu, rtol=1e-4, atol=1e-4,
        err_msg="SCF gradients diverge between CPU and GPU"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
