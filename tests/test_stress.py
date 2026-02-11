"""Stress tests for JIT compilation speed.

These tests verify that parallel compilation is working correctly.
If ptxas version mismatches the driver, XLA disables parallel compilation
and these tests will be 10-100x slower (and fail the time threshold).

Uses deterministic, well-conditioned data (same as consistency tests)
to avoid NaN issues from random inputs blowing up through SCF iterations.
"""

import time
import pytest
import jax
import jax.numpy as jnp
from jax import random


# Generous threshold -- with parallel compilation these take 1-10s.
# Without it they take minutes. 60s catches the problem with margin.
COMPILATION_TIMEOUT = 60.0


@pytest.mark.gpu
@pytest.mark.stress
def test_stress_compilation():
    """Stress test mimicking a Quantum Chemistry SCF loop.

    15 unrolled iterations with eigendecomposition, einsum
    contractions, and full gradient computation.
    """

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

    step_fn = jax.jit(jax.grad(loss_fn))

    N = 64
    # Deterministic initialization to avoid numerical instability
    h1e = jnp.diag(jnp.arange(N, dtype=jnp.float64))
    params = jnp.ones((N, N), dtype=jnp.float64) * 0.0001
    params = 0.5 * (params + params.T)
    eri = jnp.zeros((N, N), dtype=jnp.float64)

    start = time.time()
    grads = step_fn(params, h1e, eri)
    grads.block_until_ready()
    elapsed = time.time() - start

    assert grads.shape == (N, N)
    assert not jnp.any(jnp.isnan(grads)), "NaN in SCF gradients"
    assert elapsed < COMPILATION_TIMEOUT, (
        f"SCF compilation took {elapsed:.1f}s (>{COMPILATION_TIMEOUT}s). "
        "Parallel compilation is likely disabled -- check ptxas version."
    )


@pytest.mark.gpu
@pytest.mark.stress
def test_vmap_stress():
    """VMAP over batched SCF-like function with eigh."""

    N = 32
    B = 10  # Batch size

    def scf_body(x):
        res = x
        for _ in range(10):
            w, v = jnp.linalg.eigh(res + res.T)
            res = jnp.dot(res, res.T) + jnp.sum(w)
        return jnp.sum(res)

    batch_fn = jax.jit(jax.vmap(scf_body))

    key = random.PRNGKey(123)
    x = random.normal(key, (B, N, N))

    start = time.time()
    res = batch_fn(x)
    res.block_until_ready()
    elapsed = time.time() - start

    assert res.shape == (B,)
    # Note: NaN is expected -- dot(res, res.T) + sum(w) diverges by design.
    # This test measures compilation speed, not numerical correctness.
    assert elapsed < COMPILATION_TIMEOUT, (
        f"VMAP compilation took {elapsed:.1f}s (>{COMPILATION_TIMEOUT}s). "
        "Parallel compilation is likely disabled -- check ptxas version."
    )


@pytest.mark.gpu
@pytest.mark.stress
def test_nn_plus_scf_stress():
    """NN + SCF loop: VMAP + GRAD + JIT (hardest stress test).

    Requires flax. Skipped if not installed.
    """
    nn = pytest.importorskip("flax.linen")

    class SimpleMLP(nn.Module):
        features: list

        @nn.compact
        def __call__(self, x):
            for feat in self.features:
                x = nn.Dense(feat)(x)
                x = nn.gelu(x)
            return nn.Dense(1)(x)

    mlp = SimpleMLP(features=[64, 64])

    def scf_with_nn(params, inputs, h1e):
        correction = mlp.apply(params, inputs)
        fock = h1e + correction
        dm = jnp.zeros_like(h1e)
        for _ in range(10):
            w, v = jnp.linalg.eigh(fock + jnp.eye(fock.shape[0]))
            nocc = fock.shape[0] // 2
            occ = jnp.concatenate([jnp.ones(nocc) * 2.0, jnp.zeros(nocc)])
            dm = jnp.einsum('ui,i,vi->uv', v, occ, v)
            J = jnp.einsum('uv,uv->', dm, fock) * jnp.eye(fock.shape[0])
            fock = fock + J + correction
        return jnp.sum(dm * fock)

    def loss_fn(params, inputs, h1e):
        return scf_with_nn(params, inputs, h1e)

    def batch_loss(params, batch_inputs, batch_h1e):
        return jnp.mean(jax.vmap(loss_fn, in_axes=(None, 0, 0))(
            params, batch_inputs, batch_h1e))

    update_step = jax.jit(jax.grad(batch_loss))

    N = 32
    B = 8  # Batch size
    key = random.PRNGKey(42)
    k1 = random.split(key, 1)[0]

    dummy_input = jnp.ones((1, N, N))
    params = mlp.init(k1, dummy_input)

    # Deterministic, well-conditioned inputs to avoid NaN
    batch_inputs = jnp.stack([jnp.ones((N, N), dtype=jnp.float64) * 0.01
                              for _ in range(B)])
    batch_h1e = jnp.stack([jnp.diag(jnp.arange(N, dtype=jnp.float64))
                           for _ in range(B)])

    start = time.time()
    grads = update_step(params, batch_inputs, batch_h1e)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), grads)
    elapsed = time.time() - start

    # Check that gradient pytree has no NaNs
    leaves = jax.tree_util.tree_leaves(grads)
    for leaf in leaves:
        assert not jnp.any(jnp.isnan(leaf)), "NaN in NN+SCF gradients"

    assert elapsed < COMPILATION_TIMEOUT, (
        f"NN+SCF compilation took {elapsed:.1f}s (>{COMPILATION_TIMEOUT}s). "
        "Parallel compilation is likely disabled -- check ptxas version."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
