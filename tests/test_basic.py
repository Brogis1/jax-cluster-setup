"""Test basic JAX GPU operations: matmul, JIT compilation."""

import time
import pytest
import jax
import jax.numpy as jnp
from jax import random


@pytest.mark.gpu
def test_basic_computation():
    """Matrix multiplication on GPU with device placement."""
    key = random.PRNGKey(0)
    x = random.normal(key, (1000, 1000))
    y = random.normal(random.split(key)[1], (1000, 1000))
    result = jnp.dot(x, y)
    result.block_until_ready()

    assert result.shape == (1000, 1000)
    assert not jnp.any(jnp.isnan(result)), "NaN in matmul result"


@pytest.mark.gpu
def test_jit_compilation():
    """JIT compilation: cached calls should be faster."""

    @jax.jit
    def compute(x, y):
        return jnp.dot(x, y) + jnp.sin(x) + jnp.cos(y)

    key = random.PRNGKey(42)
    x = random.normal(key, (500, 500))
    y = random.normal(random.split(key)[1], (500, 500))

    # First call (compilation)
    start = time.time()
    result1 = compute(x, y)
    result1.block_until_ready()
    compile_time = time.time() - start

    # Second call (cached)
    start = time.time()
    result2 = compute(x, y)
    result2.block_until_ready()
    cached_time = time.time() - start

    assert not jnp.any(jnp.isnan(result2))
    assert cached_time < compile_time, (
        f"Cached call ({cached_time:.4f}s) should be faster "
        f"than first call ({compile_time:.4f}s)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
