"""Pytest configuration for JAX GPU test suite."""

import pytest
import jax


@pytest.fixture(scope="session", autouse=True)
def enable_x64():
    """Enable float64 precision (matches quantum chemistry workloads)."""
    jax.config.update("jax_enable_x64", True)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires GPU backend")
    config.addinivalue_line("markers", "stress: long-running compilation stress test")


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU tests if no GPU available."""
    try:
        gpu_available = any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        gpu_available = False

    if not gpu_available:
        skip_gpu = pytest.mark.skip(reason="No GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
