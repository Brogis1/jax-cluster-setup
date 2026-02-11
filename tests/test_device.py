"""Test GPU device detection."""

import pytest
import jax


@pytest.mark.gpu
def test_gpu_detected():
    """At least one GPU device must be available."""
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    assert len(gpu_devices) > 0, (
        f"No GPU detected. Devices: {devices}. "
        "Are you on a GPU node? Did you source activate.sh?"
    )


@pytest.mark.gpu
def test_default_backend_is_gpu():
    """Default backend should be GPU when GPU is available."""
    assert jax.default_backend() == "gpu"


def test_jax_version():
    """JAX version should be 0.4.29 (the compatible version for this cluster)."""
    assert jax.__version__ == "0.4.29", (
        f"Expected JAX 0.4.29, got {jax.__version__}. "
        "Other versions are incompatible with driver 525."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
