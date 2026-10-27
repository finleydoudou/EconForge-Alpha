"""Lazy admission of the frozen CPU AD runtime; no configuration mutation."""

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType

from econforge.ad_policy import BACKEND_VERSIONS
from econforge.model_errors import ModelError


@dataclass(frozen=True)
class ADBackend:
    jax: ModuleType
    numpy: ModuleType
    versions: tuple[tuple[str, str], ...]


def load_backend() -> ADBackend:
    """Require explicit pre-import x64/CPU/no-JIT settings and exact versions."""
    try:
        versions = tuple((name, version(name)) for name, _ in BACKEND_VERSIONS)
        if versions != BACKEND_VERSIONS:
            raise ModelError("BACKEND_VERSION_MISMATCH", repr(versions), unsupported=True)
        import jax
        import jaxlib
        import numpy
        import scipy
        import jax.numpy as jnp
    except (ImportError, PackageNotFoundError) as exc:
        raise ModelError("DEPENDENCY_MISSING", str(exc), unsupported=True) from exc
    loaded = tuple((module.__name__, module.__version__) for module in (jax, jaxlib, numpy, scipy))
    if loaded != BACKEND_VERSIONS:
        raise ModelError("BACKEND_VERSION_MISMATCH", f"Loaded modules: {loaded}", unsupported=True)
    if not jax.config.x64_enabled:
        raise ModelError("DTYPE_MISMATCH", "Set JAX_ENABLE_X64=true before importing JAX", unsupported=True)
    if not jax.config.jax_disable_jit or jax.config.jax_platforms != "cpu":
        raise ModelError("BACKEND_CONFIG_MISMATCH", "Require JAX_DISABLE_JIT=true and JAX_PLATFORMS=cpu", unsupported=True)
    if any(device.platform != "cpu" for device in jax.devices()):
        raise ModelError("BACKEND_DEVICE_MISMATCH", "CPU devices only", unsupported=True)
    return ADBackend(jax, jnp, versions)

