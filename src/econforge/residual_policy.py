"""Frozen P3.1A constants; no optimizer or derivative settings."""

from econforge.model_errors import ModelError, real

SPEC_REVISION = "v0.1-math-freeze-1"
TOLERANCE_PROFILE = "float64_reference_v1"
SS_RESID_ATOL = 1e-10
SS_RESID_RTOL = 1e-8
MODEL_BYTES_MAX = 1048576
EXPRESSION_BYTES_MAX = 16384
EXPRESSION_NODES_MAX = 2048
EXPRESSION_DEPTH_MAX = 64
MODEL_SYMBOLS_MAX = 256
CANONICAL_ROWS_MAX = 256
NUMERIC_LITERAL_CHARS_MAX = 64


def residual_thresholds(scale: float, magnitude: float, source: str) -> tuple[float, float]:
    """The single section-6 S/L acceptance rule, with representability checks."""
    raw = real(scale*SS_RESID_ATOL + magnitude*SS_RESID_RTOL, source)
    normalized = real(SS_RESID_ATOL + (magnitude/scale)*SS_RESID_RTOL, source)
    if raw <= 0:
        raise ModelError("TOLERANCE_UNDERFLOW", "Raw acceptance threshold cannot be represented positively", source)
    return raw, normalized

