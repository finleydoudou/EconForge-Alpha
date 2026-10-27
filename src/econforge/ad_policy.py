"""Frozen P3.2 policy: MATH_SPEC sections 7, 11 and 15, no run overrides."""

BACKEND_NAME = "jax.jacfwd"
BACKEND_VERSIONS = (
    ("jax", "0.6.2"), ("jaxlib", "0.6.2"),
    ("numpy", "2.2.6"), ("scipy", "1.15.3"),
)
DERIVATIVE_ATOL = 1e-7
DERIVATIVE_RTOL = 1e-5
FD_STEP_FACTOR = 1.0
FD_STEP_MULTIPLIERS = (0.5, 1.0, 2.0)

