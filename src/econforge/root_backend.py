"""Pinned TRF adapter using existing scalar F and AD J, without fallback."""

import hashlib
import json
import math
from time import monotonic
from typing import Any

from econforge.ad_backend import load_backend
from econforge.model import ValidatedModel
from econforge.model_errors import ModelError, real
from econforge.residuals import candidate_values, evaluate_residuals
from econforge.steady_jacobian import evaluate_steady_jacobian
from econforge.root_policy import (SolveLimits, OPTIMIZER_FTOL, OPTIMIZER_XTOL,
                                  OPTIMIZER_GTOL, ROOT_METHOD, ROOT_LOSS, ROOT_TR_SOLVER)


def require_record_identity(record: Any, model: ValidatedModel, x: tuple[float, ...]) -> None:
    """Reject stale F/J artifacts even when numeric dimensions happen to agree."""
    if record.variable_order != model.variable_order or record.equation_order != model.equation_order:
        raise ModelError("SOLVER_AXIS_MISMATCH", "F/J axes differ from canonical model identities")
    if record.candidate != x:
        raise ModelError("STALE_CANDIDATE_METADATA", "F/J record belongs to another candidate")
    digest = hashlib.sha256((model.digest + json.dumps(x, allow_nan=False)).encode()).hexdigest()
    expected = {"model_hash": model.digest, "evaluation_hash": digest,
                "parameters": model.parameters,
                "row_scales": tuple((e.scale, e.magnitude) for e in model.equations),
                "verification_equation_order": tuple(e.equation_id for e in model.equations),
                "source_to_canonical_variables": model.source_to_canonical_variables,
                "canonical_to_source_variables": model.canonical_to_source_variables,
                "source_to_canonical_equations": model.source_to_canonical_equations,
                "canonical_to_source_equations": model.canonical_to_source_equations}
    metadata = dict(record.metadata)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ModelError("STALE_MODEL_METADATA", "F/J hash, calibration, scales or permutation mismatch")


def backend_array(value: Any, shape: tuple[int, ...], label: str) -> tuple:
    """Require an actual finite real float64 backend array before conversion."""
    import numpy as np
    if type(value) is not np.ndarray:
        raise ModelError("MALFORMED_BACKEND_ARRAY", f"{label} must be a NumPy ndarray")
    if value.shape != shape:
        raise ModelError("SOLVER_DIMENSION_MISMATCH", f"{label}: expected {shape}, got {value.shape}")
    if value.dtype.kind == "c":
        raise ModelError("COMPLEX_BACKEND_VALUE", f"{label} requires real values")
    if value.dtype != np.dtype("float64"):
        raise ModelError("BACKEND_DTYPE_MISMATCH", f"{label} must be float64")
    for item in value.reshape(-1).tolist():
        real(item, label)
    return tuple(value.tolist())


class EvaluationSession:
    """Per-attempt counters and callbacks; no shared cache or model mutation."""

    def __init__(self, model: ValidatedModel, initial: tuple[float, ...], limits: SolveLimits):
        self.model, self.initial, self.limits = model, initial, limits
        self.started = monotonic()
        self.residual_budget_charges = 0
        self.fun_calls = self.jac_calls = self.post_calls = 0
        self.events: list[dict[str, object]] = []
        self.upstream_diagnostics: list = []
        self.last_jacobian = None
        self.backend_started = False
        self.runtime_versions: tuple[tuple[str, str], ...] = ()

    def check_time(self) -> None:
        if monotonic() - self.started >= self.limits.wall_seconds:
            raise ModelError("TIMEOUT", "Steady attempt wall budget exceeded; no late result accepted")

    def charge(self, count: int, *, post: bool = False) -> None:
        """Reserve one scalar post-check; charge AD primal plus scalar admission."""
        self.check_time()
        reserve = 0 if post else 1
        if self.residual_budget_charges + count + reserve > self.limits.residual_calls:
            raise ModelError("RESIDUAL_EVALUATION_LIMIT", "Total residual/diagnostic budget exhausted")
        self.residual_budget_charges += count

    def residual(self, value: Any) -> Any:
        import numpy as np
        x = candidate_values(self.model, backend_array(value, (len(self.initial),), "trial candidate"))
        if self.fun_calls == 0 and x != self.initial:
            raise ModelError("BACKEND_INITIAL_GUESS_CHANGED", "TRF adjusted the supplied start; attempt refused without evaluating a repaired guess", unsupported=True)
        self.charge(1)
        self.fun_calls += 1
        result = evaluate_residuals(self.model, x)
        self.events.append({"kind": "scalar_F", "candidate": x, "evaluation_valid": result.evaluation_valid,
                            "residual_vector": result.residual_vector})
        self.check_time()
        if not result.evaluation_valid:
            self.upstream_diagnostics.extend(result.diagnostics)
            raise ModelError("RESIDUAL_EVALUATION_FAILED", "Scalar F failed at a required trial")
        require_record_identity(result, self.model, x)
        array = np.array(result.normalized_residual_vector, dtype=np.float64)
        backend_array(array, (len(self.initial),), "normalized residual")
        return array

    def jacobian(self, value: Any) -> Any:
        import numpy as np
        x = candidate_values(self.model, backend_array(value, (len(self.initial),), "AD trial candidate"))
        # P3.2 performs one scalar F admission and one differentiated primal.
        # This is a conservative budget charge even if AD refuses before tracing.
        self.charge(2)
        self.jac_calls += 1
        result = evaluate_steady_jacobian(self.model, x)
        self.events.append({"kind": "AD_J", "candidate": x, "derivative_valid": result.derivative_valid})
        self.check_time()
        if not result.derivative_valid:
            self.upstream_diagnostics.extend(result.diagnostics)
            raise ModelError("AD_JACOBIAN_FAILED", "Required production AD evaluation failed; no fallback")
        require_record_identity(result, self.model, x)
        n = len(self.initial)
        if result.shape != (n, n) or result.dtype != "float64":
            raise ModelError("JACOBIAN_SHAPE_DTYPE_MISMATCH", "AD shape/dtype metadata is inconsistent")
        raw, normalized = result.jacobian, result.row_normalized_jacobian
        if type(raw) is not tuple or type(normalized) is not tuple or len(raw) != n or len(normalized) != n:
            raise ModelError("JACOBIAN_SHAPE_MISMATCH", "AD matrix row count mismatch")
        for row, scaled, equation in zip(raw, normalized, (e for e in self.model.equations if e.kind != "identity")):
            if type(row) is not tuple or type(scaled) is not tuple or len(row) != n or len(scaled) != n:
                raise ModelError("JACOBIAN_SHAPE_MISMATCH", "AD matrix column count mismatch")
            for value_raw, value_scaled in zip(row, scaled):
                if real(value_scaled, equation.equation_id) != real(value_raw, equation.equation_id) / equation.scale:
                    raise ModelError("JACOBIAN_SCALING_MISMATCH", "AD normalized matrix must equal S^-1 J")
        self.last_jacobian = result
        array = np.array(normalized, dtype=np.float64)
        backend_array(array, (n, n), "normalized Jacobian")
        return array


def run_backend(session: EvaluationSession) -> Any:
    """Single approved call; a callable jac prevents SciPy numerical differencing."""
    runtime = load_backend()
    session.runtime_versions = runtime.versions
    import numpy as np
    from scipy.optimize import least_squares
    lower, upper = [], []
    for variable in session.model.variables:
        lo, hi = variable.domain
        # Positivity for level log-deviation coordinates is already a P3.1A rule.
        if variable.coordinate == "log_deviation":
            lo = max(0.0, lo) if lo is not None else 0.0
        lower.append(-np.inf if lo is None else lo)
        upper.append(np.inf if hi is None else hi)
    session.check_time()
    session.backend_started = True
    with np.errstate(all="raise"):
        result = least_squares(session.residual, np.array(session.initial, dtype=np.float64),
                               jac=session.jacobian, bounds=(lower, upper), method=ROOT_METHOD,
                               loss=ROOT_LOSS, tr_solver=ROOT_TR_SOLVER,
                               x_scale=np.array([v.scale for v in session.model.variables], dtype=np.float64),
                               ftol=OPTIMIZER_FTOL, xtol=OPTIMIZER_XTOL, gtol=OPTIMIZER_GTOL,
                               max_nfev=session.limits.backend_evaluations, verbose=0)
    session.check_time()
    return result
