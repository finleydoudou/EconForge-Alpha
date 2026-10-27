"""Inspect backend termination separately from fresh scalar root verification."""

import math
from typing import Any

from econforge.model_errors import ModelError
from econforge.residuals import candidate_values, evaluate_residuals
from econforge.residual_policy import SS_RESID_ATOL, SS_RESID_RTOL
from econforge.root_backend import EvaluationSession, backend_array, require_record_identity
from econforge.root_results import SolveAttempt


def inspect_termination(output: Any, attempt: SolveAttempt, session: EvaluationSession) -> None:
    """Do not turn malformed success flags or inconsistent counters into success."""
    import numpy as np
    from scipy.optimize import OptimizeResult
    if type(output) is not OptimizeResult:
        raise ModelError("MALFORMED_SOLVER_RESULT", "Expected pinned SciPy OptimizeResult")
    required = {"x", "fun", "jac", "cost", "optimality", "status", "message", "success", "nfev", "njev"}
    if not required <= output.keys():
        raise ModelError("MALFORMED_SOLVER_RESULT", f"Missing backend fields: {sorted(required-output.keys())}")
    if type(output.success) not in (bool, np.bool_) or type(output.status) not in (int, np.int64, np.int32):
        raise ModelError("MALFORMED_TERMINATION", "Termination status and success must be typed values")
    if output.status not in (0, 1, 2, 3, 4) or bool(output.success) != (output.status > 0) or type(output.message) is not str:
        raise ModelError("MALFORMED_TERMINATION", "Unknown/inconsistent TRF termination metadata")
    attempt.termination_code, attempt.termination_message = int(output.status), output.message
    attempt.solver_converged = bool(output.success)
    for field, expected in (("nfev", session.fun_calls), ("njev", session.jac_calls)):
        value = output[field]
        if type(value) not in (int, np.int64, np.int32) or value != expected or value < 1:
            raise ModelError("BACKEND_COUNTER_MISMATCH", f"{field} does not match observed callback calls")
    attempt.function_evaluations, attempt.jacobian_evaluations = int(output.nfev), int(output.njev)
    n = len(session.initial)
    candidate = backend_array(output.x, (n,), "returned candidate")
    attempt.candidate_finite = True
    attempt.solution_candidate = candidate_values(session.model, candidate)
    backend_array(output.fun, (n,), "returned normalized F")
    backend_array(output.jac, (n, n), "returned normalized J")
    for field in ("cost", "optimality"):
        if type(output[field]) not in (int, float, np.float64) or not math.isfinite(output[field]) or output[field] < 0:
            raise ModelError("MALFORMED_SOLVER_RESULT", f"{field} must be finite and nonnegative")
        attempt.metadata[f"backend_{field}"] = float(output[field])
    last = session.last_jacobian
    if last is None or last.candidate != candidate or tuple(tuple(r) for r in output.jac.tolist()) != last.row_normalized_jacobian:
        raise ModelError("BACKEND_JACOBIAN_MISMATCH", "Returned J is not the last validated AD J at the returned candidate")


def verify_candidate(output: Any, attempt: SolveAttempt, session: EvaluationSession) -> None:
    """Fresh P3.1A scalar call checks all original rows, including identities."""
    session.charge(1, post=True)
    session.post_calls += 1
    result = evaluate_residuals(session.model, attempt.solution_candidate)
    attempt.verification = result
    session.check_time()
    if not result.evaluation_valid:
        session.upstream_diagnostics.extend(result.diagnostics)
        raise ModelError("POST_SOLVE_EVALUATION_FAILED", "Independent scalar candidate evaluation failed")
    require_record_identity(result, session.model, attempt.solution_candidate)
    if tuple(row.equation_id for row in result.rows) != tuple(e.equation_id for e in session.model.equations):
        raise ModelError("VERIFICATION_ROW_MISMATCH", "Original rows or reporting identities were omitted/reordered")
    attempt.independently_verified = result.residuals_within_tolerance
    attempt.checks.extend(result.checks)
    if not result.residuals_within_tolerance:
        session.upstream_diagnostics.extend(result.diagnostics)
        raise ModelError("RESIDUAL_TOO_LARGE", "Backend candidate failed independent original-row acceptance")
    for actual, expected in zip(output.fun.tolist(), result.normalized_residual_vector):
        error = abs(actual - expected)
        if not math.isfinite(error) or error > SS_RESID_ATOL + SS_RESID_RTOL * abs(expected):
            raise ModelError("BACKEND_RESIDUAL_MISMATCH", "Cached backend F disagrees with fresh scalar F")
