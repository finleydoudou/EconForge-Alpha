"""P3.3 bounded local root attempts with fresh scalar verification, no dynamics."""

import math
import re
from time import monotonic

from econforge.model import ValidatedModel, require_valid_model
from econforge.model_errors import Diagnostic, ModelError, real
from econforge.residuals import candidate_values
from econforge.root_backend import EvaluationSession, run_backend
from econforge.root_policy import (SolveLimits, frozen_settings, STEADY_STARTS_MAX,
                                  BRANCH_ATOL, BRANCH_RTOL, RANK_ATOL, RANK_RTOL,
                                  RANK_AMBIGUITY_FACTOR, SOLVER_STEP_ATOL, SOLVER_STEP_RTOL)
from econforge.root_results import Start, SolveAttempt, SteadyStateResult
from econforge.root_verification import inspect_termination, verify_candidate
from econforge.status import (AssessmentStatus, CheckResult, CheckStatus, EvidenceReference,
                             ExecutionStatus, aggregate_assessment)


def check(name: str, status: CheckStatus, message: str, digest: str) -> CheckResult:
    return CheckResult(name, status, message,
                       (EvidenceReference("computed_root_attempt", f"inline:{name}", digest),))


def jacobian_diagnostics(session: EvaluationSession) -> dict[str, object]:
    """Scaled singular values are diagnostics, not a uniqueness/root theorem."""
    import numpy as np
    from scipy.linalg import svdvals
    record = session.last_jacobian
    if record is None:
        return {"status": "NOT_RUN"}
    with np.errstate(all="raise"):
        matrix = np.array(record.row_normalized_jacobian, dtype=np.float64)
        matrix *= np.array([v.scale for v in session.model.variables], dtype=np.float64)
        values = svdvals(matrix, check_finite=True)
    if not np.all(np.isfinite(values)):
        raise ModelError("JACOBIAN_DIAGNOSTIC_NONFINITE", "Scaled singular values are nonfinite")
    maximum, minimum = float(values[0]), float(values[-1])
    threshold = max(RANK_ATOL, RANK_RTOL * maximum)
    resolved = minimum > RANK_AMBIGUITY_FACTOR * threshold
    condition = maximum / minimum if minimum > 0 else None
    if condition is not None and not math.isfinite(condition):
        condition = None
    return {"status": "RESOLVED_FULL_NUMERICAL_RANK" if resolved else "JACOBIAN_RANK_UNRESOLVED",
            "candidate": record.candidate, "scaled_definition": "S^-1 J diag(d)",
            "singular_values": tuple(float(x) for x in values), "tau": threshold,
            "ambiguity_interval": (threshold / RANK_AMBIGUITY_FACTOR, threshold * RANK_AMBIGUITY_FACTOR),
            "condition_2": condition, "condition_unbounded_or_unrepresentable": condition is None,
            "exact_structural_rank": "NOT_ESTABLISHED", "economic_conclusion": "NONE"}


def solve_attempt(model: ValidatedModel, start: Start, limits: SolveLimits) -> SolveAttempt:
    """One start, no retries, no hidden replacement of invalid trial residuals."""
    attempt = SolveAttempt(start.attempt_id)
    session = None
    try:
        attempt.initial_candidate = candidate_values(model, start.candidate)
        attempt.checks.append(check("initial.valid", CheckStatus.PASS, "Complete finite domain-valid stored vector", model.digest))
        session = EvaluationSession(model, attempt.initial_candidate, limits)
        output = run_backend(session)
        attempt.solver_execution_status = ExecutionStatus.COMPLETED
        inspect_termination(output, attempt, session)
        attempt.checks.append(check("backend.output", CheckStatus.PASS, "Typed, finite, consistent backend output", model.digest))
        # Nonconvergence is retained even if the independently checked point is a root.
        if not attempt.solver_converged:
            attempt.diagnostics.append(Diagnostic("BACKEND_EVALUATION_LIMIT" if output.status == 0 else "STEADY_NONCONVERGENCE", output.message))
            attempt.checks.append(check("backend.termination", CheckStatus.FAIL, output.message, model.digest))
        else:
            attempt.checks.append(check("backend.termination", CheckStatus.PASS, output.message, model.digest))
        try:
            verify_candidate(output, attempt, session)
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError("POST_SOLVE_VERIFICATION_EXCEPTION", f"{type(exc).__name__}: {exc}") from exc
        changes = tuple(real(abs((b-a)/v.scale), "scaled displacement") for a, b, v in zip(attempt.initial_candidate, attempt.solution_candidate, model.variables))
        attempt.metadata["initial_to_final_scaled_displacement"] = changes
        attempt.metadata["displacement_is_not_last_iteration_step"] = True
        accepted = [event["candidate"] for event in session.events if event["kind"] == "AD_J" and event["derivative_valid"]]
        if len(accepted) >= 2:
            delta = tuple(real(abs((b-a)/v.scale), "last AD update") for a, b, v in zip(accepted[-2], accepted[-1], model.variables))
            attempt.metadata["last_AD_update_scaled"] = delta
            magnitudes = tuple(real(abs(x/v.scale), "last AD scaled value") for x, v in zip(accepted[-1], model.variables))
            attempt.metadata["last_AD_update_small_diagnostic_only"] = all(dx <= SOLVER_STEP_ATOL + SOLVER_STEP_RTOL*value for dx, value in zip(delta, magnitudes))
        attempt.execution_status = ExecutionStatus.COMPLETED
    except ModelError as exc:
        attempt.diagnostics.append(exc.diagnostic)
        attempt.checks.append(check("attempt.failure", CheckStatus.UNSUPPORTED if exc.diagnostic.unsupported else CheckStatus.FAIL, str(exc), model.digest))
        attempt.execution_status = ExecutionStatus.NOT_RUN if exc.diagnostic.unsupported and (session is None or not session.backend_started) else ExecutionStatus.FAILED
    except Exception as exc:
        # Backend linear algebra and unexpected failures remain visible, not retries.
        code = "BACKEND_LINEAR_ALGEBRA_FAILURE" if type(exc).__name__ == "LinAlgError" else "SOLVER_EXCEPTION"
        attempt.diagnostics.append(Diagnostic(code, f"{type(exc).__name__}: {exc}"))
        attempt.checks.append(check("attempt.exception", CheckStatus.FAIL, str(attempt.diagnostics[-1]), model.digest))
        attempt.execution_status = ExecutionStatus.FAILED
    if session is not None:
        # Diagnostics cannot prevent the mandatory fresh scalar verification.
        # A diagnostic failure is retained without replacing a prior row failure.
        if session.last_jacobian is not None and attempt.solver_execution_status is ExecutionStatus.COMPLETED:
            try:
                session.check_time()
                details = jacobian_diagnostics(session)
                session.check_time()
                attempt.metadata["jacobian_diagnostics"] = details
                if details["status"] == "JACOBIAN_RANK_UNRESOLVED":
                    attempt.warnings.append("JACOBIAN_RANK_UNRESOLVED: residual root only; no invertibility or uniqueness certified")
                    attempt.checks.append(check("jacobian.rank_diagnostic", CheckStatus.WARN, attempt.warnings[-1], model.digest))
            except Exception as exc:
                diagnostic = exc.diagnostic if isinstance(exc, ModelError) else Diagnostic("JACOBIAN_DIAGNOSTIC_FAILED", f"{type(exc).__name__}: {exc}")
                attempt.diagnostics.append(diagnostic)
                attempt.checks.append(check("jacobian.diagnostic_failure", CheckStatus.FAIL, str(diagnostic), model.digest))
                attempt.execution_status = ExecutionStatus.FAILED
        if session.backend_started and attempt.solver_execution_status is ExecutionStatus.NOT_RUN:
            attempt.solver_execution_status = ExecutionStatus.FAILED
        attempt.diagnostics.extend(session.upstream_diagnostics)
        attempt.metadata.update({"residual_budget_charges": session.residual_budget_charges,
                                "backend_versions": session.runtime_versions,
                                "dtype": "float64", "platform": "cpu",
                                "observed_function_callbacks": session.fun_calls,
                                "observed_AD_callbacks": session.jac_calls,
                                "fresh_post_solve_scalar_calls": session.post_calls,
                                "budget_accounting": "one per scalar F, two per AD request (scalar admission plus differentiated primal), one per fresh post-check; includes failed dispatches",
                                "elapsed_seconds": monotonic()-session.started,
                                "evaluation_events": session.events})
        # Record the last available derivative even after failed verification.
        if "jacobian_diagnostics" not in attempt.metadata and session.last_jacobian is not None:
            attempt.metadata["last_AD_candidate"] = session.last_jacobian.candidate
    attempt.assessment_status = aggregate_assessment(tuple(attempt.checks))
    return attempt


def admit_starts(initial_guess: object, starts: object) -> tuple[Start, ...]:
    """No implicit model-reference start; multiple attempts must be explicit."""
    if initial_guess is not None and starts is not None:
        raise ModelError("AMBIGUOUS_INITIALIZATION", "Supply initial_guess OR an ordered Start list, not both")
    if starts is None:
        if initial_guess is None:
            raise ModelError("MISSING_INITIAL_GUESS", "Supply a complete explicit initial guess")
        return (Start("start_001", initial_guess),)
    if type(starts) not in (list, tuple) or not 1 <= len(starts) <= STEADY_STARTS_MAX:
        raise ModelError("INVALID_START_LIST", f"Supply 1..{STEADY_STARTS_MAX} explicit Start records")
    seen: set[str] = set()
    for start in starts:
        if type(start) is not Start or type(start.attempt_id) is not str or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", start.attempt_id):
            raise ModelError("INVALID_ATTEMPT_ID", "Each start needs a stable plain ASCII ID")
        if start.attempt_id in seen:
            raise ModelError("DUPLICATE_ATTEMPT_ID", start.attempt_id)
        seen.add(start.attempt_id)
    return tuple(starts)


def cluster_attempts(attempts: tuple[SolveAttempt, ...], scales: tuple[float, ...]) -> tuple[tuple[str, ...], ...]:
    """Frozen deterministic complete-link rule; no transitive proximity chain."""
    clusters: list[list[SolveAttempt]] = []
    for attempt in sorted(attempts, key=lambda item: item.attempt_id):
        if attempt.assessment_status not in (AssessmentStatus.PASS, AssessmentStatus.PASS_WITH_WARNINGS):
            continue
        if not attempt.independently_verified:
            raise ModelError("INCONSISTENT_VERIFICATION_STATE", "Passing attempt has no independent verification")
        point = tuple(x/d for x, d in zip(attempt.solution_candidate, scales))
        if any(not math.isfinite(x) for x in point):
            raise ModelError("BRANCH_SCALING_NONFINITE", "Cannot cluster candidate in finite scaled coordinates")
        for cluster in clusters:
            compatible = True
            for member in cluster:
                other = tuple(x/d for x, d in zip(member.solution_candidate, scales))
                differences = tuple(real(abs(a-b), "branch distance") for a, b in zip(point, other))
                if not all(delta <= BRANCH_ATOL + BRANCH_RTOL*max(abs(a), abs(b)) for delta, a, b in zip(differences, point, other)):
                    compatible = False
                    break
            if compatible:
                cluster.append(attempt)
                break
        else:
            clusters.append([attempt])
    return tuple(tuple(a.attempt_id for a in cluster) for cluster in clusters)


def solve_steady_state(model: ValidatedModel, initial_guess: object = None, *,
                       starts: list[Start] | tuple[Start, ...] | None = None,
                       selected_attempt_id: str | None = None,
                       limits: SolveLimits | None = None) -> SteadyStateResult:
    """Solve only frozen F=0; no global uniqueness or dynamic model conclusion."""
    checks: list[CheckResult] = []
    diagnostics: list[Diagnostic] = []
    warnings: list[str] = []
    attempts: tuple[SolveAttempt, ...] = ()
    clusters: tuple[tuple[str, ...], ...] = ()
    selected = solution = None
    variables = equations = ()
    metadata: dict[str, object] = {}
    execution = ExecutionStatus.NOT_RUN
    digest = ""
    try:
        require_valid_model(model)
        digest, variables, equations = model.digest, model.variable_order, model.equation_order
        limits = SolveLimits() if limits is None else limits
        if type(limits) is not SolveLimits:
            raise ModelError("INVALID_OPTIONS", "Only a typed SolveLimits record is supported")
        limits.validate()
        declared = admit_starts(initial_guess, starts)
        if selected_attempt_id is not None and (type(selected_attempt_id) is not str or selected_attempt_id not in [s.attempt_id for s in declared]):
            raise ModelError("UNKNOWN_BRANCH_SELECTION", "Select a declared attempt ID; roots are never chosen by economic preference")
        metadata.update(frozen_settings(limits))
        metadata.update({"model_hash": digest, "model_id": model.model_id, "model_branch_label": model.branch_id,
                         "x_scale": tuple(v.scale for v in model.variables),
                         "domains": tuple(v.domain for v in model.variables),
                         "row_scales": tuple((e.scale, e.magnitude) for e in model.equations),
                         "source_to_canonical_variables": model.source_to_canonical_variables,
                         "source_to_canonical_equations": model.source_to_canonical_equations,
                         "reference_environment": "NOT_ADMITTED", "global_uniqueness": "NOT_ESTABLISHED",
                         "selection_rule": "explicit attempt ID, or stable first ID in the only accepted complete-link cluster",
                         "requested_selection": selected_attempt_id})
        attempts = tuple(solve_attempt(model, start, limits) for start in declared)
        for attempt in attempts:
            checks.extend(attempt.checks)
            diagnostics.extend(attempt.diagnostics)
            warnings.extend(attempt.warnings)
        clusters = cluster_attempts(attempts, tuple(v.scale for v in model.variables))
        if len(clusters) > 1 and selected_attempt_id is None:
            diagnostics.append(Diagnostic("MULTIPLE_STEADY_BRANCHES", "Distinct admissible roots observed; select an explicit attempt ID", unsupported=True))
            checks.append(check("branch.selection", CheckStatus.UNSUPPORTED, diagnostics[-1].message, digest))
        elif clusters:
            selected = selected_attempt_id or clusters[0][0]
            chosen = next(a for a in attempts if a.attempt_id == selected)
            if not any(selected in c for c in clusters):
                raise ModelError("SELECTED_BRANCH_FAILED", "Selected attempt has no accepted root")
            if len(clusters) > 1:
                warnings.append("MULTIPLE_STEADY_BRANCHES: explicitly selected locally found root, not uniqueness")
                checks.append(check("branch.selection", CheckStatus.WARN, warnings[-1], digest))
            else:
                checks.append(check("branch.selection", CheckStatus.PASS, "One locally found cluster; no global uniqueness claim", digest))
            solution = chosen.solution_candidate
        execution = ExecutionStatus.FAILED if any(a.execution_status is ExecutionStatus.FAILED for a in attempts) else ExecutionStatus.COMPLETED
        if all(a.execution_status is ExecutionStatus.NOT_RUN for a in attempts):
            execution = ExecutionStatus.NOT_RUN
    except ModelError as exc:
        diagnostics.append(exc.diagnostic)
        checks.append(check("request.failure", CheckStatus.UNSUPPORTED if exc.diagnostic.unsupported else CheckStatus.FAIL, str(exc), digest))
        execution = ExecutionStatus.NOT_RUN if exc.diagnostic.unsupported else ExecutionStatus.FAILED
    except Exception as exc:
        diagnostics.append(Diagnostic("SOLVE_REQUEST_EXCEPTION", f"{type(exc).__name__}: {exc}"))
        checks.append(check("request.exception", CheckStatus.FAIL, str(diagnostics[-1]), digest))
        execution = ExecutionStatus.FAILED
    status = aggregate_assessment(tuple(checks))
    if status not in (AssessmentStatus.PASS, AssessmentStatus.PASS_WITH_WARNINGS):
        selected = solution = None
    metadata["selected_branch_id"] = next((c[0] for c in clusters if selected in c), None)
    metadata["observed_branch_count"] = len(clusters)
    return SteadyStateResult(status, execution, attempts, clusters, selected, solution,
                             variables, equations, tuple(checks), tuple(diagnostics), tuple(warnings), metadata)
