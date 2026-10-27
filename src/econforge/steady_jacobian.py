"""First AD derivative of substituted F at supplied x; no search or dynamics."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

from econforge.ad_backend import ADBackend, load_backend
from econforge.ad_policy import BACKEND_NAME
from econforge.ast_jax import LoweredSteadyResidual, lower_steady_ast
from econforge.model import ValidatedModel
from econforge.model_errors import Diagnostic, ModelError
from econforge.residual_policy import SS_RESID_ATOL, SS_RESID_RTOL
from econforge.residuals import evaluate_residuals
from econforge.status import (
    AssessmentStatus, CheckResult, CheckStatus, EvidenceReference, ExecutionStatus,
    ModelStatus, CapabilityMaturity, aggregate_assessment,
)

Matrix = tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class JacobianResult:
    assessment_status: AssessmentStatus
    execution_status: ExecutionStatus
    ad_execution_status: ExecutionStatus
    derivative_valid: bool
    jacobian: Matrix | None
    row_normalized_jacobian: Matrix | None
    shape: tuple[int, int] | None
    equation_order: tuple[str, ...]
    variable_order: tuple[str, ...]
    candidate: tuple[float, ...] | None
    residual_vector: tuple[float, ...] | None
    backend: str
    dtype: str | None
    checks: tuple[CheckResult, ...]
    diagnostics: tuple[Diagnostic, ...]
    metadata: tuple[tuple[str, object], ...]
    warnings: tuple[str, ...] = ()
    independent_validation_status: CheckStatus = CheckStatus.NOT_RUN
    benchmark_status: CheckStatus = CheckStatus.NOT_RUN
    model_status: ModelStatus = ModelStatus.NOT_EVALUATED
    capability_maturity: CapabilityMaturity = CapabilityMaturity.IMPLEMENTED
    schema_version: int = 2
    claim: str = "steady_residual_AD_Jacobian_point_checks"

    @property
    def status(self) -> AssessmentStatus:
        return self.assessment_status

    @property
    def failure_reason(self) -> str | None:
        return self.diagnostics[0].code if self.diagnostics else None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["checks"] = [check.to_dict() for check in self.checks]
        payload["metadata"] = dict(self.metadata)
        payload["failure_reason"] = self.failure_reason
        return payload


def checked_array(value: Any, shape: tuple[int, ...], label: str,
                  rows: tuple[str, ...] = (), columns: tuple[str, ...] = ()) -> Any:
    """Validate backend data BEFORE conversion; no casts or repaired entries."""
    if tuple(value.shape) != shape:
        raise ModelError("AD_SHAPE_MISMATCH", f"{label}: expected {shape}, got {value.shape}")
    dtype = str(value.dtype)
    if dtype.startswith("complex"):
        raise ModelError("AD_COMPLEX_VALUE", f"{label} requires real float64")
    if dtype != "float64":
        raise ModelError("DTYPE_MISMATCH", f"{label}: expected float64, got {dtype}")
    entries = value.reshape(-1).tolist()
    for i, item in enumerate(entries):
        if not math.isfinite(item):
            code = "AD_NAN_VALUE" if math.isnan(item) else "AD_INF_VALUE"
            source = rows[i // len(columns)] if rows and columns else label
            symbol = columns[i % len(columns)] if columns else None
            raise ModelError(code, f"{label} flat index {i} is nonfinite", source, symbol=symbol)
    return value


def execute_ad(model: ValidatedModel, candidate: tuple[float, ...],
               runtime: ADBackend) -> tuple[LoweredSteadyResidual, Any, Any, Any]:
    """Execute exactly one first jacfwd transform; validity is checked separately."""
    lowered = lower_steady_ast(model, runtime)
    if lowered.equation_order != model.equation_order:
        raise ModelError("AD_ROW_ORDER_MISMATCH", "Lowered rows differ from canonical equation IDs")
    if lowered.variable_order != model.variable_order:
        raise ModelError("AD_COLUMN_ORDER_MISMATCH", "Lowered columns differ from canonical symbol IDs")
    n = len(model.variable_order)
    x = runtime.numpy.asarray(candidate, dtype=runtime.numpy.float64)
    checked_array(runtime.jax.device_get(x), (n,), "candidate")
    matrix, (ad_primal, intermediates) = runtime.jax.jacfwd(lowered.function, has_aux=True)(x)
    return lowered, runtime.jax.device_get(matrix), runtime.jax.device_get(ad_primal), runtime.jax.device_get(intermediates)


def verify_ad_output(model: ValidatedModel, primal: tuple[float, ...], runtime: ADBackend,
                     output: tuple[LoweredSteadyResidual, Any, Any, Any]) -> tuple[Matrix, Matrix, tuple[tuple[str, object], ...]]:
    """Independent point checks after successful AD execution; no output repairs."""
    lowered, matrix, ad_primal, trace = output
    m, n = len(model.equation_order), len(model.variable_order)
    host_matrix = checked_array(matrix, (m, n), "Jacobian", model.equation_order, model.variable_order)
    host_primal = checked_array(ad_primal, (m,), "AD primal")
    # Give source spans for lowered intermediate failures, including cancellation.
    if tuple(trace.shape) != (len(lowered.nodes),):
        raise ModelError("AD_SHAPE_MISMATCH", "Intermediate trace shape mismatch")
    if str(trace.dtype) != "float64":
        raise ModelError("DTYPE_MISMATCH", "Intermediate trace must be float64")
    for node, value in zip(lowered.nodes, trace.tolist()):
        if not math.isfinite(value):
            raise ModelError("AD_INTERMEDIATE_NONFINITE", "Lowered primal intermediate is nonfinite",
                             node.source, start=node.start, end=node.end)
    scales = tuple(e.scale for e in model.equations if e.kind != "identity")
    discrepancies = []
    for i, (value, reference, scale) in enumerate(zip(host_primal.tolist(), primal, scales)):
        difference = abs(value / scale - reference / scale)
        threshold = SS_RESID_ATOL + SS_RESID_RTOL * abs(reference / scale)
        if not math.isfinite(difference) or difference > threshold:
            raise ModelError("AD_PRIMAL_MISMATCH", "JAX and scalar normalized F disagree", model.equation_order[i])
        discrepancies.append(difference)
    raw = tuple(tuple(row) for row in host_matrix.tolist())
    normalized = tuple(tuple(item / scale for item in row) for row, scale in zip(raw, scales))
    if any(not math.isfinite(item) for row in normalized for item in row):
        raise ModelError("AD_NORMALIZATION_NONFINITE", "Row normalization produced nonfinite derivative")
    details = (
        ("backend_versions", runtime.versions), ("actual_platform", "cpu"), ("jit_disabled", True),
        ("raw_definition", "dF/dx in canonical stored variables"),
        ("row_normalized_definition", "S^-1 dF/dx; all dated slots already collapsed"),
        ("primal_comparison_atol", SS_RESID_ATOL), ("primal_comparison_rtol", SS_RESID_RTOL),
        ("primal_normalized_absolute_differences", tuple(discrepancies)),
        ("independent_validation", "NOT_RUN for this request; test evidence is separate"),
    )
    return raw, normalized, details


def evaluate_steady_jacobian(model: ValidatedModel, candidate: Mapping[str, float] | Sequence[float],
                             *, backend: str = BACKEND_NAME) -> JacobianResult:
    """Return scoped AD point checks; a nonzero F does not invalidate dF/dx."""
    primal = evaluate_residuals(model, candidate)
    checks: list[CheckResult] = []
    diagnostics: list[Diagnostic] = []
    metadata = primal.metadata
    raw = normalized = shape = None
    dtype = None
    ad_execution = ExecutionStatus.NOT_RUN
    execution = ExecutionStatus.NOT_RUN
    digest = str(dict(metadata).get("evaluation_hash", ""))
    evidence = (EvidenceReference("computed_AD_point_record", "inline:jacobian", digest),)
    try:
        if not primal.evaluation_valid:
            diagnostics.extend(primal.diagnostics)
            raise ModelError("PRIMAL_EVALUATION_FAILED", "Cannot differentiate an invalid primal evaluation",
                             unsupported=primal.status is AssessmentStatus.REFUSED)
        checks.append(CheckResult("primal.valid", CheckStatus.PASS, "All original supplied-point rows evaluated", evidence))
        if type(backend) is not str or backend != BACKEND_NAME:
            raise ModelError("AD_BACKEND_UNSUPPORTED", "Only frozen jax.jacfwd is allowed; no FD fallback", unsupported=True)
        runtime = load_backend()
        checks.append(CheckResult("backend.admitted", CheckStatus.PASS, "Pinned CPU x64 no-JIT backend", evidence))
        ad_execution = ExecutionStatus.FAILED
        output = execute_ad(model, primal.candidate, runtime)
        ad_execution = ExecutionStatus.COMPLETED
        raw, normalized, details = verify_ad_output(model, primal.residual_vector, runtime, output)
        metadata += details
        shape = (len(primal.equation_order), len(primal.variable_order))
        dtype = "float64"
        checks.append(CheckResult("derivative.point_checks", CheckStatus.PASS,
                                  "Canonical axes, finite float64 derivatives and scalar/JAX primal agreement", evidence))
        execution = ExecutionStatus.COMPLETED
    except ModelError as exc:
        diagnostics.append(exc.diagnostic)
        checks.append(CheckResult("derivative.request", CheckStatus.UNSUPPORTED if exc.diagnostic.unsupported else CheckStatus.FAIL,
                                  str(exc), evidence))
        execution = ExecutionStatus.NOT_RUN if ad_execution is ExecutionStatus.NOT_RUN and exc.diagnostic.unsupported else ExecutionStatus.FAILED
    except Exception as exc:
        # A backend exception must remain an explicit failed request, never FD.
        diagnostics.append(Diagnostic("AD_EVALUATION_EXCEPTION", f"{type(exc).__name__}: {exc}"))
        checks.append(CheckResult("derivative.request", CheckStatus.FAIL, str(diagnostics[-1]), evidence))
        execution = ExecutionStatus.FAILED
    valid = execution is ExecutionStatus.COMPLETED
    if not valid:
        raw = normalized = shape = dtype = None
    return JacobianResult(
        aggregate_assessment(tuple(checks)), execution, ad_execution, valid,
        raw, normalized, shape, primal.equation_order, primal.variable_order,
        primal.candidate, primal.residual_vector, BACKEND_NAME, dtype,
        tuple(checks), tuple(diagnostics), metadata,
    )
