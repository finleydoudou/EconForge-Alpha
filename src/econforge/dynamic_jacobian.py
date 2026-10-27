"""P3.4 original-integrand date partials at a verified deterministic reference."""

from dataclasses import asdict, dataclass
import math
from typing import Any

from econforge.ad_backend import ADBackend, load_backend
from econforge.ad_policy import BACKEND_NAME
from econforge.dynamic_ast import DATE_OFFSETS, LoweredDynamicResidual, lower_dynamic_ast
from econforge.model import ValidatedModel
from econforge.model_errors import Diagnostic, ModelError
from econforge.residual_policy import SS_RESID_ATOL, SS_RESID_RTOL
from econforge.residuals import evaluate_integrand, evaluate_residuals
from econforge.steady_jacobian import Matrix, checked_array
from econforge.status import (AssessmentStatus, CapabilityMaturity, CheckResult, CheckStatus,
                             EvidenceReference, ExecutionStatus, ModelStatus, aggregate_assessment)


@dataclass(frozen=True)
class DateDerivativeBlock:
    name: str
    offset: int
    jacobian: Matrix
    row_normalized_jacobian: Matrix
    coordinate_normalized_jacobian: Matrix
    shape: tuple[int, int]
    equation_order: tuple[str, ...]
    variable_order: tuple[str, ...]
    source_to_canonical_variables: tuple[int, ...]
    canonical_to_source_variables: tuple[int, ...]
    source_to_canonical_equations: tuple[int, ...]
    canonical_to_source_equations: tuple[int, ...]
    row_permutation_scope: str = "all original verification rows including identities; derivative rows are equation_order"


@dataclass(frozen=True)
class DynamicADOutput:
    lowered: LoweredDynamicResidual
    reference: tuple[float, ...]
    matrix: Any
    primal: Any
    trace: Any


@dataclass(frozen=True)
class DynamicDerivativeResult:
    assessment_status: AssessmentStatus
    execution_status: ExecutionStatus
    ad_execution_status: ExecutionStatus
    derivative_valid: bool
    blocks: tuple[DateDerivativeBlock, ...] | None
    flattened_jacobian: Matrix | None
    shape: tuple[int, int] | None
    equation_order: tuple[str, ...]
    variable_order: tuple[str, ...]
    slot_order: tuple[tuple[int, str], ...]
    reference: tuple[float, ...] | None
    residual_vector: tuple[float, ...] | None
    backend: str
    dtype: str | None
    checks: tuple[CheckResult, ...]
    diagnostics: tuple[Diagnostic, ...]
    metadata: tuple[tuple[str, object], ...]
    input_admission_status: str = "NOT_ADMITTED"
    warnings: tuple[str, ...] = ()
    independent_validation_status: CheckStatus = CheckStatus.NOT_RUN
    benchmark_status: CheckStatus = CheckStatus.NOT_RUN
    model_status: ModelStatus = ModelStatus.NOT_EVALUATED
    capability_maturity: CapabilityMaturity = CapabilityMaturity.IMPLEMENTED
    schema_version: int = 2
    claim: str = "original_integrand_date_partials_at_verified_reference_point_checks"

    @property
    def status(self) -> AssessmentStatus:
        return self.assessment_status

    @property
    def failure_reason(self) -> str | None:
        return self.diagnostics[0].code if self.diagnostics else None

    def to_dict(self) -> dict[str, object]:
        record = asdict(self)
        record["checks"] = [c.to_dict() for c in self.checks]
        record["metadata"] = dict(self.metadata)
        record["failure_reason"] = self.failure_reason
        return record


def validate_dynamic_axes(model: ValidatedModel, lowered: LoweredDynamicResidual) -> None:
    """Reject date/identity metadata mismatches even if array dimensions agree."""
    if type(lowered) is not LoweredDynamicResidual or lowered.model_hash != model.digest:
        raise ModelError("DYNAMIC_MODEL_MISMATCH", "Lowering does not belong to the admitted model")
    if lowered.equation_order != model.equation_order:
        raise ModelError("AD_ROW_ORDER_MISMATCH", "Dynamic row identities differ from model")
    if lowered.variable_order != model.variable_order:
        raise ModelError("AD_COLUMN_ORDER_MISMATCH", "Dynamic column identities differ from model")
    expected = tuple((offset, name) for offset in DATE_OFFSETS for name in model.variable_order)
    if (type(lowered.slot_order) is not tuple
            or any(type(s) is not tuple or len(s) != 2 or type(s[0]) is not int or type(s[1]) is not str for s in lowered.slot_order)
            or lowered.slot_order != expected):
        raise ModelError("DYNAMIC_SLOT_MISMATCH", "Require explicit complete -1/0/+1 canonical slot order")
    if lowered.row_kinds != tuple(e.kind for e in model.equations if e.kind != "identity"):
        raise ModelError("DYNAMIC_ROW_KIND_MISMATCH", "Whole-integrand expectation/pathwise tags changed")


def execute_dynamic_ad(model: ValidatedModel, reference: tuple[float, ...], runtime: ADBackend) -> DynamicADOutput:
    """Differentiate independent slots once, then evaluate at the diagonal reference."""
    lowered = lower_dynamic_ast(model, runtime)
    validate_dynamic_axes(model, lowered)
    x = runtime.numpy.asarray(reference * 3, dtype=runtime.numpy.float64)
    checked_array(runtime.jax.device_get(x), (3 * len(reference),), "dynamic reference")
    matrix, (primal, trace) = runtime.jax.jacfwd(lowered.function, has_aux=True)(x)
    return DynamicADOutput(lowered, reference, runtime.jax.device_get(matrix),
                           runtime.jax.device_get(primal), runtime.jax.device_get(trace))


def verify_dynamic_output(model: ValidatedModel, reference: tuple[float, ...], scalar: tuple[float, ...],
                          output: DynamicADOutput) -> tuple[tuple[DateDerivativeBlock, ...], Matrix, tuple]:
    """Check original scalar integrands, trace, axes, dtype and frozen S/T maps."""
    if type(output) is not DynamicADOutput or output.reference != reference:
        raise ModelError("DYNAMIC_REFERENCE_MISMATCH", "Backend record belongs to another reference")
    validate_dynamic_axes(model, output.lowered)
    rows = tuple(e for e in model.equations if e.kind != "identity")
    n = len(model.variable_order)
    labels = tuple(f"{name}@{offset:+d}" for offset, name in output.lowered.slot_order)
    matrix = checked_array(output.matrix, (n, 3*n), "dynamic Jacobian", model.equation_order, labels)
    primal = checked_array(output.primal, (n,), "dynamic AD primal")
    # Retain the original AST source span when a hidden intermediate is nonfinite.
    trace = output.trace
    if tuple(trace.shape) != (len(output.lowered.nodes),):
        raise ModelError("AD_SHAPE_MISMATCH", "Dynamic intermediate trace shape mismatch")
    if str(trace.dtype) != "float64":
        raise ModelError("DTYPE_MISMATCH", "Dynamic intermediates require real float64")
    for node, value in zip(output.lowered.nodes, trace.tolist()):
        if not math.isfinite(value):
            raise ModelError("AD_INTERMEDIATE_NONFINITE", "Dynamic primal intermediate is nonfinite",
                             node.source, start=node.start, end=node.end)
    # All scalar original equations are evaluated through the accepted P3.1A API.
    slots = {(name, offset): value for offset in DATE_OFFSETS for name, value in zip(model.variable_order, reference)}
    slots.update({(s.name, offset): 0.0 for s in model.symbols if s.role == "innovation" for offset in DATE_OFFSETS})
    original = tuple(evaluate_integrand(model, e.equation_id, slots).value for e in rows)
    differences = []
    for equation, actual, expected, collapsed in zip(rows, primal.tolist(), original, scalar):
        for other in (actual, collapsed):
            difference = abs(other/equation.scale - expected/equation.scale)
            if not math.isfinite(difference) or difference > SS_RESID_ATOL + SS_RESID_RTOL*abs(expected/equation.scale):
                raise ModelError("DYNAMIC_PRIMAL_MISMATCH", "Original, collapsed scalar and JAX integrands disagree", equation.equation_id)
        differences.append(abs(actual/equation.scale - expected/equation.scale))
    scales = tuple(e.scale for e in rows)
    coordinates = tuple(value if v.coordinate == "log_deviation" else v.scale if v.coordinate == "scaled_level" else 1.0
                        for v, value in zip(model.variables, reference))
    raw = tuple(tuple(row) for row in matrix.tolist())
    blocks = []
    for i, (name, offset) in enumerate(zip(("J_minus", "J_current", "J_plus"), DATE_OFFSETS)):
        block = tuple(row[i*n:(i+1)*n] for row in raw)
        normalized = tuple(tuple(value/scale for value in row) for row, scale in zip(block, scales))
        coordinate = tuple(tuple(value*t for value, t in zip(row, coordinates)) for row in normalized)
        if any(not math.isfinite(v) for values in (normalized, coordinate) for row in values for v in row):
            raise ModelError("AD_NORMALIZATION_NONFINITE", "Dynamic S/T normalization is nonfinite")
        blocks.append(DateDerivativeBlock(name, offset, block, normalized, coordinate, (n, n),
                                           model.equation_order, model.variable_order,
                                           model.source_to_canonical_variables, model.canonical_to_source_variables,
                                           model.source_to_canonical_equations, model.canonical_to_source_equations))
    details = (("row_scales_S", scales), ("coordinate_diagonal_T", coordinates),
               ("raw_definition", "partial original signed integrand / partial stored w(offset)"),
               ("row_normalized_definition", "S^-1 J_slot"),
               ("coordinate_normalized_definition", "S^-1 J_slot T"),
               ("interpretation", "untransformed original integrand; no timing reduction or nonlinear expectation"),
               ("solver_ready_admission", "NOT_EVALUATED"), ("row_kinds", output.lowered.row_kinds),
               ("original_slots", output.lowered.original_slots), ("innovations", "fixed zero; not differentiated"),
               ("primal_normalized_absolute_differences", tuple(differences)),
               ("intermediate_count", len(trace)), ("date_offsets", DATE_OFFSETS))
    return tuple(blocks), raw, details


def evaluate_dynamic_jacobian(model: ValidatedModel, reference: object,
                               *, backend: str = BACKEND_NAME) -> DynamicDerivativeResult:
    """Verify an explicitly supplied steady reference, then report date partials only."""
    primal = evaluate_residuals(model, reference)
    metadata = primal.metadata
    # The existing scalar checkpoint admits both the model and supplied candidate.
    # Track the checkpoint independently of metadata presence; never invent a hash.
    input_admission_status = "ADMITTED" if any(
        c.check_id == "input.valid" and c.status is CheckStatus.PASS for c in primal.checks
    ) else "NOT_ADMITTED"
    checks: list[CheckResult] = []
    diagnostics: list[Diagnostic] = []
    blocks = raw = shape = dtype = None
    execution = ad_execution = ExecutionStatus.NOT_RUN
    digest = str(dict(metadata).get("evaluation_hash", ""))
    evidence = (EvidenceReference("computed_dynamic_AD_point_record", "inline:dynamic_jacobian", digest),)
    slot_order = tuple((offset, name) for offset in DATE_OFFSETS for name in primal.variable_order)
    try:
        if not primal.evaluation_valid or not primal.residuals_within_tolerance:
            diagnostics.extend(primal.diagnostics)
            raise ModelError("STEADY_REFERENCE_REQUIRED", "All original scalar reference rows and identities must pass",
                             unsupported=primal.status is AssessmentStatus.REFUSED)
        checks.append(CheckResult("reference.verified", CheckStatus.PASS, "Fresh original scalar steady acceptance", evidence))
        if type(backend) is not str or backend != BACKEND_NAME:
            raise ModelError("AD_BACKEND_UNSUPPORTED", "Only pinned JAX AD; no production finite differences", unsupported=True)
        runtime = load_backend()
        metadata += (("scalar_backend", "python_scalar_binary64"), ("backend", BACKEND_NAME),
                     ("backend_versions", runtime.versions), ("actual_platform", "cpu"), ("jit_disabled", True))
        checks.append(CheckResult("backend.admitted", CheckStatus.PASS, "Pinned CPU x64 no-JIT runtime", evidence))
        ad_execution = ExecutionStatus.FAILED
        output = execute_dynamic_ad(model, primal.candidate, runtime)
        ad_execution = ExecutionStatus.COMPLETED
        blocks, raw, details = verify_dynamic_output(model, primal.candidate, primal.residual_vector, output)
        metadata += details
        shape, dtype = (len(primal.equation_order), 3*len(primal.variable_order)), "float64"
        checks.append(CheckResult("dynamic_derivative.point_checks", CheckStatus.PASS,
                                  "Dated axes, finite float64 blocks, S/T and scalar/JAX primal agreement", evidence))
        execution = ExecutionStatus.COMPLETED
    except ModelError as exc:
        diagnostics.append(exc.diagnostic)
        checks.append(CheckResult("dynamic_derivative.request", CheckStatus.UNSUPPORTED if exc.diagnostic.unsupported else CheckStatus.FAIL,
                                  str(exc), evidence))
        execution = ExecutionStatus.NOT_RUN if ad_execution is ExecutionStatus.NOT_RUN and exc.diagnostic.unsupported else ExecutionStatus.FAILED
    except Exception as exc:
        diagnostics.append(Diagnostic("DYNAMIC_AD_EXCEPTION", f"{type(exc).__name__}: {exc}"))
        checks.append(CheckResult("dynamic_derivative.request", CheckStatus.FAIL, str(diagnostics[-1]), evidence))
        execution = ExecutionStatus.FAILED
    valid = execution is ExecutionStatus.COMPLETED
    if not valid:
        blocks = raw = shape = dtype = None
    return DynamicDerivativeResult(aggregate_assessment(tuple(checks)), execution, ad_execution, valid,
                                    blocks, raw, shape, primal.equation_order, primal.variable_order, slot_order,
                                    primal.candidate, primal.residual_vector, BACKEND_NAME, dtype,
                                    tuple(checks), tuple(diagnostics), metadata,
                                    input_admission_status=input_admission_status)
