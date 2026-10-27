"""Evaluate supplied candidates with explicit raw/scaled residual obligations."""

from dataclasses import asdict, dataclass
import hashlib
import json
import sys
from typing import Mapping, Sequence

from econforge.model import ValidatedModel, check_domain, expand_aliases, require_valid_model
from econforge.model_ast import evaluate_ast
from econforge.model_errors import Diagnostic, ModelError, real
from econforge.residual_policy import SPEC_REVISION, SS_RESID_ATOL, SS_RESID_RTOL, TOLERANCE_PROFILE, residual_thresholds
from econforge.status import (
    AssessmentStatus, CheckResult, CheckStatus, EvidenceReference, ExecutionStatus,
    ModelStatus, CapabilityMaturity, aggregate_assessment,
)
from econforge.steady_transform import steady_transform


@dataclass(frozen=True)
class RowResidual:
    equation_id: str
    source_index: int
    kind: str
    expression: str
    residual: float | None
    normalized_residual: float | None
    raw_threshold: float | None
    normalized_threshold: float | None
    raw_margin: float | None
    normalized_margin: float | None
    within_tolerance: bool
    diagnostic: Diagnostic | None = None


@dataclass(frozen=True)
class ResidualResult:
    assessment_status: AssessmentStatus
    execution_status: ExecutionStatus
    evaluation_valid: bool
    residuals_within_tolerance: bool
    variable_order: tuple[str, ...]
    equation_order: tuple[str, ...]
    candidate: tuple[float, ...] | None
    rows: tuple[RowResidual, ...]
    residual_vector: tuple[float, ...] | None
    normalized_residual_vector: tuple[float, ...] | None
    max_absolute_residual: float | None
    residual_norm: float | None
    normalized_residual_norm: float | None
    worst_equation: str | None
    diagnostics: tuple[Diagnostic, ...]
    checks: tuple[CheckResult, ...]
    metadata: tuple[tuple[str, object], ...]
    warnings: tuple[str, ...] = ()
    schema_version: int = 2
    claim: str = "supplied_candidate_steady_residuals_within_tolerance"
    model_status: ModelStatus = ModelStatus.NOT_EVALUATED
    norm_definition: str = "infinity_norm"
    capability_maturity: CapabilityMaturity = CapabilityMaturity.IMPLEMENTED

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


def candidate_values(model: ValidatedModel, candidate: Mapping[str, float] | Sequence[float]) -> tuple[float, ...]:
    """Maps are keyed by identity; sequences are explicitly in canonical w order."""
    if type(candidate) is dict:
        if any(type(key) is not str for key in candidate):
            raise ModelError("MALFORMED_CANDIDATE", "Candidate keys must be plain symbol strings")
        unknown = set(candidate) - set(model.variable_order)
        missing = set(model.variable_order) - set(candidate)
        if unknown:
            raise ModelError("UNKNOWN_VARIABLE", f"Unknown candidate keys {sorted(map(str, unknown))}")
        if missing:
            raise ModelError("MISSING_VARIABLE", f"Missing candidate keys {sorted(missing)}")
        values = tuple(candidate[name] for name in model.variable_order)
    elif type(candidate) in (list, tuple):
        if len(candidate) != len(model.variable_order):
            raise ModelError("CANDIDATE_DIMENSION", "Candidate must contain exactly n values")
        values = tuple(candidate)
    else:
        raise ModelError("MALFORMED_CANDIDATE", "Supply a plain dict, list or tuple")
    result = tuple(real(value, "candidate", symbol=name) for name, value in zip(model.variable_order, values))
    for variable, value in zip(model.variables, result):
        check_domain(value, variable.domain, variable.symbol.name)
        if variable.coordinate == "log_deviation" and value <= 0:
            raise ModelError("INVALID_DOMAIN", "Log-deviation level must stay positive", variable.symbol.name)
    return result


def _check(check_id: str, passed: bool, summary: str, digest: str, *, unsupported: bool = False) -> CheckResult:
    status = CheckStatus.PASS if passed else (CheckStatus.UNSUPPORTED if unsupported else CheckStatus.FAIL)
    return CheckResult(check_id, status, summary, (EvidenceReference("computed_residual_record", f"inline:{check_id}", digest),))


def evaluate_residuals(model: ValidatedModel, candidate: Mapping[str, float] | Sequence[float]) -> ResidualResult:
    """Check F(x) at supplied x; PASS certifies only the scoped residual claim."""
    diagnostics: list[Diagnostic] = []
    rows: list[RowResidual] = []
    checks: list[CheckResult] = []
    values = None
    digest = ""
    metadata: tuple[tuple[str, object], ...] = ()
    variable_order: tuple[str, ...] = ()
    equation_order: tuple[str, ...] = ()
    try:
        require_valid_model(model)
        variable_order, equation_order = model.variable_order, model.equation_order
        if sys.float_info.mant_dig != 53 or sys.float_info.max_exp != 1024:
            raise ModelError("DTYPE_UNSUPPORTED", "IEEE binary64 Python float required", unsupported=True)
        values = candidate_values(model, candidate)
        digest = hashlib.sha256((model.digest + json.dumps(values, allow_nan=False)).encode()).hexdigest()
        registry = {s.name: s for s in model.symbols}
        slots = {(name, 0): value for name, value in (*model.parameters, *zip(variable_order, values))}
        slots.update({(s.name, 0): 0.0 for s in model.symbols if s.role == "innovation"})
        metadata = (
            ("specification_revision", SPEC_REVISION), ("tolerance_profile", TOLERANCE_PROFILE),
            ("ss_resid_atol", SS_RESID_ATOL), ("ss_resid_rtol", SS_RESID_RTOL),
            ("backend", "python_scalar_binary64"), ("model_hash", model.digest), ("evaluation_hash", digest),
            ("model_id", model.model_id), ("branch_id", model.branch_id),
            ("frequency", model.frequency), ("parameters", model.parameters),
            ("source_to_canonical_variables", model.source_to_canonical_variables),
            ("canonical_to_source_variables", model.canonical_to_source_variables),
            ("source_to_canonical_equations", model.source_to_canonical_equations),
            ("canonical_to_source_equations", model.canonical_to_source_equations),
            ("coordinate_metadata", tuple((v.symbol.name, v.representation, v.coordinate, v.scale, v.baseline, v.display_unit) for v in model.variables)),
            ("row_scales", tuple((e.scale, e.magnitude) for e in model.equations)),
            ("verification_equation_order", tuple(e.equation_id for e in model.equations)),
            ("row_permutation_scope", "all verification rows including reporting identities; F excludes identities"),
            ("interpretation", "deterministic substitution; no stochastic expectation"),
        )
        checks.append(_check("input.valid", True, "Model, explicit ordering and candidate admitted", digest))
        for transformed in steady_transform(model):
            equation = transformed.original
            raw = normalized = raw_margin = normalized_margin = None
            within = False
            diagnostic = None
            threshold = normalized_threshold = None
            try:
                threshold, normalized_threshold = residual_thresholds(equation.scale, equation.magnitude, equation.equation_id)
                raw = evaluate_ast(transformed.tree, registry, slots)
                normalized = real(raw/equation.scale, equation.equation_id)
                raw_margin = real(threshold-abs(raw), equation.equation_id)
                normalized_margin = real(normalized_threshold-abs(normalized), equation.equation_id)
                within = abs(raw) <= threshold and abs(normalized) <= normalized_threshold
                if not within:
                    diagnostic = Diagnostic("RESIDUAL_TOO_LARGE", "Supplied candidate fails row acceptance", equation.equation_id)
            except ModelError as exc:
                diagnostic = exc.diagnostic
            if diagnostic is not None:
                diagnostics.append(diagnostic)
            rows.append(RowResidual(equation.equation_id, equation.source_index, equation.kind, equation.expression, raw, normalized, threshold, normalized_threshold, raw_margin, normalized_margin, within, diagnostic))
            checks.append(_check(f"row.{equation.equation_id}", within, "Raw and normalized supplied-point residual test", digest, unsupported=bool(diagnostic and diagnostic.unsupported)))
    except ModelError as exc:
        diagnostics.append(exc.diagnostic)
        checks.append(_check("input.valid", False, str(exc), digest, unsupported=exc.diagnostic.unsupported))
    valid = bool(rows) and len(rows) == len(model.equations) and all(row.raw_margin is not None and row.normalized_margin is not None for row in rows)
    canonical_rows = tuple(row for row in rows if row.kind != "identity")
    residual_vector = tuple(row.residual for row in canonical_rows) if valid else None
    normalized_vector = tuple(row.normalized_residual for row in canonical_rows) if valid else None
    maximum = max(map(abs, residual_vector)) if residual_vector else None
    normalized_maximum = max(map(abs, normalized_vector)) if normalized_vector else None
    worst = None
    if valid:
        worst = rows[max(range(len(rows)), key=lambda i: abs(rows[i].normalized_residual))].equation_id
    assessment = aggregate_assessment(tuple(checks))
    execution = ExecutionStatus.COMPLETED if valid else ExecutionStatus.FAILED
    if assessment is AssessmentStatus.REFUSED and not rows:
        execution = ExecutionStatus.NOT_RUN
    return ResidualResult(
        assessment, execution,
        valid, valid and all(row.within_tolerance for row in rows), variable_order, equation_order,
        values, tuple(rows), residual_vector, normalized_vector, maximum, maximum, normalized_maximum,
        worst, tuple(diagnostics), tuple(checks), metadata,
    )


@dataclass(frozen=True)
class IntegrandResult:
    value: float
    equation_id: str
    row_kind: str
    interpretation: str = "residual_integrand_at_explicit_slots_not_a_stochastic_expectation"


def evaluate_integrand(model: ValidatedModel, equation_id: str, slots: dict[tuple[str, int], float], *, expectation: bool = False) -> IntegrandResult:
    """Explicit-slot integrand only; conditional expectation requests are refused."""
    require_valid_model(model)
    if expectation:
        raise ModelError("EXPECTATION_UNSUPPORTED", "This evaluates an integrand, not a nonlinear expectation", equation_id, unsupported=True)
    registry = {s.name: s for s in model.symbols}
    if type(slots) is not dict:
        raise ModelError("MALFORMED_CANDIDATE", "Slots must be a plain dictionary")
    clean = {(name, 0): value for name, value in model.parameters}
    for key, value in slots.items():
        if type(key) is not tuple or len(key) != 2 or type(key[0]) is not str or type(key[1]) is not int or key[1] not in (-1, 0, 1):
            raise ModelError("INVALID_TIMING", "Expected (symbol, offset) with offset -1/0/+1")
        if key[0] not in registry or registry[key[0]].role in ("parameter", "alias", "observable"):
            raise ModelError("UNKNOWN_VARIABLE", "Only variable/innovation slots may be supplied")
        clean[key] = real(value, equation_id, symbol=key[0])
        for variable in model.variables:
            if variable.symbol.name == key[0]:
                check_domain(clean[key], variable.domain, key[0])
                if variable.coordinate == "log_deviation" and clean[key] <= 0:
                    raise ModelError("INVALID_DOMAIN", "Log-deviation level must stay positive", key[0])
    equation = next((e for e in model.equations if e.equation_id == equation_id), None)
    if equation is None:
        raise ModelError("UNKNOWN_EQUATION", equation_id)
    tree = expand_aliases(equation.tree, dict(model.aliases), registry)
    return IntegrandResult(evaluate_ast(tree, registry, clean), equation.equation_id, equation.kind)

