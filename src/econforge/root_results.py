"""Claim-scoped solve records; execution, convergence and verification differ."""

from dataclasses import asdict, dataclass, field

from econforge.model_errors import Diagnostic
from econforge.residuals import ResidualResult
from econforge.status import AssessmentStatus, CapabilityMaturity, CheckResult, CheckStatus, ExecutionStatus, ModelStatus


@dataclass(frozen=True)
class Start:
    """User-supplied attempt identity and canonical vector or named mapping."""

    attempt_id: str
    candidate: object


@dataclass
class SolveAttempt:
    """All attempted outcomes survive, including failed starts and verification."""

    attempt_id: str
    initial_candidate: tuple[float, ...] | None = None
    solution_candidate: tuple[float, ...] | None = None
    assessment_status: AssessmentStatus = AssessmentStatus.NOT_RUN
    execution_status: ExecutionStatus = ExecutionStatus.NOT_RUN
    solver_execution_status: ExecutionStatus = ExecutionStatus.NOT_RUN
    solver_converged: bool = False
    candidate_finite: bool = False
    independently_verified: bool = False
    verification: ResidualResult | None = None
    iterations: int | None = None
    function_evaluations: int | None = None
    jacobian_evaluations: int | None = None
    termination_code: int | None = None
    termination_message: str | None = None
    checks: list[CheckResult] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        record = asdict(self)
        record["checks"] = [c.to_dict() for c in self.checks]
        record["verification"] = self.verification.to_dict() if self.verification else None
        record["failure_reason"] = self.diagnostics[0].code if self.diagnostics else None
        return record


@dataclass(frozen=True)
class SteadyStateResult:
    assessment_status: AssessmentStatus
    execution_status: ExecutionStatus
    attempts: tuple[SolveAttempt, ...]
    clusters: tuple[tuple[str, ...], ...]
    selected_attempt_id: str | None
    solution_candidate: tuple[float, ...] | None
    variable_order: tuple[str, ...]
    equation_order: tuple[str, ...]
    checks: tuple[CheckResult, ...]
    diagnostics: tuple[Diagnostic, ...]
    warnings: tuple[str, ...]
    metadata: dict[str, object]
    schema_version: int = 2
    claim: str = "locally_found_steady_root_with_independent_scalar_verification"
    model_status: ModelStatus = ModelStatus.NOT_EVALUATED
    benchmark_status: CheckStatus = CheckStatus.NOT_RUN
    external_validation: CheckStatus = CheckStatus.NOT_RUN
    capability_maturity: CapabilityMaturity = CapabilityMaturity.IMPLEMENTED

    @property
    def status(self) -> AssessmentStatus:
        return self.assessment_status

    @property
    def failure_reason(self) -> str | None:
        return self.diagnostics[0].code if self.diagnostics else None

    def to_dict(self) -> dict[str, object]:
        record = asdict(self)
        record["attempts"] = [a.to_dict() for a in self.attempts]
        record["checks"] = [c.to_dict() for c in self.checks]
        record["failure_reason"] = self.failure_reason
        return record
