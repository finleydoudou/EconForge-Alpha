"""Status contracts shared by the CLI, reports, and future application layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExecutionStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NOT_RUN = "NOT_RUN"
    SKIPPED = "SKIPPED"


class CheckStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    UNSUPPORTED = "UNSUPPORTED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    NOT_RUN = "NOT_RUN"


class AssessmentStatus(str, Enum):
    """Claim-scoped scientific envelope v2 aggregate statuses."""

    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FAIL = "FAIL"
    REFUSED = "REFUSED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    NOT_RUN = "NOT_RUN"


def aggregate_assessment(checks: tuple[CheckResult, ...]) -> AssessmentStatus:
    """Aggregate exactly the caller's nonempty required-obligation set."""
    if not checks:
        return AssessmentStatus.NOT_RUN
    precedence = (
        (CheckStatus.FAIL, AssessmentStatus.FAIL),
        (CheckStatus.UNSUPPORTED, AssessmentStatus.REFUSED),
        (CheckStatus.NOT_IMPLEMENTED, AssessmentStatus.NOT_IMPLEMENTED),
        (CheckStatus.NOT_RUN, AssessmentStatus.NOT_RUN),
        (CheckStatus.WARN, AssessmentStatus.PASS_WITH_WARNINGS),
    )
    for check_status, aggregate in precedence:
        if any(check.status is check_status for check in checks):
            return aggregate
    return AssessmentStatus.PASS


class ModelStatus(str, Enum):
    UNIQUE_LOCAL_SOLUTION = "UNIQUE_LOCAL_SOLUTION"
    INDETERMINATE = "INDETERMINATE"
    NO_LOCAL_SOLUTION = "NO_LOCAL_SOLUTION"
    NUMERICALLY_UNRESOLVED = "NUMERICALLY_UNRESOLVED"
    UNSUPPORTED = "UNSUPPORTED"
    NOT_EVALUATED = "NOT_EVALUATED"


class CapabilityMaturity(str, Enum):
    DRAFT_MATH = "DRAFT_MATH"
    REVIEWED_SPEC = "REVIEWED_SPEC"
    IMPLEMENTED = "IMPLEMENTED"
    TESTED = "TESTED"
    REFERENCE_CHECKED = "REFERENCE_CHECKED"


@dataclass(frozen=True)
class EvidenceReference:
    """A stable reference to evidence produced by an actual check."""

    kind: str
    location: str
    digest: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "kind": self.kind,
            "location": self.location,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class CheckResult:
    """Result of one check without conflating it with program or model status."""

    check_id: str
    status: CheckStatus
    summary: str
    evidence: tuple[EvidenceReference, ...] = field(default_factory=tuple)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, CheckStatus):
            raise ValueError("status must be a CheckStatus value")
        if not self.check_id.strip():
            raise ValueError("check_id must not be empty")
        if self.status is CheckStatus.PASS and not self.evidence:
            raise ValueError("PASS requires at least one evidence reference")

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": self.status.value,
            "summary": self.summary,
            "evidence": [item.to_dict() for item in self.evidence],
            "details": dict(self.details),
        }

