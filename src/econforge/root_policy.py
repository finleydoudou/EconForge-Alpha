"""Frozen P3.3 settings, MATH_SPEC sections 6, 11 and 15."""

from dataclasses import asdict, dataclass

from econforge.model_errors import ModelError
from econforge.residual_policy import SPEC_REVISION, SS_RESID_ATOL, SS_RESID_RTOL, TOLERANCE_PROFILE

ROOT_BACKEND = "scipy.optimize.least_squares"
ROOT_METHOD = "trf"
ROOT_LOSS = "linear"
ROOT_TR_SOLVER = "exact"
OPTIMIZER_FTOL = OPTIMIZER_XTOL = OPTIMIZER_GTOL = 1e-12
SOLVER_STEP_ATOL, SOLVER_STEP_RTOL = 1e-12, 1e-10
BRANCH_ATOL, BRANCH_RTOL = 1e-8, 1e-6
RANK_ATOL, RANK_RTOL, RANK_AMBIGUITY_FACTOR = 1e-12, 1e-10, 10
STEADY_STARTS_MAX = 16
STEADY_RESIDUAL_CALLS_MAX = 2000
STEADY_WALL_SECONDS = 60


@dataclass(frozen=True)
class SolveLimits:
    """Operational caps only; reductions cannot loosen mathematical acceptance."""

    residual_calls: int = STEADY_RESIDUAL_CALLS_MAX
    backend_evaluations: int = STEADY_RESIDUAL_CALLS_MAX
    wall_seconds: float = STEADY_WALL_SECONDS

    def validate(self) -> None:
        for name, cap in (("residual_calls", STEADY_RESIDUAL_CALLS_MAX),
                          ("backend_evaluations", STEADY_RESIDUAL_CALLS_MAX)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= cap:
                raise ModelError("INVALID_RESOURCE_LIMIT", f"{name} must be an integer in [1,{cap}]")
        if type(self.wall_seconds) not in (int, float) or not 0 < self.wall_seconds <= STEADY_WALL_SECONDS:
            raise ModelError("INVALID_RESOURCE_LIMIT", "wall_seconds must be positive and at most 60")


def frozen_settings(limits: SolveLimits) -> dict[str, object]:
    """One manifest source; no public tolerance/backend override."""
    return {"specification_revision": SPEC_REVISION, "tolerance_profile": TOLERANCE_PROFILE, "backend": ROOT_BACKEND,
            "method": ROOT_METHOD, "loss": ROOT_LOSS, "tr_solver": ROOT_TR_SOLVER,
            "optimizer_ftol": OPTIMIZER_FTOL, "optimizer_xtol": OPTIMIZER_XTOL,
            "optimizer_gtol": OPTIMIZER_GTOL, "ss_resid_atol": SS_RESID_ATOL,
            "ss_resid_rtol": SS_RESID_RTOL, "solver_step_atol": SOLVER_STEP_ATOL,
            "solver_step_rtol": SOLVER_STEP_RTOL, "branch_atol": BRANCH_ATOL,
            "branch_rtol": BRANCH_RTOL, "rank_atol": RANK_ATOL,
            "rank_rtol": RANK_RTOL, "rank_ambiguity_factor": RANK_AMBIGUITY_FACTOR,
            "limits": asdict(limits), "maximum_starts": STEADY_STARTS_MAX,
            "coordinates": "stored canonical x; fun=F/S; jac=S^-1 J; x_scale=declared d",
            "iteration_limit": None, "iteration_count": "not exposed by pinned TRF",
            "wall_limit_enforcement": "checked before/after each backend callback and kernel; no late result accepted"}
