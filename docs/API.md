# Supported solver API

`parse_model(source: str)` accepts the bounded JSON schema and returns a validated
model. Unknown syntax, invalid domains, conflicting metadata and unsafe expression
constructs are rejected. Keep original JSON and hashes with results.

`evaluate_residuals(model, candidate)` evaluates the deterministic steady residual
at a complete named dictionary or canonical sequence. It does not search for a
root. Nonzero residuals can be validly evaluated while failing the residual check.

`evaluate_steady_jacobian(model, candidate)` returns the raw steady-residual AD
Jacobian and separately labeled row-normalized Jacobian. This is a point check,
not certification of smoothness throughout arbitrary neighborhoods. Inspect
`derivative_valid`, statuses and `to_dict()`; invalid matrices remain unavailable.

`solve_steady_state(model, initial_guess=None, *, starts=None,
selected_attempt_id=None, limits=None)` requires explicit initial input. Use either
one initial_guess or an ordered list of `Start(id, candidate)` records. Each
attempt uses the accepted scalar residual and validated AD; there are no hidden
restarts. Several distinct verified branches without explicit selection refuse
an aggregate solution. Failed requested attempts are retained.

`SolveLimits` can only reduce frozen operational caps. The result exposes
`status`, `solution_candidate`, `attempts` and `to_dict()`. Check status before
using a candidate. Inspect each attempt's solver_converged, independently_verified,
verification and diagnostics. Candidate tuples follow canonical declared order.
Fresh scalar verification checks original equations after backend termination.
A successful root keeps model_status=NOT_EVALUATED. `iterations` can be null because
the backend does not expose that counter; it must not be interpreted as zero.

The package top level exposes __version__ only. Helpers not listed here are not
stable public APIs. `econforge doctor` is an engineering/configuration diagnostic.
Advanced dynamic/stochastic modules are implemented but outside this support scope.
