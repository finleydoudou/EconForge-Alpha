# Numerical assumptions and tolerances

The model is deterministic at its declared stationary reference. Steady evaluation
collapses dated unknowns and sets innovations to their specified steady convention;
it does not evaluate general nonlinear conditional expectations. Stored variables,
canonical ordering, units, strict domains and positive scales are explicit.

RBC-A uses alpha=1/3, beta=24/25, rho=9/10, delta=1 and log utility with fixed labor.
RBC-B uses alpha=.36, beta=.99, rho=.9, delta=.025 and CRRA gamma=2. NK-A uses
beta=.99, sigma=1, kappa=.1, phi_pi=1.5, phi_x=.5 and rho_v=.5. These are original
project benchmark inputs; no empirical dataset or externally licensed fixture is
bundled. The JSON files contain the exact binary64 values used by the qualification.

Frozen residual bounds are abs(r_i) <= S_i*1e-10 + L_i*1e-8, with declared equation
scale S and magnitude L. AD verification uses its frozen dimensionless derivative
atol=1e-7 and rtol=1e-5. Optimizer ftol/xtol/gtol are 1e-12. Root solve uses SciPy
least_squares with TRF, linear loss, exact trust-region solver and callable JAX AD.
No user override of scientific tolerances is offered. The separate analytical
example comparison uses normalized error <= 1e-8 + 1e-6*abs(target/variable_scale).
These thresholds are project contracts, not universal forward-error guarantees.

At most 16 explicit starts, 2000 residual-budget charges/start and 60 seconds/start.
Deadlines are checked around callbacks/kernels, not guaranteed native-call preemption.
A singular steady Jacobian may warn; residual-root verification is not a uniqueness
certificate. No FD production fallback, regularization trick or pseudoinverse is
introduced. Finite differences remain validation-only.
