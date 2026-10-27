# Reproducing a calculation

Record the exact distribution SHA256, package version, model JSON, case ID, explicit initial guess, variable/equation ordering, complete structured result and diagnostics. Record Python, operating system, architecture, dependency versions, CPU/x64/no-JIT configuration and thread settings. The bundled fixtures are deterministic analytical examples; random tests do not replace them.

The qualified stack is macOS 26.5 arm64 / CPython 3.12.14 with JAX and jaxlib 0.6.2, NumPy 2.2.6, SciPy 1.15.3, ml_dtypes 0.6.0 and opt_einsum 3.4.0. All five benchmark starts are tested against their frozen analytical roots. Their origins and comparison tolerances are recorded in `examples/cases.json` and the numerical-assumptions guide.

This publication tree derives from frozen RC1. It retains all 55 library modules without changes to their bytes relative to RC1. RC1 changed only version text and the package docstring relative to the accepted scientific source. Documentation and packaging metadata are finalized separately, so the publication artifacts have different hashes.

RC1 recorded 221 distinct unchanged public-scope regression tests against each installed final distribution, without skips, and repeatable local builds. Publication artifacts require new installation and example checks plus acceptance of their exact final bytes. These engineering checks do not replace full-system scientific validation or establish cross-platform reproducibility.

Build requirements are pinned in `pyproject.toml`. Local release tooling preserves raw build output and normalizes only source-archive transport metadata (ordering, times and ownership); file contents are unchanged by normalization. This private preparation retains its evidence outside the publication tree. A package version string, checksum or successful installation alone is not scientific acceptance.

No external Dynare comparison, EF-REF-1 admission, Linux H recovery or complete-system release validation is claimed. Frozen full-system requirements remain authoritative.
