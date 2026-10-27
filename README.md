# EconForge

**Numerical Economics Research Toolkit**

EconForge is an experimental Python toolkit for numerical economic modeling and reproducible quantitative research.

Its initial Alpha release focuses on steady-state computations for Real Business Cycle (RBC) and New Keynesian (NK) model examples.

The project aims to make numerical economics workflows easier to inspect, reproduce, and extend.

**Current status: Experimental Alpha.**

## What is EconForge?

Economic models describe relationships between quantities such as capital, consumption, productivity, inflation and the output gap. At a steady state, the model's equilibrium equations must hold simultaneously. EconForge reads a bounded JSON model, evaluates how far a proposed state is from satisfying those equations, computes its steady-state Jacobian, and searches for a root from an explicit starting point.

The solver checks the original scalar equations again after the numerical search. Its structured result separates execution, numerical checks and model status. A successful local root does not establish that the model is economically appropriate or that the equilibrium is unique.

## Why use EconForge?

- **Learn numerical economics:** follow small RBC and NK examples from equations and initial guesses to a checked steady state.
- **Reproduce a calculation:** retain the model, starting point, package checksum, environment and complete diagnostics.
- **Inspect research prototypes:** evaluate residuals and Jacobians and investigate solver refusal or nonconvergence. Independently validate each new model and parameter region before relying on it.

## Features

| Functionality | Alpha scope |
| --- | --- |
| RBC steady states | RBC-A with log utility and full depreciation; RBC-B with CRRA utility and partial depreciation; two fixed starts each |
| New Keynesian steady state | NK-A linear IS / Phillips curve / Taylor-rule benchmark, with a nonzero supplied start |
| Model parsing and residuals | Bounded JSON grammar, explicit variable domains and deterministic scalar residuals |
| Steady-state Jacobians | Validated automatic differentiation with the pinned CPU float64 configuration |
| Root solving | SciPy trust-region reflective least squares, explicit starts, fixed tolerances and fresh scalar residual verification |
| Diagnostics | Structured statuses, attempts, checks and failure/refusal details |

The supported interfaces are `parse_model`, `evaluate_residuals`, `evaluate_steady_jacobian` and `solve_steady_state`, plus the documented `Start` and `SolveLimits` records. See [API details](docs/API.md). Other importable modules are outside this Alpha's supported scope.

## Installation

The qualified environment is **macOS 26.5 on Apple silicon, CPython 3.12.14**. Other platforms and Python versions have not been qualified. Python package metadata permits 3.12 only.

GitHub Releases is the distribution channel. EconForge is not published to PyPI or TestPyPI by this release process. From the [Alpha release page](https://github.com/finleydoudou/EconForge-Alpha/releases/tag/v0.1.0-alpha.1), download these files into one directory:

- `econforge-0.1.0a1-py3-none-any.whl`
- `econforge-0.1.0a1.tar.gz`
- `SHA256SUMS.txt`

Verify the downloaded files and install the wheel:

```sh
shasum -a 256 -c SHA256SUMS.txt
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install './econforge-0.1.0a1-py3-none-any.whl[ad]'
python -m pip check
```

The `[ad]` extra installs the six pinned numerical dependencies from your configured package index. The wheel without that extra supports parsing and scalar residual evaluation only. The portable wheel filename does not imply numerical qualification on every platform.

To install from source instead, create a separate environment and use:

```sh
python3.12 -m venv .venv-source
source .venv-source/bin/activate
python -m pip install './econforge-0.1.0a1.tar.gz[ad]'
python -m pip check
```

The source build uses the pinned build requirements in `pyproject.toml`. See [installation details](docs/INSTALLATION.md), including offline installation and troubleshooting.

## Quick Start

Activate the installed environment. Set these values **before Python imports the numerical libraries**:

```sh
export JAX_ENABLE_X64=true JAX_PLATFORMS=cpu JAX_DISABLE_JIT=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

Run all five packaged analytical benchmark cases from any directory:

```sh
python - <<'PYTHON'
from importlib.resources import files
import runpy
example = files("econforge") / "_resources" / "examples" / "steady_models.py"
runpy.run_path(str(example), run_name="__main__")
PYTHON
```

The script prints full results and checks each solution against its frozen analytical reference. All five cases must report `PASS`. The examples require no source checkout.

## RBC Example

This example uses the packaged RBC-A model and its explicit primary start:

```python
import json
from importlib.resources import files
from econforge.model import parse_model
from econforge.steady_state import solve_steady_state

examples = files("econforge") / "_resources" / "examples"
case = next(c for c in json.loads((examples / "cases.json").read_text())
            if c["case_id"] == "RBC-A_primary")
model = parse_model((examples / case["model_file"]).read_text())
result = solve_steady_state(model, initial_guess=case["initial_guess"])
assert result.status.value == "PASS" and result.solution_candidate is not None
print(result.solution_candidate)  # canonical order: K, z, c
```

RBC-A uses fixed labor, log utility and full depreciation. RBC-B adds CRRA utility and partial depreciation. These are analytical teaching and verification benchmarks, not empirical estimates. The [runnable RBC script](examples/quickstart_rbc.py) prints the complete structured result.

## New Keynesian Example

NK-A combines a linear IS equation, a New Keynesian Phillips curve and a Taylor-rule substitution. Its zero steady state is approached from a supplied nonzero initial guess:

```python
import json
from importlib.resources import files
from econforge.model import parse_model
from econforge.steady_state import solve_steady_state

examples = files("econforge") / "_resources" / "examples"
case = next(c for c in json.loads((examples / "cases.json").read_text())
            if c["case_id"] == "NK-A")
model = parse_model((examples / case["model_file"]).read_text())
result = solve_steady_state(model, initial_guess=case["initial_guess"])
assert result.status.value == "PASS" and result.solution_candidate is not None
print(result.solution_candidate)  # canonical order: v, x, pi; near zero
```

See the [runnable NK script](examples/quickstart_nk.py) and [numerical assumptions](docs/NUMERICAL_ASSUMPTIONS.md).

## Scientific Scope and Limitations

This is a narrow experimental steady-state Alpha, not a certified complete DSGE system. Qualification covers the five supplied starts on the recorded macOS/Python stack. Custom models and arbitrary parameter regions require separate scientific validation.

A verified residual root does not prove global uniqueness, optimality, stationarity, empirical fit or a solved stochastic equilibrium. Dynamic derivatives, matrix assembly, QZ/BK, policy recovery, innovation loading, stochastic-law modules and finite IRFs remain unsupported even when their modules are present. No estimation, stochastic simulation, higher-order analysis or P3.10 functionality is supported.

No complete Linux or Windows scientific validation, universal cross-platform reproducibility, successful P3.8 H recovery on Linux, broader G1/T1 activation, or stable P3.9 stochastic support is claimed. Frozen full-system, external-reference and platform release requirements remain incomplete and distinct from this Alpha's installation and example checks. See [all limitations](docs/LIMITATIONS.md).

## Documentation

- [Installation and troubleshooting](docs/INSTALLATION.md)
- [Supported API and result interpretation](docs/API.md)
- [Numerical assumptions and tolerances](docs/NUMERICAL_ASSUMPTIONS.md)
- [Scientific limitations](docs/LIMITATIONS.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Runnable examples](examples/)
- [Contributing](CONTRIBUTING.md)

The Python distributions also install the guides and example files under `econforge/_resources`.

## Reproducibility

The numerical stack pins JAX/jaxlib **0.6.2**, NumPy **2.2.6**, SciPy **1.15.3**, ml_dtypes **0.6.0** and opt_einsum **3.4.0**. Use CPU float64, disabled JIT and the thread settings above.

Release qualification includes 221 unchanged public-scope regression tests and five analytical example checks against each installed distribution. Controlled local builds must produce identical wheels and identical normalized source archives, and final publication requires independent acceptance of the exact source and artifact hashes. A version number alone never identifies accepted bytes. These are local checks, not independent cross-platform certification.

Keep your downloaded artifact SHA256, model JSON, explicit initial guess, complete result and environment with every research result. See [reproduction guidance](docs/REPRODUCIBILITY.md).

## License

EconForge is licensed under the [MIT License](LICENSE). Copyright (c) 2026 finleydoudou. The public developer name is a pseudonym; it does not identify a separate company or change ownership. Dependencies are separately installed and retain their own licenses; see [third-party notices](THIRD_PARTY_NOTICES.txt).

## Contributing

Report reproducible problems in [GitHub Issues](https://github.com/finleydoudou/EconForge-Alpha/issues). Include the version and artifact hash, environment, minimal model and initial guess, full result and expected analytical behavior. Remove credentials and private data. See [CONTRIBUTING.md](CONTRIBUTING.md) before proposing numerical changes.

## Citation

Cite **EconForge, version 0.1.0a1, Experimental Alpha**, the [GitHub release](https://github.com/finleydoudou/EconForge-Alpha/releases/tag/v0.1.0-alpha.1), the artifact SHA256 and your access date. Cite model equations and economic sources separately. No DOI, paper or institutional endorsement is claimed.
