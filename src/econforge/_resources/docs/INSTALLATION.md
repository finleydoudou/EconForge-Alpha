# Installation and troubleshooting

Use CPython 3.12.14 on macOS 26.5 arm64 for the qualified configuration. The wheel tag describes packaging compatibility, not scientific certification. Linux, Windows, Intel Mac and other interpreter versions are unqualified.

Download the wheel, source archive and SHA256SUMS.txt from its release page. Follow the README's exact filenames and verify both downloads with `shasum -a 256 -c SHA256SUMS.txt`. Never substitute an unverified package found under the same name on an index.

## Wheel

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install './econforge-0.1.0a1-py3-none-any.whl[ad]'
python -m pip check
```

## Source archive

```sh
python3.12 -m venv .venv-source
source .venv-source/bin/activate
python -m pip install './econforge-0.1.0a1.tar.gz[ad]'
python -m pip check
```

Pip installs the build requirements `setuptools==84.0.0`, `wheel==0.48.0` and `packaging==26.3` in an isolated build environment. Runtime dependencies are pinned by the `[ad]` extra. Network installation uses your configured package index; it does not publish EconForge to that index.

## Offline installation

Prepare a trusted directory of wheels for the six pinned numerical dependencies and the three build requirements. Set WHEELHOUSE to its location, then use:

```sh
python -m pip install --no-index --find-links "$WHEELHOUSE" './econforge-0.1.0a1-py3-none-any.whl[ad]'
```

For a source installation, pip can also find the isolated build requirements there:

```sh
python -m pip install --no-index --find-links "$WHEELHOUSE" './econforge-0.1.0a1.tar.gz[ad]'
```

The project does not distribute a dependency wheel bundle. Each dependency retains its license and notices.

## Numerical configuration

Set these values before imports, in every new shell:

```sh
export JAX_ENABLE_X64=true JAX_PLATFORMS=cpu JAX_DISABLE_JIT=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

Run the README quick start to check all five examples. Missing dependencies, incorrect versions or wrong numerical configuration cause conservative refusal. Read the structured diagnostics; do not weaken tolerances to force a pass. `econforge version` prints the package version. `econforge doctor --json` is an engineering diagnostic whose PASS is not scientific certification; its historical capability listing is not a current support inventory.

Uninstall with `python -m pip uninstall econforge`. Pip leaves separately installed dependencies; removing the disposable virtual environment removes them too.
