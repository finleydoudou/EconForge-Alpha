"""Non-numerical command-line interface for the Phase 1 foundation."""

from __future__ import annotations

import argparse
import json
import sys
from enum import IntEnum
from pathlib import Path
from typing import Sequence, TextIO

from econforge.config import load_config
from econforge.errors import ConfigurationError
from econforge.provenance import runtime_snapshot
from econforge.version import __version__


class ExitCode(IntEnum):
    OK = 0
    CONFIGURATION_ERROR = 3
    INTERNAL_ERROR = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="econforge",
        description=(
            "EconForge engineering tools. Supplied-candidate residual evaluation "
            "is available through the Python API."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="optional application configuration in TOML or JSON format",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="show package version")
    version_parser.set_defaults(handler=_command_version)

    doctor_parser = subparsers.add_parser(
        "doctor", help="inspect the non-numerical runtime environment"
    )
    doctor_parser.add_argument("--json", action="store_true", dest="as_json")
    doctor_parser.set_defaults(handler=_command_doctor)

    check_parser = subparsers.add_parser(
        "config-check", help="validate application configuration"
    )
    check_parser.add_argument("path", type=Path)
    check_parser.add_argument("--json", action="store_true", dest="as_json")
    check_parser.set_defaults(handler=_command_config_check)
    return parser


def _emit(payload: dict[str, object], *, as_json: bool, stdout: TextIO) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), file=stdout)
        return
    for key, value in payload.items():
        print(f"{key}: {value}", file=stdout)


def _command_version(
    args: argparse.Namespace, *, stdout: TextIO, stderr: TextIO
) -> int:
    del args, stderr
    print(__version__, file=stdout)
    return ExitCode.OK


def _command_doctor(
    args: argparse.Namespace, *, stdout: TextIO, stderr: TextIO
) -> int:
    del stderr
    config = load_config(args.config)
    payload: dict[str, object] = {
        "execution_status": "COMPLETED",
        "check_status": "PASS",
        "evidence": ["inline:runtime", "inline:configuration"],
        "report_kind": "engineering_diagnostic",
        "capabilities": {
            "steady_state_residual_evaluation": "IMPLEMENTED",
            "steady_state_solving": "IMPLEMENTED",
            "independent_post_solve_verification": "IMPLEMENTED",
            "root_runtime_validation": "NOT_RUN",
            "automatic_differentiation": "IMPLEMENTED",
            "automatic_differentiation_scope": "steady_residual_variables_only",
            "steady_residual_jacobian": "IMPLEMENTED",
            "dynamic_derivative_blocks": "NOT_IMPLEMENTED",
            "derivative_runtime_validation": "NOT_RUN",
            "dynamic_solution": "NOT_IMPLEMENTED",
            "external_validation": "NOT_RUN",
        },
        "runtime": runtime_snapshot().to_dict(),
        "configuration": config.to_dict(),
    }
    _emit(payload, as_json=args.as_json, stdout=stdout)
    return ExitCode.OK


def _command_config_check(
    args: argparse.Namespace, *, stdout: TextIO, stderr: TextIO
) -> int:
    del stderr
    config = load_config(args.path)
    payload: dict[str, object] = {
        "execution_status": "COMPLETED",
        "check_status": "PASS",
        "configuration": config.to_dict(),
        "evidence": [f"config:{args.path.resolve(strict=False)}"],
    }
    _emit(payload, as_json=args.as_json, stdout=stdout)
    return ExitCode.OK


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args, stdout=stdout, stderr=stderr))
    except ConfigurationError as exc:
        print(f"CONFIGURATION_ERROR: {exc}", file=stderr)
        return int(ExitCode.CONFIGURATION_ERROR)


def entrypoint() -> None:
    raise SystemExit(main())

