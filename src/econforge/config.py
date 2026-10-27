"""Strict application configuration for non-numerical Phase 1 features."""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from econforge.errors import ConfigurationError


SUPPORTED_SCHEMA_VERSION = 1
_TOP_LEVEL_KEYS = {"schema_version", "output_dir", "offline", "logging"}
_LOGGING_KEYS = {"level", "format"}
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_LOG_FORMATS = {"text", "json"}


def _unknown_keys(data: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigurationError(
            f"Unknown {context} configuration key(s): {', '.join(unknown)}"
        )


def _require_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ConfigurationError(f"{field_name} must be a boolean")
    return value


def _environment_bool(value: str, field_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(
        f"{field_name} must be one of true/false, yes/no, on/off, or 1/0"
    )


@dataclass(frozen=True)
class LoggingSettings:
    level: str = "INFO"
    format: str = "text"

    def __post_init__(self) -> None:
        normalized_level = self.level.upper()
        normalized_format = self.format.lower()
        if normalized_level not in _LOG_LEVELS:
            raise ConfigurationError(
                f"logging.level must be one of {sorted(_LOG_LEVELS)}"
            )
        if normalized_format not in _LOG_FORMATS:
            raise ConfigurationError(
                f"logging.format must be one of {sorted(_LOG_FORMATS)}"
            )
        object.__setattr__(self, "level", normalized_level)
        object.__setattr__(self, "format", normalized_format)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LoggingSettings":
        _unknown_keys(data, _LOGGING_KEYS, "logging")
        return cls(
            level=data.get("level", "INFO"),
            format=data.get("format", "text"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "format": self.format}


@dataclass(frozen=True)
class AppConfig:
    schema_version: int = SUPPORTED_SCHEMA_VERSION
    output_dir: Path = Path("artifacts")
    offline: bool = True
    logging: LoggingSettings = LoggingSettings()

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise ConfigurationError("schema_version must be an integer")
        if self.schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ConfigurationError(
                "Unsupported schema_version "
                f"{self.schema_version}; expected {SUPPORTED_SCHEMA_VERSION}"
            )

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "AppConfig":
        _unknown_keys(data, _TOP_LEVEL_KEYS, "top-level")
        logging_data = data.get("logging", {})
        if not isinstance(logging_data, Mapping):
            raise ConfigurationError("logging must be a table or object")

        raw_output = data.get("output_dir", "artifacts")
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise ConfigurationError("output_dir must be a non-empty string")
        output_path = Path(raw_output)
        if base_dir is not None and not output_path.is_absolute():
            output_path = base_dir / output_path

        return cls(
            schema_version=data.get("schema_version", SUPPORTED_SCHEMA_VERSION),
            output_dir=output_path.resolve(strict=False),
            offline=_require_bool(data.get("offline", True), "offline"),
            logging=LoggingSettings.from_mapping(logging_data),
        )

    def with_environment(
        self,
        environ: Mapping[str, str] | None = None,
        *,
        base_dir: Path | None = None,
    ) -> "AppConfig":
        source = os.environ if environ is None else environ
        output_dir = self.output_dir
        if "ECONFORGE_OUTPUT_DIR" in source:
            candidate = Path(source["ECONFORGE_OUTPUT_DIR"])
            if not candidate.is_absolute() and base_dir is not None:
                candidate = base_dir / candidate
            output_dir = candidate.resolve(strict=False)

        offline = self.offline
        if "ECONFORGE_OFFLINE" in source:
            offline = _environment_bool(
                source["ECONFORGE_OFFLINE"], "ECONFORGE_OFFLINE"
            )

        logging_settings = LoggingSettings(
            level=source.get("ECONFORGE_LOG_LEVEL", self.logging.level),
            format=source.get("ECONFORGE_LOG_FORMAT", self.logging.format),
        )
        return AppConfig(
            schema_version=self.schema_version,
            output_dir=output_dir,
            offline=offline,
            logging=logging_settings,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "output_dir": str(self.output_dir),
            "offline": self.offline,
            "logging": self.logging.to_dict(),
        }


def load_config(
    path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load strict TOML or JSON configuration and apply explicit env overrides."""

    if path is None:
        base_dir = Path.cwd()
        config = AppConfig.from_mapping({}, base_dir=base_dir)
        return config.with_environment(environ, base_dir=base_dir)

    config_path = Path(path).resolve(strict=False)
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        if config_path.suffix.lower() == ".toml":
            with config_path.open("rb") as stream:
                raw = tomllib.load(stream)
        elif config_path.suffix.lower() == ".json":
            with config_path.open("r", encoding="utf-8") as stream:
                raw = json.load(stream)
        else:
            raise ConfigurationError(
                "Configuration file must use the .toml or .json extension"
            )
    except (OSError, tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Unable to parse configuration: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigurationError("Configuration root must be a table or object")
    config = AppConfig.from_mapping(raw, base_dir=config_path.parent)
    return config.with_environment(environ, base_dir=config_path.parent)

