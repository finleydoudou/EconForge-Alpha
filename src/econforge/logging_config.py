"""Deterministic text and JSON logging with conservative redaction."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, TextIO

from econforge.config import LoggingSettings


_SENSITIVE_FRAGMENTS = ("token", "secret", "password", "passwd", "api_key", "apikey")


def redact(value: Any, *, key: str = "") -> Any:
    """Redact common secret-bearing fields from structured log context."""

    lowered = key.lower()
    if any(fragment in lowered for fragment in _SENSITIVE_FRAGMENTS):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): redact(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    """One JSON object per line using UTC timestamps."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if context is not None:
            payload["context"] = redact(context)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_logging(
    settings: LoggingSettings,
    *,
    stream: TextIO | None = None,
    logger_name: str = "econforge",
) -> logging.Logger:
    """Configure and return the project logger without altering the root logger."""

    logger = logging.getLogger(logger_name)
    logger.setLevel(settings.level)
    logger.propagate = False
    logger.handlers.clear()

    handler = logging.StreamHandler(stream)
    if settings.format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    logger.addHandler(handler)
    return logger

