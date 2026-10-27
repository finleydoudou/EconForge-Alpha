"""Small provenance primitives that do not perform scientific computation."""

from __future__ import annotations

import hashlib
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from econforge.version import __version__


@dataclass(frozen=True)
class FileDigest:
    algorithm: str
    digest: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeSnapshot:
    package_version: str
    python_version: str
    python_implementation: str
    platform: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def hash_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> FileDigest:
    """Hash a file with SHA-256 without loading it all into memory."""

    target = Path(path)
    digest = hashlib.sha256()
    size = 0
    with target.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            size += len(chunk)
            digest.update(chunk)
    return FileDigest("sha256", digest.hexdigest(), size)


def runtime_snapshot() -> RuntimeSnapshot:
    return RuntimeSnapshot(
        package_version=__version__,
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
    )

