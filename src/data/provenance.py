"""Data provenance logging.

Implements the first artefact in Section 3.10: a log recording, for each source
file, its retrieval timestamp, file size, SHA-256 checksum and licence terms,
so that any later question about which version of a dataset produced a given
result can be answered definitively.

The log is append-only JSON Lines. Each line is one immutable retrieval event.
Nothing is ever rewritten, because the value of a provenance record is exactly
that it cannot be quietly amended after the fact.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    "ProvenanceRecord",
    "sha256_checksum",
    "log_retrieval",
    "read_provenance_log",
    "verify_file",
]

_CHUNK_BYTES = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class ProvenanceRecord:
    """One immutable record of a data file's retrieval."""

    path: str
    filename: str
    retrieved_at_utc: str
    size_bytes: int
    sha256: str
    source_url: str
    licence: str
    notes: str = ""
    python_version: str = ""
    platform: str = ""


def sha256_checksum(path: str | Path) -> str:
    """SHA-256 of a file, streamed so large datasets do not exhaust memory.

    Reads in 1 MiB chunks. The Lending Club extract is over a gigabyte, so
    reading it whole would be wasteful at best.

    Args:
        path: Path to the file.

    Returns:
        Lowercase hexadecimal digest.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"no such file: {file_path}")

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def log_retrieval(
    path: str | Path,
    source_url: str,
    licence: str,
    log_path: str | Path,
    notes: str = "",
) -> ProvenanceRecord:
    """Record a data file retrieval in the append-only provenance log.

    Args:
        path: Path to the retrieved file.
        source_url: Canonical URL the file was obtained from.
        licence: Licence terms under which it was obtained.
        log_path: Path to the JSON Lines provenance log.
        notes: Optional context, e.g. a mirror used or a version label.

    Returns:
        The record that was written.
    """
    file_path = Path(path)
    record = ProvenanceRecord(
        path=str(file_path.resolve()),
        filename=file_path.name,
        retrieved_at_utc=datetime.now(UTC).isoformat(),
        size_bytes=file_path.stat().st_size,
        sha256=sha256_checksum(file_path),
        source_url=source_url,
        licence=licence,
        notes=notes,
        python_version=platform.python_version(),
        platform=platform.platform(),
    )

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")

    return record


def read_provenance_log(log_path: str | Path) -> list[ProvenanceRecord]:
    """Read all records from a provenance log.

    Args:
        log_path: Path to the JSON Lines provenance log.

    Returns:
        Records in the order written. Empty list if the log does not exist.
    """
    log_file = Path(log_path)
    if not log_file.is_file():
        return []

    records = []
    with log_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(ProvenanceRecord(**json.loads(line)))
    return records


def verify_file(path: str | Path, expected_sha256: str) -> bool:
    """Confirm a file still matches a previously recorded checksum.

    Run this at the start of any step that consumes raw data. A silent change
    to an input file between profiling and modelling is the kind of failure
    that produces irreproducible results without ever raising an error.

    Args:
        path: Path to the file.
        expected_sha256: The digest recorded at retrieval.

    Returns:
        True if the file's current digest matches.
    """
    return sha256_checksum(path) == expected_sha256.lower()
