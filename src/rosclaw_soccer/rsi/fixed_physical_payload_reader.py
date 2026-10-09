"""Opt-in sealed report reader for one exact shared numerical payload.

Cache canonical encoding, not trust: authenticate source bytes on every read,
hash all payload bytes into both original seals, and return fresh JSON objects.
No default reader, running experiment, physics parameter or authority changes.
Actual archive parity and end-to-end throughput require external qualification.
"""

import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from rosclaw.growth.canonical_json_snapshot import CanonicalJSONSnapshot
from rosclaw.growth.frozen_payload_field import FrozenPayloadField
from rosclaw.growth.shared_proof_payload import MARKER, SCHEMA, canonical_hash

from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.rsi.physical_report_io import (
    CPU_MEAN_MODEL_LOCATION,
    CPU_MODEL_LOCATION,
    MEAN_MODEL_LOCATION,
    MODEL_LOCATION,
    resolve_physical_report,
)

_COMPRESSED_LIMIT = 256 * 1024**2
_DECOMPRESSED_LIMIT = 512 * 1024**2
_CANONICAL_LIMIT = 256 * 1024**2
_KEYS = {
    "schema",
    "location",
    "payload_hash",
    "logical_document_hash",
    "stripped_document",
    "envelope_hash",
}


class FixedPhysicalPayloadReader:
    """One fixed file and payload, never a global content-hash trust cache."""

    def __init__(self, payload_path: Path, *, expected_file_hash: str) -> None:
        if type(expected_file_hash) is not str or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", expected_file_hash
        ):
            raise ValueError("exact externally pinned payload file hash required")
        self._path = payload_path.absolute()
        self._file_hash = expected_file_hash
        self._verify_source()
        with gzip.open(self._path, "rb") as stream:
            encoded = stream.read(_DECOMPRESSED_LIMIT + 1)
        if len(encoded) > _DECOMPRESSED_LIMIT:
            raise ValueError("bounded complete numerical payload required")
        payload = json.loads(encoded)
        self._field = FrozenPayloadField(payload)
        self._payload_hash = self._field.payload_hash
        if len(self._field._payload_bytes) > _CANONICAL_LIMIT:
            raise ValueError("bounded canonical numerical payload required")
        if self._path.name != self._payload_hash[7:] + ".json.gz":
            raise ValueError("exact content-addressed payload filename required")
        self._verify_source()

    def _verify_source(self) -> None:
        if (
            self._path.is_symlink()
            or any(parent.is_symlink() for parent in self._path.parents)
            or not self._path.is_file()
            or self._path.stat().st_size > _COMPRESSED_LIMIT
        ):
            raise ValueError("bounded ordinary non-symlink payload file required")
        with self._path.open("rb") as stream:
            actual = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != self._file_hash:
            raise ValueError("fixed payload source bytes changed")

    def load_sealed(self, path: Path) -> dict[str, Any]:
        """Read a shared report with every original seal intact; no fallback."""
        self._verify_source()
        resolved = resolve_physical_report(path).absolute()
        if resolved.is_symlink() or any(parent.is_symlink() for parent in resolved.parents):
            raise ValueError("ordinary non-symlink physical report required")
        envelope = load_json_artifact(resolved)
        if (
            set(envelope) != _KEYS
            or envelope["schema"] != SCHEMA
            or envelope["payload_hash"] != self._payload_hash
            or self._field.payload_hash != self._payload_hash
            or type(envelope["location"]) is not list
            or envelope["location"]
            not in [
                list(MODEL_LOCATION),
                list(MEAN_MODEL_LOCATION),
                list(CPU_MODEL_LOCATION),
                list(CPU_MEAN_MODEL_LOCATION),
            ]
            or type(envelope["stripped_document"]) is not dict
            or resolved.parent.parent / ".shared-models" / self._path.name != self._path
        ):
            raise ValueError("exact fixed shared physical payload envelope required")
        if envelope["envelope_hash"] != canonical_hash(
            {k: v for k, v in envelope.items() if k != "envelope_hash"}
        ):
            raise ValueError("physical payload envelope seal changed")
        location = tuple(envelope["location"])
        fields: dict[str, Any] = CanonicalJSONSnapshot(envelope["stripped_document"]).restore()
        target = fields
        for key in location[:-1]:
            if type(target.get(key)) is not dict:
                raise ValueError("explicit physical payload dictionary path required")
            target = target[key]
        if target.get(location[-1]) != {MARKER: self._payload_hash}:
            raise ValueError("exact physical payload marker required")
        del target[location[-1]]
        if self._field.document_hash_at_path(fields, location) != envelope["logical_document_hash"]:
            raise ValueError("complete physical logical document seal changed")
        report_hash = fields.pop("report_hash", None)
        if report_hash != self._field.document_hash_at_path(fields, location):
            raise ValueError("original physical report seal changed")
        fields["report_hash"] = report_hash
        # Fresh independently owned payload; no sharing progressed state or
        # exposing the cached dictionary (there is no cached dictionary).
        encoded = self._field._payload_bytes
        if (
            type(encoded) is not bytes
            or "sha256:" + hashlib.sha256(encoded).hexdigest() != self._payload_hash
        ):
            raise ValueError("authenticated cached payload bytes required at restoration")
        target[location[-1]] = json.loads(encoded)
        self._verify_source()
        return fields
