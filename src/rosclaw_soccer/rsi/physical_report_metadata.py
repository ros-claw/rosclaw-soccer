"""Source-pinned shared report metadata and optional complete JSON restoration.

All whole-document and root seals are verified. Payload files are byte-pinned
and checked even on cached reads. The original full report reader is unchanged.
Reading numerical policy data never executes, validates or authorizes a policy.
"""

import hashlib
import re
from pathlib import Path
from typing import Any

import rosclaw.growth.compiled_shared_metadata as compiled_module
import rosclaw.growth.shared_proof_payload as payload_module

from rosclaw_soccer.rsi import json_artifact_io, physical_report_io
from rosclaw_soccer.sim import contracts
from rosclaw_soccer.sim.contracts import hash_bytes


class PhysicalReportMetadataReader:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[compiled_module.CompiledSharedMetadata, str]] = {}
        self._files: dict[str, str] = {}
        self._pins = {
            str(p): hash_bytes(p.read_bytes())
            for p in (
                Path(__file__),
                Path(compiled_module.__file__),
                Path(payload_module.__file__),
                Path(json_artifact_io.__file__),
                Path(physical_report_io.__file__),
                Path(contracts.__file__),
            )
        }

    def _stable_sources(self) -> None:
        if any(hash_bytes(Path(p).read_bytes()) != h for p, h in self._pins.items()):
            raise ValueError("fixed metadata reader dependency source changed")

    def read(self, path: Path) -> dict[str, Any]:
        """Return verified compact metadata, never a complete numerical policy."""
        return self._read(path, complete=False)

    def restore(self, path: Path) -> dict[str, Any]:
        """Return all original JSON fields after whole seals and file-byte checks.

        Explicit opt-in; requires Core's complete shared-payload restoration.
        Callers still pin sources before import, validate physical evidence and
        enforce authority separately. No metadata fields are added to the
        original report, and each result owns its complete numerical data.
        """
        return self._read(path, complete=True)

    def _read(self, path: Path, *, complete: bool) -> dict[str, Any]:
        if complete and not callable(
            getattr(compiled_module.CompiledSharedMetadata, "restore", None)
        ):
            raise ValueError("Core complete shared JSON restoration required")
        self._stable_sources()
        resolved = physical_report_io.resolve_physical_report(path)
        if resolved.is_symlink() or resolved.parent.is_symlink():
            raise ValueError("local non-symlink report required")
        envelope_hash = hash_bytes(resolved.read_bytes())
        envelope = json_artifact_io.load_json_artifact(resolved)
        digest = envelope.get("payload_hash")
        if (
            envelope.get("schema") != physical_report_io.SHARED_SCHEMA
            or envelope.get("location")
            not in (
                list(physical_report_io.MODEL_LOCATION),
                list(physical_report_io.MEAN_MODEL_LOCATION),
                list(physical_report_io.CPU_MODEL_LOCATION),
                list(physical_report_io.CPU_MEAN_MODEL_LOCATION),
            )
            or type(digest) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("exact shared physical model location and identity required")
        store = resolved.parent.parent / ".shared-models"
        payload_path = store / f"{digest[7:]}.json.gz"
        if store.is_symlink() or payload_path.is_symlink() or not payload_path.is_file():
            raise ValueError("complete local non-symlink shared proof required")
        file_hash = hash_bytes(payload_path.read_bytes())
        key = str(payload_path)
        if key not in self._cache:
            payload = json_artifact_io.load_json_artifact(payload_path)
            verifier = compiled_module.CompiledSharedMetadata(payload)
            if file_hash != hash_bytes(payload_path.read_bytes()):
                raise ValueError("shared file changed during allocation")
            self._cache[key] = (verifier, file_hash)
        verifier, expected_file_hash = self._cache[key]
        if file_hash != expected_file_hash:
            raise ValueError("cached shared proof bytes changed")
        result: dict[str, Any] = (
            verifier.restore(envelope, sealed_field="report_hash")
            if complete
            else verifier.verify(envelope, sealed_field="report_hash")
        )
        if (
            any(p.is_symlink() for p in (resolved, resolved.parent, store, payload_path))
            or not payload_path.is_file()
        ):
            raise ValueError("local non-symlink shared proof changed during verification")
        if envelope_hash != hash_bytes(resolved.read_bytes()):
            raise ValueError("report envelope changed during verification")
        if file_hash != hash_bytes(payload_path.read_bytes()):
            raise ValueError("shared proof file changed during verification")
        for p, h in ((str(resolved), envelope_hash), (key, file_hash)):
            if p in self._files and self._files[p] != h:
                raise ValueError("previously observed report input changed")
            self._files[p] = h
        if not complete:
            result.update(
                source_report_envelope_path=str(resolved),
                source_report_envelope_file_hash=envelope_hash,
                source_shared_payload_path=key,
                source_shared_payload_file_hash=file_hash,
                metadata_not_physical_replay=True,
            )
        self._stable_sources()
        return result

    def input_file_hashes(self) -> dict[str, str]:
        self._stable_sources()
        result = dict(self._files)
        result.update(self._pins)
        if any(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() != h[7:] for p, h in result.items()
        ):
            raise ValueError("observed metadata input or source changed")
        return result
