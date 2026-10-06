"""Publish complete JSON artifacts atomically, without replacing existing data.

POSIX local-filesystem experiment utility. A waiter must never observe the
destination before the whole document has been written and fsynced.
"""

import gzip
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.sim.contracts import hash_json


def write_once(path: Path, value: dict[str, Any]) -> None:
    expected = hash_json(value)  # Reject nonfinite/nonserializable input first.

    def check_existing() -> None:
        if hash_json(load_json_artifact(path)) != expected:
            raise ValueError(f"resume commitment differs: {path}")

    if path.exists():
        check_existing()
        return
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        if path.name.endswith(".json.gz"):
            with os.fdopen(descriptor, "wb") as binary:
                with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as compressed:
                    text = io.TextIOWrapper(compressed, encoding="utf-8")
                    try:
                        json.dump(value, text, indent=2, sort_keys=True, allow_nan=False)
                        text.write("\n")
                        text.flush()
                    finally:
                        text.detach()
                binary.flush()
                os.fsync(binary.fileno())
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        try:
            # Unlike replace/rename-overwrite, link fails if another writer has
            # already published. Both writers must agree on the whole payload.
            os.link(temporary, path)
        except FileExistsError:
            check_existing()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        # Only this invocation's exact private temporary path is removed.
        temporary.unlink(missing_ok=True)


def write_shared_physical_report(path: Path, value: dict[str, Any]) -> None:
    """Publish one complete model proof per run; retain every logical field.

    No old artifact is converted or removed. Parent reports without a neural proof
    remain complete ordinary gzip JSON. Readers reconstruct before normal audits.
    """
    from rosclaw.growth.shared_proof_payload import detach_payload

    from rosclaw_soccer.rsi.physical_report_io import (
        CPU_MEAN_MODEL_LOCATION,
        CPU_MODEL_LOCATION,
        MEAN_MODEL_LOCATION,
        MODEL_LOCATION,
    )

    if path.name != "report.json.gz" or path.parent.is_symlink():
        raise ValueError("new local compressed physical report required")
    keys = [
        k for k in ("contact_motor_policy", "executed_motor_policy") if value.get(k) is not None
    ]
    if len(keys) > 1:
        raise ValueError("one unambiguous physical policy proof required")
    key = keys[0] if keys else "contact_motor_policy"
    policy = value.get(key)
    if policy is not None and type(policy) is not dict:
        raise ValueError("physical policy must be a dictionary or absent")
    step = policy.get("step_motor_proof") if policy else None
    if step is not None and type(step) is not dict:
        raise ValueError("physical step proof must be a dictionary or absent")
    proof = step.get("model") if step else None
    if proof is None:
        write_once(path, value)
        return
    # Sampling views retain their own seed/noise/likelihood commitments in
    # every report while sharing only the unchanged complete mean model.
    if type(proof) is not dict:
        raise ValueError("complete physical model dictionary required")
    direct, mean = (
        (CPU_MODEL_LOCATION, CPU_MEAN_MODEL_LOCATION)
        if key == "executed_motor_policy"
        else (MODEL_LOCATION, MEAN_MODEL_LOCATION)
    )
    location = mean if type(proof.get("mean_model")) is dict else direct
    envelope, payload = detach_payload(value, location)
    store = path.parent.parent / ".shared-models"
    if store.is_symlink():
        raise ValueError("local content-addressed model store required")
    store.mkdir(exist_ok=True)
    payload_path = store / f"{envelope['payload_hash'][7:]}.json.gz"
    if payload_path.is_symlink():
        raise ValueError("shared proof cannot be a symlink")
    # Publish payload first: no visible report may reference an incomplete file.
    write_once(payload_path, payload)
    write_once(path, envelope)


def write_shared_sampling_model(path: Path, value: dict[str, Any]) -> None:
    """Retain seed/noise/logical seal per view; share its WHOLE mean model."""
    from rosclaw.growth.shared_proof_payload import detach_payload

    if (
        path.parent.name != "models"
        or not path.name.endswith(".json.gz")
        or path.parent.is_symlink()
        or path.parent.parent.is_symlink()
        or value.get("schema")
        not in (
            "soccer.rsi.smooth_memory_sampling.v1",
            "soccer.rsi.output_memory_step_sampling.v1",
            "soccer.rsi.recurrent_motor_sampling.v1",
        )
        or value.get("model_hash")
        != hash_json({k: v for k, v in value.items() if k != "model_hash"})
    ):
        raise ValueError("sealed local memory sampling model required")
    if value.get("schema") == "soccer.rsi.recurrent_motor_sampling.v1":
        from rosclaw_soccer.rsi.recurrent_sampling_motor import make_preview

        make_preview(value)
    envelope, payload = detach_payload(value, ("mean_model",))
    store = path.parent.parent / ".shared-models"
    if store.is_symlink():
        raise ValueError("local sampling proof store required")
    store.mkdir(exist_ok=True)
    payload_path = store / f"{envelope['payload_hash'][7:]}.json.gz"
    if payload_path.is_symlink():
        raise ValueError("sampling mean cannot be a symlink")
    write_once(payload_path, payload)
    write_once(path, envelope)
