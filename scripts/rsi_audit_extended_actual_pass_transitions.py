"""Read back extended pass transitions against a base and three new blocks."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.actual_pass_transitions import actual_pass_transitions
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "base", "base-audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("dataset-hash", "base-hash", "base-audit-hash", "expected-head"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("source", "audit"):
        parser.add_argument(f"--{name}", type=Path, action="append", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    paths = (
        args.dataset / "complete.json",
        args.base / "complete.json",
        args.base_audit / "complete.json",
    )
    hashes = (args.dataset_hash, args.base_hash, args.base_audit_hash)
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or len(args.source) != 3
        or len(args.audit) != 3
        or any(p.is_symlink() or digest(p) != h for p, h in zip(paths, hashes, strict=True))
    ):
        raise ValueError("exact extended bank, base and three source pairs required")
    receipt, base, base_audit = (json.loads(p.read_text()) for p in paths)
    path, base_path = args.dataset / "dataset.npz", args.base / "dataset.npz"
    if (
        receipt["status"] != "ACTUAL_PASS_CAUSAL_TRANSITIONS_EXPORTED_NOT_TRAINED"
        or len(receipt["courses"]) != 168
        or receipt["base_hash"] != args.base_hash
        or receipt["base_audit_hash"] != args.base_audit_hash
        or base_audit["status"] != "ACTUAL_PASS_CAUSAL_TRANSITIONS_READBACK_NOT_REINTEGRATED"
        or base_audit["dataset_hash"] != args.base_hash
        or path.is_symlink()
        or base_path.is_symlink()
        or digest(path) != receipt["dataset_hash"]
        or digest(base_path) != base["dataset_hash"]
    ):
        raise ValueError("verified base and exact extended numeric bank required")
    source_paths = [root / "complete.json" for root in args.source]
    audit_paths = [root / "complete.json" for root in args.audit]
    if any(
        p.is_symlink() or digest(p) != h
        for p, h in zip(source_paths, receipt["new_source_hashes"], strict=True)
    ) or any(
        p.is_symlink() or digest(p) != h
        for p, h in zip(audit_paths, receipt["new_audit_hashes"], strict=True)
    ):
        raise ValueError("exact replayed source reports and audits required")
    reports = [json.loads(p.read_text()) for p in source_paths]
    audits = [json.loads(p.read_text()) for p in audit_paths]
    if any(
        a["status"] != "PREALIGNED_FOOT_CAPTURE_DEVELOPMENT32_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS"
        or a["producer_hash"] != h
        for a, h in zip(audits, receipt["new_source_hashes"], strict=True)
    ):
        raise ValueError("independent native+neural replay required for every new block")
    original = [{r["index"]: r for r in report["records"]} for report in reports]
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    with np.load(base_path, allow_pickle=False) as f:
        old = {k: f[k].copy() for k in f.files}
    nbase = base["transitions"]
    if (
        set(data) != set(old)
        or receipt["courses"][:72] != base["courses"]
        or any(not np.array_equal(data[k][:nbase], value) for k, value in old.items())
    ):
        raise ValueError("base transitions must be retained exactly")
    cursor, rows = nbase, []
    for entry in receipt["courses"][72:]:
        source_idx = entry["source"] - 3
        if type(source_idx) is not int or not 0 <= source_idx < 3:
            raise ValueError("explicit bounded new source identity required")
        record = original[source_idx][entry["original_index"]]
        trace_path = args.source[source_idx] / record["artifact"]["path"]
        if (
            trace_path.is_symlink()
            or trace_path.parent != args.source[source_idx]
            or digest(trace_path) != record["artifact"]["hash"]
            or entry["trace_hash"] != record["artifact"]["hash"]
            or entry["course"] != record["course"]
        ):
            raise ValueError("exact independently replayed physical trace required")
        with np.load(trace_path, allow_pickle=False) as f:
            trace = {k: f[k].copy() for k in f.files if not f[k].dtype.hasobject}
        teacher = record["course"]["arm"] in {"velocity_capture", "velocity_capture_clearance"}
        expected = actual_pass_transitions(trace, teacher=teacher)
        n = len(expected["time_before"])
        if (
            entry["controls"] != n
            or entry["teacher"] != teacher
            or entry["qualified_capture"] != record["qualified_capture"]
        ):
            raise ValueError("exact teacher and episode identity required")
        if (
            any(not np.array_equal(data[k][cursor : cursor + n], v) for k, v in expected.items())
            or not np.array_equal(
                data["course_index"][cursor : cursor + n], np.full(n, entry["index"])
            )
            or not np.array_equal(
                data["source_index"][cursor : cursor + n], np.full(n, entry["source"])
            )
            or not np.array_equal(
                data["episode_qualified_capture"][cursor : cursor + n],
                np.full(n, record["qualified_capture"]),
            )
        ):
            raise ValueError("extended transitions differ from recorded source")
        rows.append(
            dict(
                course_index=entry["index"],
                source=entry["source"],
                original_index=record["index"],
                controls=n,
                trace_hash=record["artifact"]["hash"],
            )
        )
        cursor += n
    if (
        cursor != receipt["transitions"]
        or any(len(v) != cursor for v in data.values())
        or [r["course_index"] for r in rows] != list(range(72, 168))
    ):
        raise ValueError("all unique new transition samples required")
    if (
        any(digest(p) != h for p, h in zip(paths, hashes, strict=True))
        or digest(path) != receipt["dataset_hash"]
        or digest(base_path) != base["dataset_hash"]
        or any(
            digest(p) != h for p, h in zip(source_paths, receipt["new_source_hashes"], strict=True)
        )
        or any(
            digest(p) != h for p, h in zip(audit_paths, receipt["new_audit_hashes"], strict=True)
        )
    ):
        raise ValueError("source changed during readback")
    args.output.mkdir()
    write_once(
        args.output / "complete.json",
        dict(
            status="ACTUAL_PASS_CAUSAL_TRANSITIONS_READBACK_NOT_REINTEGRATED",
            source_head=args.expected_head,
            dataset_hash=args.dataset_hash,
            base_hash=args.base_hash,
            new_source_hashes=receipt["new_source_hashes"],
            new_audit_hashes=receipt["new_audit_hashes"],
            rows=rows,
            transitions=cursor,
            same_alignment_implementation=True,
            independent_physics_reintegrated=False,
            new_source_independent_physics_replayed=True,
            actor_trained=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
