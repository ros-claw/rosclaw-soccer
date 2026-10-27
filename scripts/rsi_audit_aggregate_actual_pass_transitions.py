"""Read back all aggregate pass transitions against their original SIM traces."""

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
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-hash", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    receipt_path = args.dataset / "complete.json"
    if (
        len(args.source) != 3
        or args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or receipt_path.is_symlink()
        or digest(receipt_path) != args.dataset_hash
    ):
        raise ValueError("three exact sources and unused external audit required")
    receipt = json.loads(receipt_path.read_text())
    data_path = args.dataset / "dataset.npz"
    if (
        receipt["status"] != "ACTUAL_PASS_CAUSAL_TRANSITIONS_EXPORTED_NOT_TRAINED"
        or len(receipt["courses"]) != 72
        or data_path.is_symlink()
        or digest(data_path) != receipt["dataset_hash"]
    ):
        raise ValueError("complete bound 72-course transition dataset required")
    source_paths = [root / "complete.json" for root in args.source]
    if any(
        p.is_symlink() or digest(p) != h
        for p, h in zip(source_paths, receipt["source_hashes"], strict=True)
    ):
        raise ValueError("exact original source reports required")
    reports = [json.loads(p.read_text()) for p in source_paths]
    original = [{r["index"]: r for r in report["records"]} for report in reports]
    with np.load(data_path, allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    cursor, rows = 0, []
    for entry in receipt["courses"]:
        source_idx = entry["source"]
        if type(source_idx) is not int or not 0 <= source_idx < 3:
            raise ValueError("bounded recorded source identity required")
        record = original[source_idx][entry["original_index"]]
        path = args.source[source_idx] / record["artifact"]["path"]
        if (
            path.is_symlink()
            or path.parent != args.source[source_idx]
            or digest(path) != record["artifact"]["hash"]
            or entry["trace_hash"] != record["artifact"]["hash"]
            or entry["course"] != record["course"]
        ):
            raise ValueError("exact recorded physical source trace required")
        with np.load(path, allow_pickle=False) as f:
            trace = {k: f[k].copy() for k in f.files if not f[k].dtype.hasobject}
        teacher = record["course"]["arm"] in {"velocity_capture", "velocity_capture_clearance"}
        expected = actual_pass_transitions(trace, teacher=teacher)
        n = len(expected["time_before"])
        if (
            entry["controls"] != n
            or entry["teacher"] != teacher
            or entry["qualified_capture"] != record["qualified_capture"]
        ):
            raise ValueError("recorded role and episode identity required")
        if (
            any(not np.array_equal(data[k][cursor : cursor + n], v) for k, v in expected.items())
            or not np.array_equal(
                data["course_index"][cursor : cursor + n], np.full(n, entry["index"])
            )
            or not np.array_equal(data["source_index"][cursor : cursor + n], np.full(n, source_idx))
            or not np.array_equal(
                data["episode_qualified_capture"][cursor : cursor + n],
                np.full(n, record["qualified_capture"]),
            )
        ):
            raise ValueError("aggregate data differs from causal recorded source")
        rows.append(
            dict(
                course_index=entry["index"],
                source=source_idx,
                original_index=record["index"],
                controls=n,
                trace_hash=record["artifact"]["hash"],
            )
        )
        cursor += n
    if (
        cursor != receipt["transitions"]
        or any(len(v) != cursor for v in data.values())
        or len({r["course_index"] for r in rows}) != 72
    ):
        raise ValueError("complete unique aggregate transition bank required")
    if (
        digest(receipt_path) != args.dataset_hash
        or digest(data_path) != receipt["dataset_hash"]
        or any(digest(p) != h for p, h in zip(source_paths, receipt["source_hashes"], strict=True))
    ):
        raise ValueError("source changed during readback")
    args.output.mkdir()
    write_once(
        args.output / "complete.json",
        dict(
            status="ACTUAL_PASS_CAUSAL_TRANSITIONS_READBACK_NOT_REINTEGRATED",
            dataset_hash=args.dataset_hash,
            source_hashes=receipt["source_hashes"],
            source_head=args.expected_head,
            rows=rows,
            transitions=cursor,
            same_alignment_implementation=True,
            independent_physics_reintegrated=False,
            actor_trained=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
