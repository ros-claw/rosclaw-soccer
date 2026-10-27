"""Read back causal transition export against each audited source trace."""

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
    for name in ("source", "dataset", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("source-hash", "dataset-hash", "expected-head"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    source, dataset = args.source / "complete.json", args.dataset / "complete.json"
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or source.is_symlink()
        or dataset.is_symlink()
        or digest(source) != args.source_hash
        or digest(dataset) != args.dataset_hash
    ):
        raise ValueError("exact source and unused external audit output required")
    producer, receipt = (json.loads(p.read_text()) for p in (source, dataset))
    data_path = args.dataset / "dataset.npz"
    if (
        receipt["status"] != "ACTUAL_PASS_CAUSAL_TRANSITIONS_EXPORTED_NOT_TRAINED"
        or receipt["source_hash"] != args.source_hash
        or data_path.is_symlink()
        or digest(data_path) != receipt["dataset_hash"]
        or len(producer["records"]) != 32
    ):
        raise ValueError("complete bound source and transition dataset required")
    with np.load(data_path, allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    records = {r["index"]: r for r in producer["records"]}
    if len(records) != 32 or len(receipt["courses"]) != 32:
        raise ValueError("unique original courses required")
    cursor = 0
    rows = []
    for entry in receipt["courses"]:
        r = records[entry["index"]]
        path = args.source / r["artifact"]["path"]
        if (
            path.is_symlink()
            or path.parent != args.source
            or digest(path) != r["artifact"]["hash"]
            or entry["trace_hash"] != r["artifact"]["hash"]
            or entry["course"] != r["course"]
        ):
            raise ValueError("original physical trace and course commitment required")
        with np.load(path, allow_pickle=False) as f:
            trace = {k: f[k].copy() for k in f.files if not f[k].dtype.hasobject}
        teacher = r["course"]["arm"] in {"velocity_capture", "velocity_capture_clearance"}
        expected = actual_pass_transitions(trace, teacher=teacher)
        n = len(expected["time_before"])
        if (
            entry["controls"] != n
            or entry["teacher"] != teacher
            or entry["qualified_capture"] != r["qualified_capture"]
        ):
            raise ValueError("recorded teacher and episode identity required")
        if (
            any(not np.array_equal(data[k][cursor : cursor + n], v) for k, v in expected.items())
            or not np.array_equal(data["course_index"][cursor : cursor + n], np.full(n, r["index"]))
            or not np.array_equal(
                data["episode_qualified_capture"][cursor : cursor + n],
                np.full(n, r["qualified_capture"]),
            )
        ):
            raise ValueError("dataset differs from actual causal source trace")
        rows.append(
            dict(
                course_index=r["index"],
                controls=n,
                trace_hash=r["artifact"]["hash"],
                teacher=teacher,
            )
        )
        cursor += n
    if (
        cursor != receipt["transitions"]
        or any(len(a) != cursor for a in data.values())
        or len({r["course_index"] for r in rows}) != 32
    ):
        raise ValueError("all transition samples and course identities required")
    if (
        digest(source) != args.source_hash
        or digest(dataset) != args.dataset_hash
        or digest(data_path) != receipt["dataset_hash"]
    ):
        raise ValueError("source changed during readback")
    args.output.mkdir()
    write_once(
        args.output / "complete.json",
        dict(
            status="ACTUAL_PASS_CAUSAL_TRANSITIONS_READBACK_NOT_REINTEGRATED",
            source_hash=args.source_hash,
            dataset_hash=args.dataset_hash,
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
