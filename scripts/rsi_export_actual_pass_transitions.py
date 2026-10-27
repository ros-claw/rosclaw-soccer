"""Export causally aligned transitions from audited actual-pass SIM traces."""

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
    for name in ("source", "audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("source-hash", "audit-hash", "expected-head"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    source_path, audit_path = args.source / "complete.json", args.audit / "complete.json"
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or source_path.is_symlink()
        or audit_path.is_symlink()
        or digest(source_path) != args.source_hash
        or digest(audit_path) != args.audit_hash
    ):
        raise ValueError("exact source, audit and unused external output required")
    report, audit = (json.loads(p.read_text()) for p in (source_path, audit_path))
    if (
        report["status"] != "PREALIGNED_FOOT_CAPTURE_TEACHER32_COLLECTED_PENDING_OWN_REPLAY"
        or len(report["records"]) != 32
        or audit["status"] != "PREALIGNED_FOOT_CAPTURE32_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS"
        or audit["producer_hash"] != args.source_hash
    ):
        raise ValueError("32 audited actual-pass development traces required")
    allowed = {
        "reactive_intercept",
        "velocity_intercept",
        "velocity_capture",
        "velocity_capture_clearance",
    }
    parts, entries = [], []
    for record in report["records"]:
        arm = record["course"]["arm"]
        if (
            arm not in allowed
            or type(record["index"]) is not int
            or type(record["qualified_capture"]) is not bool
        ):
            raise ValueError("explicit recorded role and outcome required")
        path = args.source / record["artifact"]["path"]
        if (
            path.is_symlink()
            or path.parent != args.source
            or digest(path) != record["artifact"]["hash"]
        ):
            raise ValueError("exact confined source trace required")
        with np.load(path, allow_pickle=False) as f:
            trace = {k: f[k].copy() for k in f.files if not f[k].dtype.hasobject}
        teacher = arm in {"velocity_capture", "velocity_capture_clearance"}
        values = actual_pass_transitions(trace, teacher=teacher)
        n = len(values["time_before"])
        values["course_index"] = np.full(n, record["index"], dtype=np.int64)
        values["episode_qualified_capture"] = np.full(n, record["qualified_capture"], dtype=bool)
        parts.append(values)
        entries.append(
            dict(
                index=record["index"],
                course=record["course"],
                controls=n,
                trace_hash=record["artifact"]["hash"],
                teacher=teacher,
                qualified_capture=record["qualified_capture"],
            )
        )
    if len({e["index"] for e in entries}) != 32 or set(
        entries[i]["index"] for i in range(32)
    ) != set(range(32)):
        raise ValueError("complete unique original trace identities required")
    dataset = {key: np.concatenate([item[key] for item in parts]) for key in parts[0]}
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            source_hash=args.source_hash,
            audit_hash=args.audit_hash,
            source_head=args.expected_head,
            courses=32,
            world="two physical G1 plus one shared ball",
            state="before action; initial state or last prior native substep",
            action="actual teacher-applied action where present, otherwise velocity policy action",
            outcome="ten post-action native substeps; do not use as same-step observation",
            episode_qualification="post-episode label, not per-step input",
            hardware_authorized=False,
        ),
    )
    path = args.output / "dataset.npz"
    np.savez_compressed(path, **dataset)
    if digest(source_path) != args.source_hash or digest(audit_path) != args.audit_hash:
        raise ValueError("source changed during export")
    write_once(
        args.output / "complete.json",
        dict(
            status="ACTUAL_PASS_CAUSAL_TRANSITIONS_EXPORTED_NOT_TRAINED",
            source_hash=args.source_hash,
            audit_hash=args.audit_hash,
            source_head=args.expected_head,
            dataset_hash=digest(path),
            courses=entries,
            transitions=len(dataset["time_before"]),
            qualified_episodes=sum(e["qualified_capture"] for e in entries),
            teacher_active_transitions=int(dataset["teacher_active"].sum()),
            pass_causally_recorded=True,
            optimizer_updates=0,
            independent_physics_reintegrated=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
