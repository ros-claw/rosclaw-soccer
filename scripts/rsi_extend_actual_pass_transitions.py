"""Extend a readback-verified pass dataset with three independently replayed blocks."""

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
    for name in ("base", "base-audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("base-hash", "base-audit-hash", "expected-head"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("source", "audit"):
        parser.add_argument(f"--{name}", type=Path, action="append", required=True)
    for name in ("source-hash", "audit-hash"):
        parser.add_argument(f"--{name}", action="append", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    base_path, base_audit_path = args.base / "complete.json", args.base_audit / "complete.json"
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(len(v) != 3 for v in (args.source, args.audit, args.source_hash, args.audit_hash))
        or base_path.is_symlink()
        or base_audit_path.is_symlink()
        or digest(base_path) != args.base_hash
        or digest(base_audit_path) != args.base_audit_hash
    ):
        raise ValueError("exact base, three source/audit pairs and new external output required")
    base, base_audit = (json.loads(p.read_text()) for p in (base_path, base_audit_path))
    base_data = args.base / "dataset.npz"
    if (
        base["status"] != "ACTUAL_PASS_CAUSAL_TRANSITIONS_EXPORTED_NOT_TRAINED"
        or len(base["courses"]) != 72
        or base_audit["status"] != "ACTUAL_PASS_CAUSAL_TRANSITIONS_READBACK_NOT_REINTEGRATED"
        or base_audit["dataset_hash"] != args.base_hash
        or base_data.is_symlink()
        or digest(base_data) != base["dataset_hash"]
    ):
        raise ValueError("readback-verified base transition bank required")
    with np.load(base_data, allow_pickle=False) as f:
        data = {k: [f[k].copy()] for k in f.files}
    entries = list(base["courses"])
    pins = {
        str(base_path): args.base_hash,
        str(base_audit_path): args.base_audit_hash,
        str(base_data): base["dataset_hash"],
    }
    blocks = ((2, 3, 5, 7), (8, 9, 11, 13), (15, 16, 17, 18))
    for block, (root, audit_root, source_hash, audit_hash) in enumerate(
        zip(args.source, args.audit, args.source_hash, args.audit_hash, strict=True)
    ):
        source, audit_path = root / "complete.json", audit_root / "complete.json"
        if (
            source.is_symlink()
            or audit_path.is_symlink()
            or digest(source) != source_hash
            or digest(audit_path) != audit_hash
        ):
            raise ValueError("exact new source/audit pair required")
        report, audit = (json.loads(p.read_text()) for p in (source, audit_path))
        if (
            report["status"] != "PREALIGNED_FOOT_CAPTURE_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
            or len(report["records"]) != 32
            or audit["status"]
            != "PREALIGNED_FOOT_CAPTURE_DEVELOPMENT32_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS"
            or audit["producer_hash"] != source_hash
            or {r["course"]["context"] for r in report["records"]} != set(blocks[block])
        ):
            raise ValueError("all 32 physically replayed records in the declared block required")
        pins[str(source)], pins[str(audit_path)] = source_hash, audit_hash
        for record in report["records"]:
            arm = record["course"]["arm"]
            if (
                arm
                not in {
                    "reactive_intercept",
                    "velocity_intercept",
                    "velocity_capture",
                    "velocity_capture_clearance",
                }
                or type(record["qualified_capture"]) is not bool
            ):
                raise ValueError("recorded velocity receiver and episode outcome required")
            path = root / record["artifact"]["path"]
            if (
                path.is_symlink()
                or path.parent != root
                or digest(path) != record["artifact"]["hash"]
            ):
                raise ValueError("exact confined recorded physical trace required")
            pins[str(path)] = record["artifact"]["hash"]
            with np.load(path, allow_pickle=False) as f:
                trace = {k: f[k].copy() for k in f.files if not f[k].dtype.hasobject}
            teacher = arm in {"velocity_capture", "velocity_capture_clearance"}
            values = actual_pass_transitions(trace, teacher=teacher)
            n, index = len(values["time_before"]), len(entries)
            values["course_index"] = np.full(n, index, dtype=np.int64)
            values["source_index"] = np.full(n, block + 3, dtype=np.int64)
            values["episode_qualified_capture"] = np.full(
                n, record["qualified_capture"], dtype=bool
            )
            if set(values) != set(data):
                raise ValueError("same explicit causal data schema required")
            for key, value in values.items():
                data[key].append(value)
            entries.append(
                dict(
                    index=index,
                    source=block + 3,
                    original_index=record["index"],
                    course=record["course"],
                    controls=n,
                    trace_hash=record["artifact"]["hash"],
                    teacher=teacher,
                    qualified_capture=record["qualified_capture"],
                )
            )
    if len(entries) != 168 or len({e["course"]["context"] for e in entries}) != 16:
        raise ValueError("168 courses across sixteen declared shared contexts required")
    combined = {k: np.concatenate(v) for k, v in data.items()}
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            base_hash=args.base_hash,
            base_audit_hash=args.base_audit_hash,
            source_hashes=args.source_hash,
            audit_hashes=args.audit_hash,
            source_head=args.expected_head,
            courses=168,
            contexts=16,
            grouping="all arms and source variants sharing an initial context remain together",
            state="actual before action",
            postaction_contacts="outcome only",
            hardware_authorized=False,
        ),
    )
    dataset_path = args.output / "dataset.npz"
    np.savez_compressed(dataset_path, **combined)
    if any(digest(Path(p)) != h for p, h in pins.items()):
        raise ValueError("source changed during extended export")
    write_once(
        args.output / "complete.json",
        dict(
            status="ACTUAL_PASS_CAUSAL_TRANSITIONS_EXPORTED_NOT_TRAINED",
            source_head=args.expected_head,
            base_hash=args.base_hash,
            base_audit_hash=args.base_audit_hash,
            new_source_hashes=args.source_hash,
            new_audit_hashes=args.audit_hash,
            dataset_hash=digest(dataset_path),
            courses=entries,
            transitions=len(combined["time_before"]),
            qualified_episodes=sum(e["qualified_capture"] for e in entries),
            teacher_active_transitions=int(combined["teacher_active"].sum()),
            context_count=16,
            pass_causally_recorded=True,
            optimizer_updates=0,
            independent_physics_reintegrated=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
