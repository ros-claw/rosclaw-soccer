"""Combine audited actual-pass velocity-receiver traces without split leakage."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.actual_pass_transitions import actual_pass_transitions
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean

STATUSES = (
    (
        "ACTUAL_PASS_ROLE_TEACHER24_COLLECTED_PENDING_OWN_REPLAY",
        "ACTUAL_PASS_ROLE24_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS",
    ),
    (
        "ACTUAL_PASS_FOOT_CAPTURE_TEACHER24_COLLECTED_PENDING_OWN_REPLAY",
        "ACTUAL_PASS_FOOT_CAPTURE24_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS",
    ),
    (
        "PREALIGNED_FOOT_CAPTURE_TEACHER32_COLLECTED_PENDING_OWN_REPLAY",
        "PREALIGNED_FOOT_CAPTURE32_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS",
    ),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "audit"):
        parser.add_argument(f"--{name}", type=Path, action="append", required=True)
    for name in ("source-hash", "audit-hash"):
        parser.add_argument(f"--{name}", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(len(v) != 3 for v in (args.source, args.audit, args.source_hash, args.audit_hash))
    ):
        raise ValueError("exact three source/audit pairs and new external output required")
    parts, entries, excluded, pins = [], [], [], {}
    for ordinal, (root, audit_root, source_hash, audit_hash, statuses) in enumerate(
        zip(args.source, args.audit, args.source_hash, args.audit_hash, STATUSES, strict=True)
    ):
        source, audit_path = root / "complete.json", audit_root / "complete.json"
        if (
            source.is_symlink()
            or audit_path.is_symlink()
            or digest(source) != source_hash
            or digest(audit_path) != audit_hash
        ):
            raise ValueError("exact audited source pair required")
        producer, audit = (json.loads(p.read_text()) for p in (source, audit_path))
        if (
            producer["status"] != statuses[0]
            or audit["status"] != statuses[1]
            or audit["producer_hash"] != source_hash
            or producer["promotion_authorized"] is not False
        ):
            raise ValueError("corresponding unpromoted native replay audit required")
        pins[str(source)], pins[str(audit_path)] = source_hash, audit_hash
        for record in producer["records"]:
            arm = record["course"]["arm"]
            if arm == "forced_kick":
                excluded.append(
                    dict(
                        source=ordinal,
                        original_index=record["index"],
                        reason="receiver uses different neural action/observation interface",
                    )
                )
                continue
            if (
                arm
                not in {
                    "reactive_intercept",
                    "velocity_wait",
                    "velocity_intercept",
                    "velocity_capture",
                    "velocity_capture_clearance",
                }
                or type(record["qualified_capture"]) is not bool
            ):
                raise ValueError("declared velocity receiver and boolean episode label required")
            trace_path = root / record["artifact"]["path"]
            if (
                trace_path.is_symlink()
                or trace_path.parent != root
                or digest(trace_path) != record["artifact"]["hash"]
            ):
                raise ValueError("exact confined physical trace required")
            pins[str(trace_path)] = record["artifact"]["hash"]
            with np.load(trace_path, allow_pickle=False) as f:
                trace = {k: f[k].copy() for k in f.files if not f[k].dtype.hasobject}
            teacher = arm in {"velocity_capture", "velocity_capture_clearance"}
            values = actual_pass_transitions(trace, teacher=teacher)
            n, index = len(values["time_before"]), len(entries)
            values["course_index"] = np.full(n, index, dtype=np.int64)
            values["source_index"] = np.full(n, ordinal, dtype=np.int64)
            values["episode_qualified_capture"] = np.full(
                n, record["qualified_capture"], dtype=bool
            )
            parts.append(values)
            entries.append(
                dict(
                    index=index,
                    source=ordinal,
                    original_index=record["index"],
                    course=record["course"],
                    controls=n,
                    trace_hash=record["artifact"]["hash"],
                    teacher=teacher,
                    qualified_capture=record["qualified_capture"],
                )
            )
    if len(entries) != 72 or len(excluded) != 8 or any(e["qualified_capture"] for e in entries):
        raise ValueError(
            "exact 72 velocity traces, eight explicit exclusions and zero successes required"
        )
    dataset = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            source_hashes=args.source_hash,
            audit_hashes=args.audit_hash,
            source_head=args.expected_head,
            courses=72,
            excluded=excluded,
            grouping=(
                "four shared initial contexts; all arms and sources of each context stay together"
            ),
            actual_actions=(
                "teacher-applied action when available, otherwise velocity policy action"
            ),
            state="before action from initial or previous final native substep",
            hardware_authorized=False,
        ),
    )
    path = args.output / "dataset.npz"
    np.savez_compressed(path, **dataset)
    if any(digest(Path(p)) != h for p, h in pins.items()):
        raise ValueError("source changed during export")
    write_once(
        args.output / "complete.json",
        dict(
            status="ACTUAL_PASS_CAUSAL_TRANSITIONS_EXPORTED_NOT_TRAINED",
            source_head=args.expected_head,
            source_hashes=args.source_hash,
            audit_hashes=args.audit_hash,
            dataset_hash=digest(path),
            courses=entries,
            excluded=excluded,
            transitions=len(dataset["time_before"]),
            qualified_episodes=0,
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
