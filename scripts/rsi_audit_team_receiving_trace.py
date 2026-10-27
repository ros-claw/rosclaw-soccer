"""Independently recompute receiving labels from full recorded SIM snapshots.

No physics is integrated and global rollout safety remains a source receipt;
the receiving window is recomputed from its original ball/contact/body fields.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.team_trace_capture import extract_team_trace
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean

PHYSICAL_KEYS = (
    "ball_pose",
    "ball_velocity",
    "ball_contact_agent_code",
    "ball_contact_effector_code",
    "ball_contact_force_n",
    "ball_nonfoot_contact_agent_code",
    "ball_nonfoot_contact_force_n",
    "receiving_feedback_qpos",
    "receiving_feedback_qvel",
)
ROSTER = tuple(
    sorted(
        f"{team}.{role}"
        for team in ("red", "blue")
        for role in ("defender", "finisher", "goalkeeper", "playmaker")
    )
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--retention-proof", type=Path)
    parser.add_argument("--retention-proof-hash")
    args = parser.parse_args()
    clean(args.expected_head)
    report_path = args.source / "complete.json"
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or report_path.is_symlink()
        or digest(report_path) != args.source_hash
    ):
        raise ValueError("unused external output and exact source receipt required")
    report = json.loads(report_path.read_text())
    expected_status = (
        "TEAM_RECEIVING_COVERAGE_PROBE_NOT_PROMOTED"
        if args.coverage
        else "TEAM_RECEIVING_EIGHT_BODY_CONSUMED_SMOKE_NOT_PROMOTED"
    )
    seeds = (223003, 238004, 252008, 255004, 258005, 258007) if args.coverage else (223001, 259011)
    expected_arms = (
        {"parent", *(f"probe-{i}" for i in range(16))}
        if args.coverage
        else {"parent", "17", "43", "71"}
    )
    if (
        report["status"] != expected_status
        or report["physical_runs"] != len(seeds) * len(expected_arms)
        or report["fresh_exam"] is not False
        or report["promotion_authorized"] is not False
    ):
        raise ValueError("complete unpromoted consumed experiment required")
    if [row["course_seed"] for row in report["results"]] != list(seeds):
        raise ValueError("exact predeclared course identity required")
    retention = {}
    if args.coverage:
        if not args.retention_proof or not args.retention_proof_hash:
            raise ValueError("coverage audit requires pinned current-parent retention proof")
        if (
            args.retention_proof.is_symlink()
            or digest(args.retention_proof) != args.retention_proof_hash
        ):
            raise ValueError("retention proof hash mismatch")
        proof = json.loads(args.retention_proof.read_text())
        if proof["status"] != "TEAM_RECEIVING_EIGHT_BODY_FULL_CONSUMED_NOT_PROMOTED":
            raise ValueError("complete unpromoted retention proof required")
        retention = {group["course_seed"]: group for group in proof["results"]}
    elif args.retention_proof or args.retention_proof_hash:
        raise ValueError("retention proof only applies to coverage")
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            source_hash=args.source_hash,
            source_head=args.expected_head,
            trace_count=len(seeds) * len(expected_arms),
            retention_proof_hash=args.retention_proof_hash,
            physical_integrations=0,
            scoring_window="red.finisher frames 20..119",
            hardware_authorized=False,
        ),
    )
    rows = []
    for group in report["results"]:
        arms = {row["arm"]: row for row in group["rows"]}
        if len(arms) != len(group["rows"]) or set(arms) != expected_arms:
            raise ValueError("exact unique experiment arms required")
        for arm, row in arms.items():
            path = args.source / row["trace_file"]
            if (
                path.is_symlink()
                or path.parent != args.source
                or digest(path) != row["trace_file_hash"]
            ):
                raise ValueError("exact confined raw trace required")
            with np.load(path, allow_pickle=False) as f:
                trace = {key: f[key].copy() for key in f.files}
            extracted = extract_team_trace(trace, require_tactical=args.coverage)
            if args.coverage and len(trace) != 106:
                raise ValueError("complete 106-field tactical trace required")
            if (
                hash_json({key: trace[key].tolist() for key in PHYSICAL_KEYS})
                != row["physical_trace_hash"]
            ):
                raise ValueError("original nine-field physical commitment mismatch")
            _, outcome = receiving_window(
                extracted, agent_ids=ROSTER, agent_id="red.finisher", start=20, frames=100
            )
            explanation = explain_receiving_window(
                extracted, agent_ids=ROSTER, agent_id="red.finisher", start=20, frames=100
            )
            if outcome["controlled_reception"] != row["controlled_reception"]:
                raise ValueError("recomputed receiving outcome differs from source")
            if args.coverage and arm == "parent":
                previous = next(
                    r for r in retention[group["course_seed"]]["rows"] if r["arm"] == "parent"
                )
                if row["physical_trace_hash"] != previous["physical_trace_hash"]:
                    raise ValueError("parent differs from current retention proof")
            if args.coverage and arm != "parent":
                expected_action = int(arm.removeprefix("probe-"))
                if (
                    row["activation_enabled"]
                    and not row["protected_episode"]
                    and (
                        row["selector_out_of_support"] or row["selector_action"] != expected_action
                    )
                ):
                    raise ValueError("forced action differs from recorded action")
            item = dict(
                course_seed=group["course_seed"],
                arm=arm,
                raw_trace_hash=row["trace_file_hash"],
                original_physical_hash=row["physical_trace_hash"],
                controlled_reception=outcome["controlled_reception"],
                window=outcome,
                criteria=explanation["criteria"],
                source_global_safety=row["safe"],
                global_safety_not_independently_recomputed=True,
            )
            rows.append(item)
            write_once(args.output / f"{group['course_seed']}-{arm}.json", item)
    write_once(
        args.output / "complete.json",
        dict(
            status="TEAM_RECEIVING_WINDOW_RECOMPUTED_NOT_REINTEGRATED",
            source_hash=args.source_hash,
            source_head=args.expected_head,
            rows=rows,
            physical_integrations=0,
            global_safety_not_independently_recomputed=True,
            coverage=args.coverage,
            retention_proof_hash=args.retention_proof_hash,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
