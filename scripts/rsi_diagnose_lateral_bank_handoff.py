"""Diagnose audited development-bank receiving failures, without changing policy."""

import argparse
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.receiving_handoff_diagnosis import diagnose_no_handoff
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def summarize(rows: list[dict]) -> dict:
    """Report overlapping observations; never assert mutually exclusive causes."""
    first = Counter()
    for row in rows:
        groups = row["first_receiver_contact_force_groups"]
        first["none" if not groups else "+".join(groups)] += 1
    return {
        "no_handoff": len(rows),
        "passer_foot_over_10n": sum(r["actual_passer_foot_over_10n"] for r in rows),
        "receiver_contact_over_1n": sum(r["actual_receiver_contact_over_1n"] for r in rows),
        "excluded_contact_ever": sum(r["excluded_contact_ever"] for r in rows),
        "first_contact_groups": dict(sorted(first.items())),
        "within_0_65m_after_pass": sum(r["minimum_root_ball_xy_m"] <= 0.65 for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank",
        nargs=4,
        action="append",
        required=True,
        metavar=("SOURCE", "AUDIT", "SOURCE_HASH", "AUDIT_HASH"),
    )
    parser.add_argument("--scene-hash", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        not 1 <= len(args.bank) <= 8
        or len({bank[0] for bank in args.bank}) != len(args.bank)
        or args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
    ):
        raise ValueError("distinct audited banks and unused external output required")
    rows, frozen = [], []
    for source_name, audit_name, expected_source, expected_audit in args.bank:
        root, audit = Path(source_name), Path(audit_name)
        source, _, source_hash = report(
            root,
            audit,
            expected_audit,
            "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY",
            worlds=32,
        )
        scene = root / "scene.xml"
        if source_hash != expected_source or scene.is_symlink() or digest(scene) != args.scene_hash:
            raise ValueError("exact audited source and scene required")
        model = mujoco.MjModel.from_xml_path(str(scene))
        layout = NativeVelocityLayout(
            model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
        )
        for row in source["records"]:
            if row["handoff"] is not None:
                continue
            if row["termination"] != "NO_RECEIVING_HANDOFF_WITHIN_ORIGINAL_VELOCITY_H300":
                raise ValueError("exact H300 no-handoff trajectory required")
            path = bind(root, row, "physics")
            if path is None:
                raise ValueError("bound physical trajectory required")
            with np.load(path, allow_pickle=False) as archive:
                outcome = diagnose_no_handoff(
                    archive["qpos"],
                    archive["qvel"],
                    archive["ball_forces"],
                    archive["body_failed"],
                    receiver_q=layout.rq,
                    receiver_v=layout.rv,
                    ball_q=layout.bq,
                    ball_v=layout.bv,
                    excluded_contact=bool(row["result"]["excluded_contact_bodies"]),
                )
            if digest(path) != row["artifacts"]["physics"]["hash"]:
                raise ValueError("physical trajectory changed during diagnosis")
            if not outcome["no_handoff_reconstruction_consistent"]:
                raise ValueError("recorded readiness disagrees with archived samples")
            rows.append(
                dict(
                    source_hash=source_hash,
                    source_trial=row["index"],
                    course=row["course"],
                    physics_hash=row["artifacts"]["physics"]["hash"],
                    **outcome,
                )
            )
        frozen.append((root / "complete.json", source_hash, audit, expected_audit, scene))
    if any(
        digest(source) != source_hash
        or digest(audit) != audit_hash
        or digest(scene) != args.scene_hash
        for source, source_hash, audit, audit_hash, scene in frozen
    ):
        raise ValueError("audited source changed during diagnosis")
    clean(args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        {
            "status": "LATERAL_BANK_NO_HANDOFF_DIAGNOSIS_NOT_INTERVENTION",
            "source_head": args.expected_head,
            "summary": summarize(rows),
            "rows": rows,
            "scene_hash": args.scene_hash,
            "mutually_exclusive_causes": False,
            "independent_fresh": False,
            "promotion_authorized": False,
            "hardware_authorized": False,
            "goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
