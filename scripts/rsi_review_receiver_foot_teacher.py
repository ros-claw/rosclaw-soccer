"""Review paired foot authority only after independent physical replay."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.paired_precontact_authority import precontact_tracking_error
from rosclaw_soccer.rsi.passer_curriculum_outcome import strict_pass_receive
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report
from scripts.rsi_review_whole_receiver_identity import artifact_equal


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "lateral-source", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "audit-hash", "lateral-hash", "lateral-audit-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external evidence output required")
    root, audit = a.source / "source", a.source / "audit/complete.json"
    old, old_audit = a.lateral_source / "source", a.lateral_source / "audit/complete.json"
    declaration = json.loads((root / "declaration.json").read_text())
    early_intercept = (
        declaration.get("protocol")
        == "soccer.neural_relay_receiver_early_intercept.development.v28"
    )
    axis_local = early_intercept or (
        declaration.get("protocol")
        == "soccer.neural_relay_receiver_axis_local_fallback.development.v25"
    )
    predictive = axis_local or (
        declaration.get("protocol")
        == "soccer.neural_relay_receiver_predictive_foot_shin.development.v24"
    )
    source, _, sh = report(
        root,
        audit,
        a.audit_hash,
        "NEURAL_RELAY_RECEIVER_EARLY_INTERCEPT16_PENDING_OWN_REPLAY"
        if early_intercept
        else "NEURAL_RELAY_RECEIVER_AXIS_LOCAL_FALLBACK16_PENDING_OWN_REPLAY"
        if axis_local
        else "NEURAL_RELAY_RECEIVER_PREDICTIVE_FOOT_SHIN16_PENDING_OWN_REPLAY"
        if predictive
        else "NEURAL_RELAY_RECEIVER_FOOT_TEACHER16_PENDING_OWN_REPLAY",
        worlds=16,
    )
    prior, _, oh = report(
        old,
        old_audit,
        a.lateral_audit_hash,
        "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    if sh != a.source_hash or oh != a.lateral_hash:
        raise ValueError("exact source and prior independently replayed studies required")
    if (
        declaration.get("receiver_foot_teacher_is_unlearned") is not True
        or declaration.get("development_context_block") != 0
        or predictive
        and declaration.get("receiver_predictive_foot_shin_is_unlearned") is not True
        or axis_local
        and declaration.get("receiver_axis_local_fallback") is not True
        or early_intercept
        and declaration.get("receiver_early_foot_intercept_is_unlearned") is not True
    ):
        raise ValueError("exact unlearned development-only foot teacher required")
    index = {
        (
            r["course"]["context"],
            r["course"]["lateral_m"],
            r["course"]["arm"]["receiver_foot_teacher_strength"],
        ): r
        for r in source["records"]
    }
    old_index = {
        (r["course"]["context"], r["course"]["lateral_m"]): r
        for r in prior["records"]
        if r["course"]["arm"]["receiver_lateral_offset_m"] == 0.12
    }
    if len(index) != 16 or len(old_index) != 8:
        raise ValueError("complete eight-condition paired study required")
    model = mujoco.MjModel.from_xml_path(str(root / "scene.xml"))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    rows = []
    for key in sorted(old_index):
        base, teacher = index[(*key, 0.0)], index[(*key, 0.5)]
        if not artifact_equal(root, base, old, old_index[key], "initial") or not artifact_equal(
            root, base, root, teacher, "initial"
        ):
            raise ValueError("same initial physical conditions required")
        identity = artifact_equal(root, base, old, old_index[key], "physics")
        outcomes = {}
        for name, row in (("baseline", base), ("teacher", teacher)):
            physical, velocity = bind(root, row, "physics"), bind(root, row, "receiver_velocity")
            if physical is None or velocity is None:
                raise ValueError("actual physical and teacher histories required")
            with (
                np.load(physical, allow_pickle=False) as raw,
                np.load(velocity, allow_pickle=False) as v,
            ):
                outcome = strict_pass_receive(
                    raw["ball_forces"],
                    raw["body_failed"],
                    handoff=row["result"]["handoff_executed"],
                )
                delta = v["FOOT_applied_delta"]
                if delta.shape != (len(v["control"]), 29) or not np.isfinite(delta).all():
                    raise ValueError("complete bounded executed teacher record required")
                changes = np.flatnonzero(np.any(delta != 0, axis=1))
                if name == "baseline" and len(changes):
                    raise ValueError("zero teacher must never alter a joint target")
                jump_matrix = np.abs(np.diff(np.vstack((np.zeros((1, 29)), delta)), axis=0))
                jumps = np.max(jump_matrix, axis=1)
                slew = 0.025 if predictive and name == "teacher" else 0.015
                fallback = v["FOOT_fallback"]
                if fallback.shape != (len(delta),) or fallback.dtype != np.bool_:
                    raise ValueError("complete boolean teacher fallback history required")
                early = v["FOOT_early_active"] if early_intercept and name == "teacher" else None
                if early is not None and (early.shape != (len(delta),) or early.dtype != np.bool_):
                    raise ValueError("complete early-activation history required")
                outcomes[name] = dict(
                    outcome=outcome,
                    precontact_tracking=precontact_tracking_error(
                        raw["target_b"],
                        raw["qpos"],
                        raw["ball_forces"],
                        joint_indices=layout.qi,
                    ),
                    active_controls=int(v["FOOT_active"].sum()),
                    early_active_controls=int(early.sum()) if early is not None else 0,
                    changed_controls=len(changes),
                    max_delta_rad=float(np.abs(delta).max()),
                    max_correction_jump_rad=float(jumps.max()),
                    max_total_joint_jump_rad=float(jump_matrix.sum(axis=1).max()),
                    slew_exceeding_controls=int((jumps > slew + 1e-9).sum()),
                    fallback_jump_controls=int(((jumps > slew + 1e-9) & fallback).sum()),
                    precise_healthy_return=row["result"]["return_0p1m_and_healthy"],
                )
        rows.append(
            dict(
                context=key[0],
                side=key[1],
                zero_teacher_matches_lateral_baseline=identity,
                **outcomes,
            )
        )
    summary = {
        name: dict(
            strict_clean=sum(r[name]["outcome"]["strict_both_clean_handoff"] for r in rows),
            foot_after_pass=sum(r[name]["outcome"]["passer_foot_precedes_receiver"] for r in rows),
            nonfoot=sum(r[name]["outcome"]["receiver_nonfoot_substep"] is not None for r in rows),
            healthy=sum(r[name]["outcome"]["both_bodies_healthy"] for r in rows),
            precise=sum(r[name]["precise_healthy_return"] for r in rows),
            changed_worlds=sum(r[name]["changed_controls"] > 0 for r in rows),
            slew_violation_worlds=sum(r[name]["slew_exceeding_controls"] > 0 for r in rows),
            fallback_jump_controls=sum(r[name]["fallback_jump_controls"] for r in rows),
            max_total_joint_jump_rad=max(r[name]["max_total_joint_jump_rad"] for r in rows),
        )
        for name in ("baseline", "teacher")
    }
    if any(
        digest(p) != h
        for p, h in (
            (root / "complete.json", sh),
            (audit, a.audit_hash),
            (old / "complete.json", oh),
            (old_audit, a.lateral_audit_hash),
        )
    ):
        raise ValueError("source and audit changed during review")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_FOOT_TEACHER_REPLAY_REVIEWED",
            source_head=a.expected_head,
            source_hash=sh,
            audit_hash=a.audit_hash,
            prior_hash=oh,
            zero_identity_count=sum(r["zero_teacher_matches_lateral_baseline"] for r in rows),
            summary=summary,
            records=rows,
            teacher_is_unlearned=True,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
