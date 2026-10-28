"""Inspect one-step foot teacher feasibility on audited, unmodified relay states.

This is *not* a counterfactual rollout: proposed corrections are never applied.
Each control uses a fresh zero-delta teacher, so no hypothetical private history
is carried across the original trajectory.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_foot_contact_teacher import (
    FootContactTeacherConfig,
    NativeFootContactTeacher,
)
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "review", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "audit-hash", "review-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    root = a.source / "source"
    audit = a.source / "audit/complete.json"
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.review.is_symlink()
        or digest(a.review) != a.review_hash
        or digest(root / "complete.json") != a.source_hash
    ):
        raise ValueError("unused external output and exact completed evidence required")
    source, _, source_hash = report(
        root,
        audit,
        a.audit_hash,
        "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    review = json.loads(a.review.read_text())
    if (
        source_hash != a.source_hash
        or review.get("source_hash") != a.source_hash
        or review.get("audit_hash") != a.audit_hash
        or review.get("status") != "RECEIVER_LATERAL_AUTHORITY_DEVELOPMENT_REVIEWED"
        or review.get("promotion_authorized") is not False
    ):
        raise ValueError("matching independent relay replay and review required")
    model = mujoco.MjModel.from_xml_path(str(root / "scene.xml"))
    data = mujoco.MjData(model)
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    ball_geoms = np.flatnonzero(model.geom_bodyid == layout.ball)
    if len(ball_geoms) != 1:
        raise ValueError("one actual ball geometry required")
    feet = [model.body("b/" + side + "_ankle_roll_link").id for side in ("left", "right")]
    config = FootContactTeacherConfig(strength=0.5)
    records = []
    for row in source["records"]:
        if row["course"]["arm"]["receiver_lateral_offset_m"] != 0.12:
            continue
        initial_path, physical_path, velocity_path = [
            bind(root, row, k) for k in ("initial", "physics", "receiver_velocity")
        ]
        if any(path is None for path in (initial_path, physical_path, velocity_path)):
            raise ValueError("complete own initial, physics and velocity traces required")
        with (
            np.load(initial_path, allow_pickle=False) as initial,
            np.load(physical_path, allow_pickle=False) as raw,
            np.load(velocity_path, allow_pickle=False) as velocity,
        ):
            n = len(velocity["control"])
            if not np.array_equal(velocity["control"], np.arange(n)) or n > len(raw["qpos"]):
                raise ValueError("complete pre-handoff control-aligned trace required")
            passer_seen = False
            first_receiver = None
            active, changed, fallback, changed_before_contact = [], [], [], []
            for step in range(n):
                data.qpos[:] = initial["qpos"] if step == 0 else raw["qpos"][step - 1, -1]
                data.qvel[:] = initial["qvel"] if step == 0 else raw["qvel"][step - 1, -1]
                mujoco.mj_forward(model, data)
                teacher = NativeFootContactTeacher(config)
                teacher.next_frame = step
                proposed = teacher.propose(
                    model=model,
                    snapshot=data,
                    layout=layout,
                    nominal=raw["target_b"][step].astype(np.float64),
                    ball_geom=int(ball_geoms[0]),
                    foot_bodies=feet,
                    enabled=passer_seen,
                    frame=step,
                )
                if proposed["active"]:
                    active.append(step)
                if proposed["fallback"]:
                    fallback.append(step)
                if np.any(proposed["applied_delta"] != 0):
                    changed.append(step)
                    if first_receiver is None:
                        changed_before_contact.append(step)
                f = raw["ball_forces"][step]
                if first_receiver is None and ((f[:, 1, :2] > 10).any() or (f[:, 1, 2] > 1).any()):
                    first_receiver = step
                passer_seen |= bool((f[:, 0, :2] > 10).any())
            records.append(
                dict(
                    context=row["course"]["context"],
                    side=row["course"]["lateral_m"],
                    controls=n,
                    first_receiver_contact_control=first_receiver,
                    active_controls=len(active),
                    changed_controls=len(changed),
                    fallback_controls=len(fallback),
                    changed_before_contact=len(changed_before_contact),
                    first_active_control=active[0] if active else None,
                    first_changed_control=changed[0] if changed else None,
                )
            )
    if (
        len(records) != 8
        or digest(root / "complete.json") != a.source_hash
        or digest(audit) != a.audit_hash
        or digest(a.review) != a.review_hash
    ):
        raise ValueError("all eight unchanged audited baseline conditions required")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="OFFLINE_RECEIVER_FOOT_TEACHER_SHADOW_ONLY",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            audit_hash=a.audit_hash,
            review_hash=a.review_hash,
            config=asdict(config),
            records=records,
            one_step_zero_history=True,
            teacher_outputs_applied=False,
            new_physics_steps=0,
            hypothetical_history_not_simulated=True,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    print(json.dumps(records))


if __name__ == "__main__":
    main()
