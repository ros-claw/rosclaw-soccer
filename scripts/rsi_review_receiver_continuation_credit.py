"""Compare fixed sparse and safety-gated rewards on replayed development worlds."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.precontact_feedback import full_history_precontact_feedback
from rosclaw_soccer.rsi.receiver_continuation_credit import receiver_continuation_credit
from rosclaw_soccer.rsi.receiver_safe_continuation_credit import receiver_safe_continuation_credit
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def returns(rewards):
    values = np.empty(len(rewards), dtype=np.float64)
    current = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        current = float(rewards[i]) + 0.99 * current
        values[i] = current
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--audit-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external evidence output required")
    source, _, source_hash = report(
        a.source,
        a.audit,
        a.audit_hash,
        "NEURAL_RELAY_CHILD_VELOCITY_CANDIDATE_SAMPLING32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    scene = a.source / "scene.xml"
    scene_hash = digest(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    rows = []
    for row in source["records"]:
        seed = row["course"]["arm"]["exploration_seed"]
        if seed is None:
            continue
        if seed not in (92117, 92143, 92171):
            raise ValueError("predeclared stochastic seeds only")
        physical, velocity = (
            bind(a.source, row, "physics"),
            bind(a.source, row, "receiver_velocity"),
        )
        if physical is None or velocity is None:
            raise ValueError("actual physical and own-action histories required")
        handoff = row["handoff"]
        direction = (
            np.asarray(handoff["return_goal"][:2]) - np.asarray(handoff["kick_origin"][:2])
            if handoff
            else np.array([1.0, 0.0])
        )
        direction = direction / np.linalg.norm(direction)
        with np.load(physical, allow_pickle=False) as f, np.load(velocity, allow_pickle=False) as v:
            controls = len(v["control"])
            if not np.array_equal(v["control"], np.arange(controls)):
                raise ValueError("complete consecutive own action history required")
            old, _ = full_history_precontact_feedback(
                f["ball_forces"][:, :, 1, :], f["body_failed"], controls=controls
            )
            kwargs = dict(
                controls=controls,
                handoff_control=handoff["control"] if handoff else None,
                ball_velocity_index=layout.bv,
                return_direction=direction,
                excluded_contact_observed=bool(row["result"]["excluded_contact_bodies"]),
            )
            v1, v1_e = receiver_continuation_credit(
                f["ball_forces"], f["qvel"], f["body_failed"], **kwargs
            )
            v2, v2_e = receiver_safe_continuation_credit(
                f["ball_forces"],
                f["qvel"],
                f["body_failed"],
                termination=row["termination"],
                **kwargs,
            )
            score = {
                name: returns(reward) for name, reward in (("old", old), ("v1", v1), ("v2", v2))
            }
            unsafe = row["termination"] == "BODY_FAILURE" or kwargs["excluded_contact_observed"]
            if unsafe and (score["v2"] > 1e-7).any():
                raise ValueError("safety-gated v2 must never credit observed failure")
            rows.append(
                dict(
                    index=row["index"],
                    course=row["course"],
                    termination=row["termination"],
                    result=row["result"],
                    initial_return={k: float(x[0]) for k, x in score.items()},
                    positive_future_return={k: bool((x > 1e-7).any()) for k, x in score.items()},
                    v1_events=v1_e["events"],
                    v2_evidence=v2_e,
                    physics_hash=row["artifacts"]["physics"]["hash"],
                    velocity_hash=row["artifacts"]["receiver_velocity"]["hash"],
                )
            )
    if len(rows) != 24 or digest(scene) != scene_hash:
        raise ValueError("complete fixed sample and unchanged scene required")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="RECEIVER_CREDIT_DEVELOPMENT_DIAGNOSTIC_ONLY",
            source_hash=source_hash,
            audit_hash=a.audit_hash,
            scene_hash=scene_hash,
            code_head=a.expected_head,
            records=rows,
            trained=False,
            physical_improvement_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
