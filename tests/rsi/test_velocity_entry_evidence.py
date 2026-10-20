"""Synthetic measured-entry evidence; no policy improvement claim."""

import numpy as np
import pytest
from test_velocity_policy_native_evidence import fixture

from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance, ball_entry_decision
from rosclaw_soccer.rsi.velocity_entry_evidence import review_entry


def entry_fixture():
    raw, p, s, _ = fixture()
    course = (0.378, -0.235, 0.0)
    raw["canonical_qpos"][:, 36:38] = course[:2]
    raw["target_body_canonical_qpos_after_substep"][:, :, 36:38] = course[:2]
    raw["ball_position_after_step_m"][:, :2] = course[:2]
    prior = SkillEntryAffordance((0.416, -0.129), 150, 1.7, "sha256:" + "0" * 64)
    decisions = [
        ball_entry_decision(q, v, frame=f, affordance=prior)
        for f, (q, v) in enumerate(zip(raw["canonical_qpos"], raw["canonical_qvel"], strict=True))
    ]
    raw.update(
        entry_target_root_xy_m=np.array([d.target_root_xy_m for d in decisions]),
        entry_error_m=np.array([d.horizontal_error_m for d in decisions]),
        entry_ready=np.array([d.ready for d in decisions], dtype=np.bool_),
        entry_timed_out=np.array([d.timed_out for d in decisions], dtype=np.bool_),
    )
    return raw, p, s, prior, course


def test_sham_uses_zero_motor_command_but_reviews_actual_entry_geometry():
    raw, p, s, prior, course = entry_fixture()
    r = review_entry(raw, p, s, prior, course, "idle")
    assert r["physical_response_guard"]
    assert not r["qualified_entry_readiness"]
    assert not r["motor_handoff_performed"]
    assert not r["football_skill_proven"]
    assert r["latest_ready_frame"] == 90
    assert r["entry_radius_m"] == 0.04
    assert r["minimum_entry_error_frames30_to90_m"] > 0.1


@pytest.mark.parametrize("kind", ["gate", "dtype", "error", "missing", "command", "initial"])
def test_entry_evidence_mutations_fail_closed(kind):
    raw, p, s, prior, course = entry_fixture()
    arm = "idle"
    if kind == "gate":
        raw["entry_ready"][30] = True
    elif kind == "dtype":
        raw["entry_ready"] = raw["entry_ready"].astype(np.int64)
    elif kind == "error":
        raw["entry_error_m"][50] = 0.0
    elif kind == "missing":
        del raw["entry_timed_out"]
    elif kind == "command":
        arm = "measured_entry"
    else:
        course = (0.4, -0.235, 0.0)
    with pytest.raises(ValueError):
        review_entry(raw, p, s, prior, course, arm)


def test_unknown_arm_not_silently_zeroed():
    raw, p, s, prior, course = entry_fixture()
    with pytest.raises(ValueError):
        review_entry(raw, p, s, prior, course, "adaptive")
