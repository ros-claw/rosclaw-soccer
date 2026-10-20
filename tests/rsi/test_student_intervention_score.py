"""Synthetic original-gate/continuity tests, not evidence of physical recovery."""

from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
from rosclaw_soccer.rsi.full_entry_forecast import score_full_entry
from rosclaw_soccer.rsi.student_intervention_score import (
    fuse_physical_intervention30,
    score_student_intervention30,
)


def fixture():
    names = [
        s + t for s in ("left", "right") for t in ("_ankle_roll_link", "_ankle_pitch_link")
    ] + ["pelvis"]
    model = SimpleNamespace(
        nbody=6,
        body_rootid=np.array([0, 5, 5, 5, 5, 5]),
        body=lambda n: SimpleNamespace(id=names.index(n) + 1),
    )
    q, v = np.zeros((300, 10, 43)), np.zeros((300, 10, 41))
    q[:, :, 2], q[:, :, 3] = 0.8, 1.0
    q[:, :, 36] = np.linspace(0, 1.0, 3000).reshape(300, 10)
    v[:, :, 35] = 0.6
    canonical_q = np.concatenate((q[0, :1], q[:-1, -1]))
    canonical_v = np.concatenate((v[0, :1], v[:-1, -1]))
    forces = np.zeros((300, 10, 6))
    forces[50, 0, 1] = 2.0
    data = dict(
        canonical_qpos=canonical_q,
        canonical_qvel=canonical_v,
        joint_target_rad=np.zeros((300, 29)),
        pd_requested_torque_nm=np.zeros((300, 10, 29)),
        torque_nm=np.zeros((300, 10, 29)),
        actual_actuator_force_nm=np.zeros((300, 10, 29)),
        pelvis_z_per_substep_m=np.full((300, 10), 0.8),
        force_n=np.zeros((300, 6)),
        ball_position_after_step_m=q[:, -1, 36:39].copy(),
        target_body_canonical_qpos_after_substep=q,
        target_body_canonical_qvel_after_substep=v,
        ball_contact_force_by_body_n=forces,
    )
    return (
        model,
        {k: v[:30].copy() for k, v in data.items()},
        {k: v[30:].copy() for k, v in data.items()},
    )


def test_original280_gate_and_student10_are_retained():
    m, p, s = fixture()
    full = fuse_physical_intervention30(m, p, s)
    old = score_full_entry(m, {k: v[20:] for k, v in full.items()})
    new = score_student_intervention30(m, p, s)
    assert old["qualified"] and new["qualified"]
    assert new["first_foot_contact_global_frame"] == 50
    assert new["cost"] == old["cost"]
    with pytest.raises(ValueError, match="frame20..299"):
        score_full_entry(m, s)
    p["ball_contact_force_by_body_n"][25, 0, 5] = 2
    assert not score_student_intervention30(m, p, s)["qualified"]


def test_preintervention_touch_cannot_be_claimed_as_teacher_first_touch():
    m, p, s = fixture()
    p["ball_contact_force_by_body_n"][25, 0, 1] = 2
    result = score_student_intervention30(m, p, s)
    assert result["original_complete280_qualified"]
    assert not result["qualified"] and result["cost"] is None
    assert not result["post_boundary_control_can_cause_first_touch"]
    assert result["first_foot_contact_global_frame"] == 25


@pytest.mark.parametrize(
    "kind", ["short", "float32", "nonfinite", "negative", "q_jump", "v_jump", "signed_zero"]
)
def test_no_dropped_steps_nonfinite_or_physical_boundary_reset(kind):
    m, p, s = fixture()
    if kind == "short":
        s["canonical_qpos"] = s["canonical_qpos"][:-1]
    elif kind == "float32":
        s["torque_nm"] = s["torque_nm"].astype(np.float32)
    elif kind == "nonfinite":
        p["force_n"][29, 0] = np.nan
    elif kind == "negative":
        p["ball_contact_force_by_body_n"][0, 0, 0] = -1.0
    elif kind == "q_jump":
        s["canonical_qpos"][0, 0] += 0.01
    elif kind == "v_jump":
        s["canonical_qvel"][0, 0] += 0.01
    else:
        s["canonical_qvel"][0, 0] = -0.0
    with pytest.raises(ValueError):
        fuse_physical_intervention30(m, p, s)


@pytest.mark.parametrize("frames", [True, 30.0, 29, 0, 300])
def test_unqualified_prefix_boundary_rejected_before_data_access(frames):
    with pytest.raises(ValueError, match="intervention30"):
        diagnose_entry_prefix(None, {}, frames=frames)
