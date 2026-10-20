"""Synthetic causal-boundary audit checks, not G1 dynamic feasibility."""

import copy
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.rsi_review_compact_source import audit_candidate


def fixture(*, handoff=40):
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.rsi.bilateral_entry import observed_contact_foot
    from rosclaw_soccer.rsi.student_intervention_score import (
        fuse_physical_intervention30,
        score_student_intervention30,
    )

    names = [
        s + t for s in ("left", "right") for t in ("_ankle_roll_link", "_ankle_pitch_link")
    ] + ["pelvis"]
    m = NS(
        nbody=6, body_rootid=np.array([0, 5, 5, 5, 5, 5]), body=lambda n: NS(id=names.index(n) + 1)
    )
    q, v = np.zeros((300, 10, 43)), np.zeros((300, 10, 41))
    q[:, :, 2], q[:, :, 3], v[:, :, 35] = 0.8, 1.0, 0.6
    q[:, :, 36] = np.linspace(0, 1, 3000).reshape(300, 10)
    f = np.zeros((300, 10, 6))
    f[50, 0, 1] = 2.0
    fields = dict(
        canonical_qpos=np.concatenate((q[0, :1], q[:-1, -1])),
        canonical_qvel=np.concatenate((v[0, :1], v[:-1, -1])),
        joint_target_rad=np.tile(G1SonicRunupController.default_angles, (300, 1)),
        pd_requested_torque_nm=np.zeros((300, 10, 29)),
        torque_nm=np.zeros((300, 10, 29)),
        actual_actuator_force_nm=np.zeros((300, 10, 29)),
        pelvis_z_per_substep_m=np.full((300, 10), 0.8),
        force_n=np.zeros((300, 6)),
        ball_position_after_step_m=q[:, -1, 36:39].copy(),
        target_body_canonical_qpos_after_substep=q,
        target_body_canonical_qvel_after_substep=v,
        ball_contact_force_by_body_n=f,
    )
    p = {k: v[:30].copy() for k, v in fields.items()}
    data = {k: v[30:].copy() for k, v in fields.items()}
    stage = np.zeros(270, np.int64)
    if handoff is not None:
        stage[handoff - 30 :] = 1
    data.update(
        neural_action_isaac=np.zeros((270, 29), np.float32),
        encoder_reference_poses=np.zeros((270, 10, 36)),
        entry_stage=stage,
        entry_command=np.zeros((270, 3)),
        entry_target_root_xy_m=np.zeros((270, 2)),
        entry_error_m=np.zeros(270),
    )
    joined = fuse_physical_intervention30(m, p, data)
    c = dict(
        report=dict(
            source_window=[466, 767],
            source_playback_speed_changed=False,
            source_lead_protocol="compact60_before_declared526_event_original120Hz",
            actual_entry_start_frame=30,
            original_task_horizon=300,
            body=dict(attempted_steps=3000, integrated_steps=3000),
            handoff_frame=handoff,
            handoff_native_state_unchanged_hash="fixture",
        ),
        new_integrated_native_steps=2700,
        new_attempted_native_steps=2700,
        score=score_student_intervention30(m, p, data),
        observed_contact_foot=observed_contact_foot(m, joined, global_start_frame=0),
    )
    return m, p, data, c, copy.deepcopy(data), copy.deepcopy(c)


def test_unchanged_pre_handoff_and_bound_native_fields_accepted():
    m, p, data, c, old, oldc = fixture()
    assert len(audit_candidate(m, p, data, c, old, oldc)) == 12


@pytest.mark.parametrize(
    "kind", ["hash", "handoff", "early_reference", "boundary_q", "invalid_stage", "torque_target"]
)
def test_source_crop_cannot_hide_other_changes(kind):
    m, p, data, c, old, oldc = fixture()
    if kind == "hash":
        oldc["report"]["handoff_native_state_unchanged_hash"] = "changed"
    elif kind == "handoff":
        oldc["report"]["handoff_frame"] = 39
    elif kind == "early_reference":
        old["encoder_reference_poses"][5, 0, 0] = 1.0
    elif kind == "boundary_q":
        old["canonical_qpos"][10, 0] = 1.0
    elif kind == "invalid_stage":
        data["entry_stage"][2] = 1
    else:
        data["neural_action_isaac"][50, 0] = 1.0
    with pytest.raises((ValueError, AssertionError)):
        audit_candidate(m, p, data, c, old, oldc)


def test_no_handoff_requires_whole270_to_stay_numerically_identical():
    m, p, data, c, old, oldc = fixture(handoff=None)
    audit_candidate(m, p, data, c, old, oldc)
    old["encoder_reference_poses"][200, 0, 0] = 1.0
    with pytest.raises(AssertionError, match="bytes differ"):
        audit_candidate(m, p, data, c, old, oldc)
