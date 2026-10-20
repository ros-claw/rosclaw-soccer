"""Read-only scoring/continuity fixtures, not physical pilot outcomes."""

import copy
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.rsi_review_motion_entry_pilot import review_episode


def fixture(*, handoff=50):
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.rsi.bilateral_entry import observed_contact_foot
    from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
    from rosclaw_soccer.rsi.full_entry_forecast import score_full_entry

    names = [
        s + t for s in ("left", "right") for t in ("_ankle_roll_link", "_ankle_pitch_link")
    ] + ["pelvis"]
    m = NS(
        nbody=6, body_rootid=np.array([0, 5, 5, 5, 5, 5]), body=lambda n: NS(id=names.index(n) + 1)
    )
    q, v = np.zeros((300, 10, 43)), np.zeros((300, 10, 41))
    q[:, :, 2], q[:, :, 3], v[:, :, 35] = 0.8, 1.0, 0.6
    q[:, :, 36] = np.linspace(0, 1, 3000).reshape(300, 10)
    forces = np.zeros((300, 10, 6))
    forces[80, 0, 1] = 2.0
    stage = np.zeros(300, np.int64)
    if handoff is not None:
        stage[handoff:] = 1
    raw = dict(
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
        ball_contact_force_by_body_n=forces,
        neural_action_isaac=np.zeros((300, 29), np.float32),
        encoder_reference_poses=np.zeros((300, 10, 36)),
        entry_stage=stage,
        entry_command=np.zeros((300, 3)),
        entry_target_root_xy_m=np.zeros((300, 2)),
        entry_error_m=np.zeros(300),
    )
    pd = diagnose_entry_prefix(m, {k: x[:20] for k, x in raw.items()})
    s = score_full_entry(m, {k: x[20:] for k, x in raw.items()})
    s["original_suffix_qualified"], s["common_prefix_physical_safe"] = s["qualified"], True
    row = dict(
        reference_side="right",
        law_index=0,
        source_window=[466, 767],
        prefix_diagnostics=pd,
        score=s,
        causal_qualified=True,
        observed_contact_foot=observed_contact_foot(m, raw, global_start_frame=0),
        new_native_steps=3000,
        new_optimizer_steps=0,
        initial_state_adopted_from_other_episode=False,
        known_intent_before_frame0=True,
        report=dict(
            handoff_frame=handoff,
            source_window=[466, 767],
            source_playback_speed_changed=False,
            body=dict(frames=300, attempted_steps=3000, integrated_steps=3000),
        ),
    )
    return m, raw, row


def test_actual_source_tracking_contact_classified_separately():
    m, raw, row = fixture()
    r = review_episode(m, raw, row)
    assert r["qualified"] and r["first_touch_during_source_tracking"]
    assert not r["first_touch_without_source_tracking"] and r["handoff_to_contact_frames"] == 30


def test_navigation_contact_does_not_become_source_strike_claim():
    m, raw, row = fixture(handoff=None)
    r = review_episode(m, raw, row)
    assert r["qualified"] and r["first_touch_without_source_tracking"]
    assert not r["reference_handoff_occurred"] and not r["first_touch_during_source_tracking"]


@pytest.mark.parametrize("kind", ["jump", "target", "count", "score", "stage", "negative_force"])
def test_raw_mismatch_fails_closed(kind):
    m, raw, row = fixture()
    raw, row = copy.deepcopy(raw), copy.deepcopy(row)
    if kind == "jump":
        raw["canonical_qpos"][100, 0] = 1.0
    elif kind == "target":
        raw["neural_action_isaac"][100, 0] = 1.0
    elif kind == "count":
        row["new_native_steps"] = 2990
    elif kind == "score":
        row["score"]["qualified"] = False
    elif kind == "stage":
        raw["entry_stage"][40] = 1
    else:
        raw["ball_contact_force_by_body_n"][100, 0, 0] = -1
    with pytest.raises((ValueError, AssertionError)):
        review_episode(m, raw, row)
