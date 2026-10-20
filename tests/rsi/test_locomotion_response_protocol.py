"""Synthetic protocol fixtures are not native locomotion evidence."""

import json
from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
from rosclaw_soccer.rsi.locomotion_response_protocol import (
    preflight_response_courses,
    response_courses,
    summarize_response,
)


def test_whole_bank_and_exogenous_pulse_schedule():
    bank = response_courses()
    assert len(bank) == 15 and sum(c.split == "train" for c in bank) == 9
    assert len(preflight_response_courses(bank)) == 71
    for row in bank:
        assert row.ball_course == (3.0, 3.0, 0.0)
        for frame in range(300):
            active = row.onset_frame <= frame < row.onset_frame + 100
            assert row.command(frame) == ((*row.command_xy_mps, 0.0) if active else (0.0, 0.0, 0.0))
        with pytest.raises(ValueError):
            row.command(300)


@pytest.mark.parametrize("change", ["missing", "reorder", "speed", "onset", "ball", "type"])
def test_no_course_replacement_or_silent_domain_expansion(change):
    bank = list(response_courses())
    if change == "missing":
        bank.pop()
    elif change == "reorder":
        bank.reverse()
    else:
        field = dict(
            speed={"command_xy_mps": (0.7, 0.0)},
            onset={"onset_frame": 51},
            ball={"ball_course": (0.4, 0.0, 0.0)},
            type={"index": True},
        )[change]
        bank[0] = replace(bank[0], **field)
    with pytest.raises(ValueError):
        preflight_response_courses(tuple(bank))


def fixture():
    c = response_courses()[1]
    q, v = np.zeros((300, 43)), np.zeros((300, 41))
    q[:, 2], q[:, 3] = 0.793, 1.0
    q[:, 7:36] = G1SonicRunupController.default_angles
    q[:, 36:39], q[:, 39] = [3.0, 3.0, 0.13], 1.0
    raw = dict(
        canonical_qpos=q,
        canonical_qvel=v,
        joint_target_rad=np.tile(G1SonicRunupController.default_angles, (300, 1)),
        pd_requested_torque_nm=np.zeros((300, 10, 29)),
        torque_nm=np.zeros((300, 10, 29)),
        actual_actuator_force_nm=np.zeros((300, 10, 29)),
        pelvis_z_per_substep_m=np.full((300, 10), 0.793),
        force_n=np.zeros((300, 6)),
        ball_position_after_step_m=q[:, 36:39].copy(),
        target_body_canonical_qpos_after_substep=np.repeat(q[:, None, :], 10, axis=1),
        target_body_canonical_qvel_after_substep=np.repeat(v[:, None, :], 10, axis=1),
        ball_contact_force_by_body_n=np.zeros((300, 10, 46)),
        neural_action_isaac=np.zeros((300, 29), dtype=np.float32),
        encoder_reference_poses=np.zeros((300, 10, 36)),
        navigation_command=np.array([c.command(f) for f in range(300)]),
    )
    return raw, c


def test_summary_never_promotes_a_standing_synthetic_record():
    raw, c = fixture()
    r = summarize_response(raw, c)
    assert r["stable_height_and_upright"] and r["no_robot_ball_contact"]
    assert r["pulse_displacement_xy_m"] == [0.0, 0.0]
    assert not r["football_skill_proven"] and not r["learned_policy_improvement_proven"]


@pytest.mark.parametrize("key", ["navigation_command", "canonical_qpos", "joint_target_rad"])
def test_corrupt_pulse_continuity_or_action_mapping_rejected(key):
    raw, c = fixture()
    raw[key][75, 0] += 0.01
    with pytest.raises((ValueError, AssertionError)):
        summarize_response(raw, c)


def test_actual_repeat_metadata_normalizes_only_json_containers_and_npz_hash():
    from scripts.rsi_collect_locomotion_response import compare_response_row

    row = dict(course=dict(command_xy_mps=(0.1, 0.0)), native_hash="first", native_steps=3000)
    saved = json.loads(json.dumps(row))
    saved["native_hash"] = "different_npz_container"
    compare_response_row(row, saved)
    saved["native_steps"] = 2999
    with pytest.raises(ValueError):
        compare_response_row(row, saved)
