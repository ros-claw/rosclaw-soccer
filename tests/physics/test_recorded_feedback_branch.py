"""Synthetic native toy equivalence only, not G1 football performance."""

import copy

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecaster
from rosclaw_soccer.physics.recorded_feedback_branch import RecordedFeedbackBranch
from rosclaw_soccer.sim.contracts import hash_bytes
from tests.physics.test_g1_target_episode import body


def fixture():
    episode = body()
    for f in range(31):
        episode.step(f, np.zeros(29))
    binary = np.zeros(mujoco.mj_sizeModel(episode.model), dtype=np.uint8)
    mujoco.mj_saveModel(episode.model, None, binary)
    wh = hash_bytes(binary.tobytes())
    reference = G1PDTorqueForecaster(
        episode.model, expected_world_hash=wh, kp=episode.kp, kd=episode.kd, max_forecast_steps=40
    )
    owned = mujoco.MjData(episode.model)
    mujoco.mj_copyData(owned, episode.model, episode.data)
    records = []
    targets = [np.full(29, j * 0.001) for j in range(4)]
    for j in range(4):
        offset = np.zeros(29)
        if j < 3:
            offset[0] = 0.1
        result = reference.forecast(owned, targets[j], offset)
        mujoco.mj_copyData(owned, episode.model, reference._scratch)
        records.append(result)
    fields = (
        "canonical_qpos_after_substep",
        "canonical_qvel_after_substep",
        "requested_pd_torque_nm",
        "applied_control_nm",
        "actual_actuator_force_nm",
        "pelvis_height_solver_epoch_m",
        "robot_force_on_ball_world_n",
        "completed_foot_ground_normal_n",
    )
    raw = {k: np.concatenate([getattr(r, k) for r in records]) for k in fields}
    raw["requested_offset_nm"] = np.zeros((40, 29))
    raw["requested_offset_nm"][:30, 0] = 0.1
    raw["end_integration_state"] = records[-1].end_integration_state.copy()
    raw.update({f"next_{j}_joint_target_rad": targets[j].copy() for j in range(1, 4)})
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        episode.model, owned, mujoco.mjtObj.mjOBJ_BODY, episode.ball, velocity, 0
    )
    return episode, wh, raw, targets[0], velocity[3:]


def test_native_branch_end_and_cached_velocity_exact_donor_unchanged():
    episode, wh, raw, target, velocity = fixture()
    before = episode.canonical()
    verifier = RecordedFeedbackBranch(episode, world_hash=wh)
    result = verifier.verify(raw, current_target=target)
    assert result["actual_new_native_steps"] == 40
    assert all(v == 0 for v in result["maximum_absolute_errors"].values())
    np.testing.assert_array_equal(result["cached_terminal_ball_velocity_m_s"], velocity)
    for a, b in zip(before, episode.canonical(), strict=True):
        np.testing.assert_array_equal(a, b)
    assert episode.next_frame == 31 and episode.attempted_steps == 310
    with pytest.raises(ValueError, match="single-use"):
        verifier.verify(raw, current_target=target)


def test_physical_mismatch_is_terminal_and_charged():
    episode, wh, raw, target, _ = fixture()
    wrong = copy.deepcopy(raw)
    wrong["canonical_qpos_after_substep"][15, 0] += 1
    verifier = RecordedFeedbackBranch(episode, world_hash=wh)
    with pytest.raises(ValueError, match="differs"):
        verifier.verify(wrong, current_target=target)
    assert verifier.native.integrated_steps == 20
    with pytest.raises(ValueError, match="single-use"):
        verifier.verify(raw, current_target=target)


def test_no_time_zero_body_with_patched_clock():
    episode = body()
    episode.next_frame = 31
    with pytest.raises(ValueError, match="prefix"):
        RecordedFeedbackBranch(episode, world_hash="sha256:" + "0" * 64)


def test_cancelled_law_cannot_be_mislabelled_as_unconditional():
    episode, wh, raw, target, _ = fixture()
    raw["requested_offset_nm"][20:30].fill(0)
    verifier = RecordedFeedbackBranch(episode, world_hash=wh)
    with pytest.raises(ValueError, match="offset law"):
        verifier.verify(raw, current_target=target)
    assert verifier.native.attempted_steps == 0
