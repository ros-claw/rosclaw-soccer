"""Independent toy physics re-integration, not neural or G1 skill qualification."""

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.recorded_feedback_replay import RecordedFeedbackReplay
from rosclaw_soccer.rsi.feedback_branch_control import G1FeedbackBranchControl
from rosclaw_soccer.sim.contracts import hash_bytes
from tests.physics.test_g1_feedback_forecast import control, world


def fixture(monkeypatch, offset=1.0):
    model, data, forecaster = world()

    def next_target(self, d, forces):
        self.consumed = True
        return np.full(29, 0.2), dict(
            joint_target_rad=np.full(29, 0.2),
            canonical_qpos=d.qpos.copy(),
            canonical_qvel=d.qvel.copy(),
        )

    monkeypatch.setattr(G1FeedbackBranchControl, "next_target", next_target)
    action = np.zeros(29)
    action[0] = offset
    out = forecaster.forecast(data, np.zeros(29), control=control(model), first_offset_nm=action)
    recorded = dict(
        out.arrays,
        offset_nm=action,
        end_integration_state=out.end_integration_state,
        **{"next_" + k: v for k, v in out.next_target_record.items()},
    )
    binary = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, binary)
    replay = RecordedFeedbackReplay(
        model,
        world_hash=hash_bytes(binary.tobytes()),
        kp=np.full(29, 10.0),
        kd=np.full(29, 0.2),
        maximum_steps=40,
    )
    return data, replay, recorded, out.donor_integration_state_hash


@pytest.mark.parametrize("offset", [0.0, 1.0, -1.0])
def test_reintegrates_full_two_target_window_without_neural_controller(monkeypatch, offset):
    d, r, a, donor = fixture(monkeypatch, offset)
    q, t = d.qpos.copy(), float(d.time)
    result = r.verify(d, first_target_rad=np.zeros(29), recorded=a, donor_hash=donor)
    assert result["all20_substeps_physically_replayed"]
    assert all(error == 0 for error in result["maximum_absolute_errors"].values())
    assert not result["next_neural_target_independently_recomputed"]
    assert not result["full_h300_teacher_qualified"]
    np.testing.assert_array_equal(d.qpos, q)
    assert d.time == t and r.attempted_steps == r.integrated_steps == 20
    r.verify(d, first_target_rad=np.zeros(29), recorded=a, donor_hash=donor)
    assert r.attempted_steps == r.integrated_steps == 40 and r.completed_windows == 2
    with pytest.raises(RuntimeError, match="exhausted"):
        r.verify(d, first_target_rad=np.zeros(29), recorded=a, donor_hash=donor)
    assert r.attempted_steps == 40


@pytest.mark.parametrize("bad", ["q", "force", "control", "foot", "nonfoot", "next_target", "end"])
def test_physical_mismatch_fails_closed_preserves_attempted_budget_and_donor(monkeypatch, bad):
    d, r, a, donor = fixture(monkeypatch)
    q = d.qpos.copy()
    if bad == "next_target":
        a["next_joint_target_rad"][0] += 1
    elif bad == "end":
        a["end_integration_state"][0] += 1
    elif bad == "nonfoot":
        a["nonfoot_robot_contact_over1n_after_substep"][0] = True
    else:
        key = {
            "q": "canonical_qpos_after_substep",
            "force": "robot_force_on_ball_world_n",
            "control": "applied_control_nm",
            "foot": "foot_contact_peak_force_n_after_substep",
        }[bad]
        a[key][0, 0] += 1
    with pytest.raises(ValueError, match="differs"):
        r.verify(d, first_target_rad=np.zeros(29), recorded=a, donor_hash=donor)
    assert r.attempted_steps == r.integrated_steps and 1 <= r.attempted_steps <= 20
    assert r.completed_windows == 0 and r.attempted_windows == 1
    np.testing.assert_array_equal(d.qpos, q)


@pytest.mark.parametrize("bad", ["hash", "cap", "nonfinite", "short", "dtype", "end_size"])
def test_invalid_inputs_rejected_before_any_physics(monkeypatch, bad):
    d, r, a, donor = fixture(monkeypatch)
    if bad == "hash":
        donor = "wrong"
    elif bad == "cap":
        a["offset_nm"][0] = 100
    elif bad == "nonfinite":
        a["robot_force_on_ball_world_n"][0, 0] = np.nan
    elif bad == "short":
        a["canonical_qpos_after_substep"] = a["canonical_qpos_after_substep"][:19]
    elif bad == "dtype":
        a["nonfoot_robot_contact_over1n_after_substep"] = np.zeros(20)
    else:
        a["end_integration_state"] = a["end_integration_state"][:-1]
    with pytest.raises(ValueError):
        r.verify(d, first_target_rad=np.zeros(29), recorded=a, donor_hash=donor)
    assert r.attempted_steps == r.integrated_steps == r.attempted_windows == 0
