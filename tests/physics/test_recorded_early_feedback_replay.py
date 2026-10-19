"""Independent80ms toy integration; no causal NN or actual G1 qualification."""

import numpy as np
import pytest

from rosclaw_soccer.physics.recorded_feedback_replay import RecordedFeedbackReplay
from tests.physics.test_g1_early_feedback_forecast import fake_controller
from tests.physics.test_g1_early_feedback_forecast import fixture as forecast_fixture
from tests.physics.test_g1_feedback_forecast import control


def fixture(monkeypatch, offset=1.0):
    model, data, f = forecast_fixture()
    fake_controller(monkeypatch, model, data)
    action = np.zeros(29)
    action[0] = offset
    out = f.forecast(data, np.zeros(29), control=control(model), early_offset_nm=action)
    a = dict(out.arrays, end_integration_state=out.end_integration_state)
    for j, record in enumerate(out.next_target_records, start=1):
        a[f"next_{j}_joint_target_rad"] = record["joint_target_rad"].copy()
        a[f"next_{j}_canonical_qpos"] = a["canonical_qpos_after_substep"][j * 10 - 1].copy()
        a[f"next_{j}_canonical_qvel"] = a["canonical_qvel_after_substep"][j * 10 - 1].copy()
    # Reuse the already admitted compiled-model identity, not a new world.
    import mujoco

    from rosclaw_soccer.sim.contracts import hash_bytes

    binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, binary)
    replay = RecordedFeedbackReplay(
        model,
        world_hash=hash_bytes(binary.tobytes()),
        kp=np.full(29, 10.0),
        kd=np.full(29, 0.2),
        maximum_steps=80,
        early80=True,
    )
    return data, replay, a, out.donor_integration_state_hash


@pytest.mark.parametrize("offset", [0.0, 1.0, -1.0])
def test_independent_three_recorded_target_schedule_and_complete_end_state(monkeypatch, offset):
    d, replay, a, donor = fixture(monkeypatch, offset)
    before = d.qpos.copy()
    result = replay.verify(d, first_target_rad=np.zeros(29), recorded=a, donor_hash=donor)
    assert result["all40_substeps_physically_replayed"]
    assert max(result["maximum_absolute_errors"].values()) == 0
    assert not result["next_neural_target_independently_recomputed"]
    assert not result["full_h300_teacher_qualified"]
    np.testing.assert_array_equal(d.qpos, before)
    assert d.time == 0.6 and replay.integrated_steps == replay.attempted_steps == 40
    replay.verify(d, first_target_rad=np.zeros(29), recorded=a, donor_hash=donor)
    assert replay.integrated_steps == 80 and replay.completed_windows == 2
    with pytest.raises(RuntimeError, match="exhausted"):
        replay.verify(d, first_target_rad=np.zeros(29), recorded=a, donor_hash=donor)


@pytest.mark.parametrize("j", [1, 2, 3])
def test_every_recorded_future_target_actually_controls_physics(monkeypatch, j):
    d, replay, a, donor = fixture(monkeypatch)
    a[f"next_{j}_joint_target_rad"][0] += 0.1
    with pytest.raises(ValueError, match="differs"):
        replay.verify(d, first_target_rad=np.zeros(29), recorded=a, donor_hash=donor)
    assert replay.attempted_steps == replay.integrated_steps == j * 10 + 1
    assert replay.failed
    with pytest.raises(ValueError):
        replay.verify(d, first_target_rad=np.zeros(29), recorded=a, donor_hash=donor)
    assert replay.attempted_steps == j * 10 + 1


@pytest.mark.parametrize("bad", ["short", "washout", "force", "nonfinite"])
def test_malformed_whole80ms_is_rejected_without_physics(monkeypatch, bad):
    d, replay, a, donor = fixture(monkeypatch)
    if bad == "short":
        a["canonical_qpos_after_substep"] = a["canonical_qpos_after_substep"][:39]
    elif bad == "washout":
        a["requested_offset_nm"][30, 0] = 1
    elif bad == "force":
        a["completed_contact_peak_force_n"][0, 0] = -1
    else:
        a["next_2_joint_target_rad"][0] = np.nan
    with pytest.raises(ValueError):
        replay.verify(d, first_target_rad=np.zeros(29), recorded=a, donor_hash=donor)
    assert replay.attempted_steps == 0


def test_causal_six_channel_contact_magnitude_mismatch_is_not_ignored(monkeypatch):
    d, replay, a, donor = fixture(monkeypatch)
    a["completed_contact_peak_force_n"][39, 0] = 1
    with pytest.raises(ValueError, match="differs"):
        replay.verify(d, first_target_rad=np.zeros(29), recorded=a, donor_hash=donor)
    assert replay.attempted_steps == replay.integrated_steps == 40
    assert replay.completed_windows == 0 and replay.failed
