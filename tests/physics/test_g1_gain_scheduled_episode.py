"""Synthetic native integration and protocol checks, not actual G1 skill proof."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.g1_gain_scheduled_episode import G1GainSchedule, G1GainScheduledEpisode
from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
from rosclaw_soccer.sim.contracts import hash_bytes
from tests.physics.test_g1_feedback_forecast import world


def schedule(mode="linear10"):
    return G1GainSchedule((10.0,) * 29, (0.2,) * 29, (20.0,) * 29, (0.4,) * 29, 30, mode)


def pair(mode="constant"):
    model, _, _ = world()
    model.dof_armature[6:35] = 0.1
    binary = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, binary)
    kwargs = dict(
        world_hash=hash_bytes(binary.tobytes()),
        kp=np.full(29, 10.0),
        kd=np.full(29, 0.2),
        course=(2.0, 1.0, 0.2),
        defaults=np.zeros(29),
        record_all_ball_contacts=True,
    )
    return G1TargetEpisode(model, **kwargs), G1GainScheduledEpisode(
        model, schedule=schedule(mode), **kwargs
    )


def test_declaration_formula_exact_endpoints_and_owned_values():
    s = schedule()
    np.testing.assert_array_equal(s.gains(29)[0], np.full(29, 10.0))
    np.testing.assert_array_equal(s.gains(30)[0], np.full(29, 11.0))
    np.testing.assert_array_equal(s.gains(38)[0], np.full(29, 19.0))
    np.testing.assert_array_equal(s.gains(39)[0], np.full(29, 20.0))
    np.testing.assert_array_equal(s.gains(299)[1], np.full(29, 0.4))
    value = s.gains(30)[0]
    value[:] = 999
    assert s.gains(30)[0][0] == 11.0
    assert s.contract_hash != schedule("constant").contract_hash


@pytest.mark.parametrize(
    "change",
    [
        dict(mode="adaptive"),
        dict(handoff_frame=True),
        dict(handoff_frame=29),
        dict(initial_kp=(0.0,) * 29),
        dict(final_kd=(float("nan"),) * 29),
        dict(final_kp=(1001.0,) * 29),
        dict(initial_kd=[0.2] * 29),
    ],
)
def test_bad_gain_declaration_rejected(change):
    with pytest.raises(ValueError):
        replace(schedule(), **change)


def test_constant_schedule_all300_native_fields_equal_original_body():
    plain, new = pair()
    data_id = id(new.data)
    for f in range(300):
        if f == 30:
            new.activate(f)
        target = np.full(29, 0.02 * np.sin(f * 0.1))
        a, b = plain.step(f, target), new.step(f, target)
        assert set(b) == set(a) | {"held_pd_kp", "held_pd_kd"}
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(b["held_pd_kp"], np.full((1, 29), 10.0))
        assert id(new.data) == data_id
    assert new.review() == plain.review()


def test_actual_handoff_required_and_cannot_be_repeated_or_retimed():
    _, new = pair("linear10")
    with pytest.raises(ValueError):
        new.activate(30)
    for f in range(30):
        new.step(f, np.zeros(29))
    q, v = new.canonical()
    time = float(new.data.time)
    with pytest.raises(ValueError):
        new.step(30, np.zeros(29))
    assert new.integrated_steps == 300
    new.activate(30)
    np.testing.assert_array_equal(new.canonical()[0], q)
    np.testing.assert_array_equal(new.canonical()[1], v)
    assert new.data.time == time
    with pytest.raises(ValueError):
        new.activate(30)
    row = new.step(30, np.full(29, 0.01))
    np.testing.assert_array_equal(row["held_pd_kp"], np.full((1, 29), 11.0))
    expected = (row["joint_target_rad"][0] - row["canonical_qpos"][0, 7:36]) * 11 - row[
        "canonical_qvel"
    ][0, 6:35] * 0.22
    np.testing.assert_array_equal(row["pd_requested_torque_nm"][0, 0], expected)
    with pytest.raises(ValueError):
        new.step(30, np.zeros(29))


def test_partial_native_fault_remains_terminal(monkeypatch):
    _, new = pair()

    def fault(*args):
        raise RuntimeError("toy-native-fault")

    monkeypatch.setattr(mujoco, "mj_step", fault)
    with pytest.raises(RuntimeError):
        new.step(0, np.zeros(29))
    assert new.failed and new.attempted_steps == 1 and new.integrated_steps == 0
    with pytest.raises(ValueError):
        new.step(0, np.zeros(29))
