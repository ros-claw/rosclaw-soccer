"""Synthetic codec/NN-reader checks, with no physical integration."""

import copy

import numpy as np
import pytest
from test_continuous_velocity_stream import stream
from test_shared_native_motor_world import source_xml

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.observed_velocity_continuation import ObservedVelocityContinuation
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from scripts.rsi_review_neural_exit_pilot import verify_velocity_history


def fixture(monkeypatch):
    import mujoco

    monkeypatch.setattr(mujoco, "mj_step", lambda *args: pytest.fail("reader must not integrate"))
    xml = source_xml().replace('name="ball"', 'name="soccer_ball"')
    model = mujoco.MjModel.from_xml_string(shared_native_scene(xml, prefixes=("a/",)))
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    owned = stream(monkeypatch)
    continuation = ObservedVelocityContinuation(owned)
    data = mujoco.MjData(model)
    data.qpos[layout.rq : layout.rq + 7] = [0, 0, 0.793, 1, 0, 0, 0]
    data.qpos[layout.qi] = np.asarray(owned.params.defaults)
    data.qpos[layout.bq : layout.bq + 7] = [2.3, 0, 0.11, 1, 0, 0, 0]
    data.qvel[layout.bv] = -0.4
    data.qvel[layout.bv + 4] = -0.4 / 0.11
    q, v = layout.canonical(data)
    targets, receipts, observations = [], [], []
    for frame in range(4):
        target = continuation.propose(q, v, v[3:6], np.zeros(3, np.float32), frame=frame)
        observations.append(owned.last_observation.copy())
        if frame == 1:
            target = np.full(29, 0.1)
        targets.append(target.copy())
        receipts.append(continuation.observe_executed(target, completed_control=frame + 1))
    forces = np.zeros((40, 1, 3))
    forces[10, 0, 0] = 2
    arrays = dict(
        qpos=np.tile(data.qpos, (40, 1)),
        qvel=np.tile(data.qvel, (40, 1)),
        joint_targets=np.stack(targets),
        commands=np.zeros((4, 3), np.float32),
        ball_forces=forces,
        velocity_control_active=np.array([True, False, True, True]),
        velocity_shadow_observations=np.stack(observations),
    )
    row = dict(
        controls=4,
        entry_control=1,
        exit_control=2,
        course=[2.3, 0, -0.4],
        experimental_fast_approach=False,
        observed_velocity_continuation=receipts,
    )
    kwargs = dict(layout=layout, model=model, params=owned.params, session=owned.session)
    return row, arrays, kwargs


def test_readback_reconstructs_actual_history_and_cpu_nn_without_step(monkeypatch):
    row, arrays, kwargs = fixture(monkeypatch)
    assert verify_velocity_history(row, arrays, **kwargs) == 4


@pytest.mark.parametrize(
    "defect", ["early_exit", "history", "phase", "target", "observation", "role"]
)
def test_tampered_history_or_future_trigger_cannot_pass(monkeypatch, defect):
    row, arrays, kwargs = fixture(monkeypatch)
    row = copy.deepcopy(row)
    if defect == "early_exit":
        row["exit_control"] = 1
    elif defect == "history":
        row["observed_velocity_continuation"][1]["actual_target_history"][0] += 0.1
    elif defect == "phase":
        row["observed_velocity_continuation"][2]["phase"] = 0
    elif defect == "target":
        arrays["joint_targets"][2, 0] += 0.01
    elif defect == "observation":
        arrays["velocity_shadow_observations"][2, 69] += 0.1
    else:
        arrays["velocity_control_active"][1] = True
    with pytest.raises(ValueError):
        verify_velocity_history(row, arrays, **kwargs)
