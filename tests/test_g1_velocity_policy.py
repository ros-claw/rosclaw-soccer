"""Synthetic codecs/session mocks; not a native locomotion qualification."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from rosclaw_soccer.providers.g1.velocity_policy import (
    OBSERVATIONS,
    G1VelocityPolicy,
    VelocityPolicyParameters,
    file_hash,
    observation98,
    phase_delta,
    target29,
)


def parameters():
    return VelocityPolicyParameters(
        "sha256:" + "a" * 64, (50.0,) * 29, (2.0,) * 29, (0.0,) * 29, (0.25,) * 29, (0.0,) * 29
    )


def config():
    return dict(
        joint_ids_map=list(range(29)),
        step_dt=0.02,
        stiffness=[50.0] * 29,
        damping=[2.0] * 29,
        default_joint_pos=[0.0] * 29,
        commands={
            "base_velocity": {
                "ranges": {
                    "lin_vel_x": [-0.5, 1.0],
                    "lin_vel_y": [-0.5, 0.5],
                    "ang_vel_z": [-1.0, 1.0],
                    "heading": None,
                }
            }
        },
        actions={
            "JointPositionAction": dict(
                clip=None, joint_names=[".*"], scale=[0.25] * 29, offset=[0.0] * 29, joint_ids=None
            )
        },
        observations={
            name: dict(
                params={"period": 0.6}
                if name == "gait_phase"
                else {"command_name": "base_velocity"}
                if name == "velocity_commands"
                else {},
                clip=None,
                scale=[1.0] * width,
                history_length=1,
            )
            for name, width in OBSERVATIONS
        },
    )


def test_full_ordered_parameter_contract(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(config(), sort_keys=False))
    p = VelocityPolicyParameters.load(path, expected_hash=file_hash(path))
    assert p.defaults == parameters().defaults and len(p.contract_hash) == 71


@pytest.mark.parametrize(
    "change", ["order", "mapping", "history", "clip", "phase", "offset", "bool", "hash"]
)
def test_unqualified_parameters_fail_before_session_allocation(tmp_path, change):
    c = config()
    if change == "order":
        c["observations"] = dict(reversed(list(c["observations"].items())))
    elif change == "mapping":
        c["joint_ids_map"][0] = 1
    elif change == "history":
        c["observations"]["last_action"]["history_length"] = 2
    elif change == "clip":
        c["actions"]["JointPositionAction"]["clip"] = [[-1.0, 1.0]] * 29
    elif change == "phase":
        c["observations"]["gait_phase"]["params"]["period"] = 1.0
    elif change == "offset":
        c["actions"]["JointPositionAction"]["offset"][0] = 1.0
    elif change == "bool":
        c["stiffness"][0] = True
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(c, sort_keys=False))
    with pytest.raises(ValueError):
        VelocityPolicyParameters.load(
            path, expected_hash="sha256:" + "0" * 64 if change == "hash" else file_hash(path)
        )


def state():
    q, v = np.zeros(43), np.zeros(41)
    q[2] = 0.793
    q[3] = q[39] = 1.0
    return q, v


def test_observation_slices_reset_phase_advance_and_idle_mask():
    q, v = state()
    q[7:36] = np.arange(29) * 0.01
    v[6:35] = np.arange(29) * 0.02
    last = np.full(29, 0.3, dtype=np.float32)
    obs, phase = observation98(
        q,
        v,
        np.array([1.0, 2.0, 3.0]),
        np.array([0.2, 0.0, 0.0], dtype=np.float32),
        last,
        phase_delta(),
        parameters(),
    )
    np.testing.assert_array_equal(obs[:3], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(obs[3:6], [0.0, 0.0, -1.0])
    np.testing.assert_array_equal(obs[6:9], np.array([0.2, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(obs[11:40], q[7:36].astype(np.float32))
    np.testing.assert_array_equal(obs[40:69], v[6:35].astype(np.float32))
    np.testing.assert_array_equal(obs[69:], last)
    assert phase == np.float32(2 * phase_delta()) and obs.dtype == np.float32
    idle, nextphase = observation98(
        q, v, np.zeros(3), np.zeros(3, dtype=np.float32), last, phase, parameters()
    )
    assert nextphase > phase and not idle[9:11].any()


def test_target_float32_action_codec_is_not_sonic_action_order_or_double_precision_math():
    p = replace(parameters(), action_scale=(float(np.float32(0.55)),) * 29)
    action = np.arange(29, dtype=np.float32) * 0.1234
    np.testing.assert_array_equal(
        target29(action, p), (action * np.float32(0.55)).astype(np.float64)
    )
    with pytest.raises(ValueError):
        target29(action.astype(np.float64), p)


def test_actor_latches_wrong_frame_without_retry_or_step():
    actor = object.__new__(G1VelocityPolicy)
    actor.params = parameters()
    actor.phase = phase_delta()
    actor.last_action = np.zeros(29, dtype=np.float32)
    actor.next_frame = 0
    actor.faulted = False
    actor.attempted_neural_calls = actor.neural_calls = 0
    actor.session = SimpleNamespace(run=lambda *args: [np.zeros((1, 29), dtype=np.float32)])
    q, v = state()
    t = actor.propose(q, v, np.zeros(3), np.array([0.2, 0.0, 0.0], dtype=np.float32), frame=0)
    assert t.dtype == np.float64 and actor.neural_calls == 1
    with pytest.raises(ValueError):
        actor.propose(q, v, np.zeros(3), np.zeros(3, dtype=np.float32), frame=0)
    with pytest.raises(ValueError, match="latched"):
        actor.propose(q, v, np.zeros(3), np.zeros(3, dtype=np.float32), frame=1)
    assert actor.neural_calls == actor.attempted_neural_calls == 1


@pytest.mark.parametrize("kind", ["command", "quaternion", "state", "last_action", "phase"])
def test_bad_current_observations_do_not_make_features(kind):
    q, v = state()
    cmd = np.zeros(3, dtype=np.float32)
    last = np.zeros(29, dtype=np.float32)
    phase = phase_delta()
    if kind == "command":
        cmd[1] = 0.7
    elif kind == "quaternion":
        q[3] = 0.1
    elif kind == "state":
        v[1] = np.nan
    elif kind == "last_action":
        last = last.astype(np.float64)
    else:
        phase = float(phase)
    with pytest.raises(ValueError):
        observation98(q, v, np.zeros(3), cmd, last, phase, parameters())
