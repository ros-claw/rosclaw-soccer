from collections import deque
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.growth.locomotion_contact_teacher import G1RollingOptionBridgeConfig
from rosclaw_soccer.providers.g1.kick_measured_history import (
    KickMeasuredSample,
    install_measured_kick_history,
)
from rosclaw_soccer.sim.contracts import hash_json

ISAAC_TO_MUJOCO = np.arange(29)[::-1]
BUFFERS = (
    "_ang_vel_buf",
    "_jpos_buf",
    "_jvel_buf",
    "_action_buf",
    "_ball_pos_buf",
    "_target_pos_buf",
)


def sample(frame):
    return KickMeasuredSample(
        frame,
        frame * 0.02,
        frame - 1,
        tuple([frame * 0.01] * 29),
        tuple([frame * 0.1] * 29),
        (0.0, 0.0, frame * 0.2),
        (0.0, 0.0, 0.0),
        (2**-0.5, 0.0, 0.0, 2**-0.5),
        (0.0, 1.0, 0.0),
        tuple([frame * 0.02] * 29),
    )


class FakeNativePolicy:
    def __init__(self, current):
        self.runtime_mode = "sim"
        self.use_body_frame_ball = False
        self.default_q_mj = np.zeros(29, dtype=np.float32)
        self.default_q_il = self.default_q_mj[ISAAC_TO_MUJOCO]
        self.action_scale_mj = np.full(29, 0.5, dtype=np.float32)
        self.action_clip_lo_il = np.full(29, -10.0)
        self.action_clip_hi_il = np.full(29, 10.0)
        self.target_pos_w = np.array([0.0, 2.0, 0.0], dtype=np.float32)
        self.state_cmd = SimpleNamespace(
            q=np.array(current.joint_position),
            dq=np.array(current.joint_velocity),
            root_ang_vel_b=np.array(current.angular_velocity),
            pelvis_pos_w=np.array(current.pelvis_position),
            pelvis_quat_w=np.array(current.pelvis_quaternion),
            ball_pos_w=np.array(current.ball_position),
            target_y_bias=0.0,
        )
        self.last_action_il = np.zeros(29, dtype=np.float32)
        for name in BUFFERS:
            setattr(self, name, deque([np.array([-9.0]) for _ in range(5)], maxlen=5))


def test_measured_frames_and_applied_targets_replace_repeated_current_state():
    samples = tuple(sample(f) for f in range(2, 7))
    policy = FakeNativePolicy(samples[-1])
    before = {k: v.copy() for k, v in vars(policy.state_cmd).items() if isinstance(v, np.ndarray)}
    receipt = install_measured_kick_history(policy, samples, frame=6, time_sec=0.12)
    assert receipt.startswith("sha256:") and len(receipt) == 71
    assert all(len(getattr(policy, n)) == 4 for n in BUFFERS)
    np.testing.assert_allclose(np.asarray(policy._jpos_buf)[:, 0], [0.02, 0.03, 0.04, 0.05])
    np.testing.assert_allclose(np.asarray(policy._action_buf)[:, 0], [0.08, 0.12, 0.16, 0.20])
    np.testing.assert_allclose(policy.last_action_il, 0.24)
    assert not np.allclose(policy.last_action_il, policy.state_cmd.q / 0.5)
    np.testing.assert_allclose(policy._ball_pos_buf[-1], [1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(policy._target_pos_buf[-1], [2, 0, 0], atol=1e-6)
    for key, value in before.items():
        np.testing.assert_array_equal(value, getattr(policy.state_cmd, key))
    repeated = FakeNativePolicy(samples[-1])
    assert install_measured_kick_history(repeated, samples, frame=6, time_sec=0.12) == receipt


@pytest.mark.parametrize(
    "fault", ["short", "time_gap", "stale", "changed_current", "zero_scale", "real"]
)
def test_invalid_history_does_not_partially_mutate_buffers(fault):
    samples = tuple(sample(f) for f in range(2, 7))
    policy = FakeNativePolicy(samples[-1])
    if fault == "short":
        samples = samples[1:]
    if fault == "time_gap":
        samples = (replace(samples[0], time_sec=0.001), *samples[1:])
    if fault == "stale":
        samples = tuple(sample(f) for f in range(1, 6))
    if fault == "changed_current":
        policy.state_cmd.q[0] += 0.1
    if fault == "zero_scale":
        policy.action_scale_mj[0] = 0.0
    if fault == "real":
        policy.runtime_mode = "real"
    with pytest.raises(ValueError):
        install_measured_kick_history(policy, samples, frame=6, time_sec=0.12)
    assert all(len(getattr(policy, n)) == 5 for n in BUFFERS)
    assert all(np.array_equal(v, [-9.0]) for n in BUFFERS for v in getattr(policy, n))
    np.testing.assert_array_equal(policy.last_action_il, np.zeros(29))


@pytest.mark.parametrize(
    "changes",
    [
        dict(frame=True),
        dict(previous_target_frame=0),
        dict(time_sec=float("nan")),
        dict(joint_velocity=tuple([float("inf")] * 29)),
        dict(pelvis_quaternion=(1.0, 1.0, 0.0, 0.0)),
        dict(previous_pd_target=[0.0] * 29),
    ],
)
def test_invalid_or_unbound_measurements_rejected(changes):
    with pytest.raises(ValueError):
        replace(sample(5), **changes)


def test_entry_goal_changes_receipt_without_altering_measured_history():
    samples = tuple(sample(f) for f in range(2, 7))
    policy = FakeNativePolicy(samples[-1])
    first = install_measured_kick_history(policy, samples, frame=6, time_sec=0.12)
    policy.target_pos_w[1] = 3.0
    second = install_measured_kick_history(policy, samples, frame=6, time_sec=0.12)
    assert first != second
    np.testing.assert_allclose(policy._target_pos_buf[-1], [3, 0, 0], atol=1e-6)


def test_entry_clock_and_projection_parameters_are_bound():
    samples = tuple(sample(f) for f in range(2, 7))
    policy = FakeNativePolicy(samples[-1])
    with pytest.raises(ValueError):
        install_measured_kick_history(policy, samples, frame=6, time_sec=0.14)
    first = install_measured_kick_history(policy, samples, frame=6, time_sec=0.12)
    policy.action_scale_mj[0] = 0.6
    second = install_measured_kick_history(policy, samples, frame=6, time_sec=0.12)
    assert first != second
    policy.action_clip_hi_il[0] = 1e100
    with pytest.raises(ValueError):
        install_measured_kick_history(policy, samples, frame=6, time_sec=0.12)


def test_numeric_trace_preserves_immutable_source_values():
    measured = sample(5)
    values = measured.numeric_record()
    assert values.shape == (103,) and values.dtype == np.float64
    np.testing.assert_array_equal(values[:3], [5, 0.1, 4])
    np.testing.assert_array_equal(values[3:32], measured.joint_position)
    np.testing.assert_array_equal(values[74:], measured.previous_pd_target)
    values[3] = 9
    assert measured.joint_position[0] == 0.05


def test_default_option_hash_stays_legacy_and_feature_requires_explicit_warmstart():
    default = G1RollingOptionBridgeConfig()
    legacy = asdict(default)
    for key in (
        "task_context_bound",
        "continuous_rearm_enabled",
        "per_player_options_enabled",
        "reference_rebase_only",
        "measured_state_history",
    ):
        legacy.pop(key)
    assert default.config_hash == hash_json(legacy)
    with pytest.raises(ValueError):
        replace(default, measured_state_history=True)
    with pytest.raises(ValueError):
        replace(default, measured_state_history=1)
    enabled = replace(default, observation_warmstart=True, measured_state_history=True)
    assert enabled.config_hash != replace(enabled, measured_state_history=False).config_hash
