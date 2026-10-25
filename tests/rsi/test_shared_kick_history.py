import numpy as np
import pytest
from test_kick_measured_history import FakeNativePolicy
from test_shared_robonaldo_player import Policy, setup

from rosclaw_soccer.rsi.shared_kick_history import measured_kick_sample
from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer

ISAAC_TO_MUJOCO = np.arange(29)[::-1]
NPZ_ANCHOR_IDX = 0


class MeasuredPolicy(Policy):
    def __init__(self, sample):
        super().__init__()
        self.__dict__.update(FakeNativePolicy(sample).__dict__)
        self.motion_total_steps = 10
        self.motion_body_pos = np.zeros((10, 1, 3))
        self.motion_body_pos[3, 0] = (0.2, 0.1, 0.8)
        self._init_to_world = np.eye(3)


def entry():
    w = setup()
    samples = []
    for frame in range(5):
        w.advance({"a/": np.full(29, frame * 0.001), "b/": np.zeros(29)})
        samples.append(measured_kick_sample(w, "a/"))
    p = MeasuredPolicy(samples[-1])
    a = SharedRoboNaldoPlayer(w, "a/", p, goal=np.array([5.0, 0.0, 0.11]))
    return w, a, tuple(samples)


def test_capture_is_actual_owned_boundary_and_read_only():
    w = setup()
    samples = []
    for frame in range(5):
        targets = {"a/": np.full(29, frame * 0.001), "b/": np.full(29, -0.02)}
        w.advance(targets)
        before = (w.data.qpos.copy(), w.data.qvel.copy(), w.data.ctrl.copy(), w.data.time)
        sample = measured_kick_sample(w, "a/")
        samples.append(sample)
        assert sample.frame == frame + 1 and sample.previous_target_frame == frame
        np.testing.assert_array_equal(sample.previous_pd_target, targets["a/"])
        np.testing.assert_array_equal(sample.joint_position, w.data.qpos[w.layouts["a/"].qi])
        for actual, expected in zip(
            (w.data.qpos, w.data.qvel, w.data.ctrl, w.data.time), before, strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
    assert [s.frame for s in samples] == list(range(1, 6))
    assert len(set(s.joint_position for s in samples)) == 5


def test_no_history_before_control_or_for_foreign_player():
    w = setup()
    with pytest.raises(ValueError):
        measured_kick_sample(w, "a/")
    w.advance({"a/": np.zeros(29), "b/": np.zeros(29)})
    with pytest.raises(ValueError):
        measured_kick_sample(w, "foreign/")


def test_measured_entry_preserves_physics_and_uses_real_previous_targets():
    w, a, samples = entry()
    before = w.data.qpos.copy(), w.data.qvel.copy(), w.data.ctrl.copy()
    receipt = a.install_measured_entry(samples, reference_frame=3)
    assert receipt.startswith("sha256:")
    assert a.policy.time_step == a.policy.WARMUP_STEPS + 3
    assert len(a.policy._action_buf) == 4
    np.testing.assert_array_equal(a.policy.last_action_il, np.full(29, 0.008, np.float32))
    np.testing.assert_array_equal(a.policy._ref_anchor_world_origin, (0.2, 0.1, 0.8))
    for expected, actual in zip(before, (w.data.qpos, w.data.qvel, w.data.ctrl), strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert w.control_steps == 5
    with pytest.raises(ValueError, match="one pre-proposal"):
        a.install_measured_entry(samples, reference_frame=3)
    target, _ = a.propose()
    w.advance({"a/": target, "b/": np.zeros(29)})
    a.commit()
    assert a.commits == 1


def test_invalid_entry_poisoned_without_world_step():
    w, a, samples = entry()
    with pytest.raises(ValueError, match="five contiguous"):
        a.install_measured_entry(samples[1:], reference_frame=0)
    assert a.failed and w.control_steps == 5
    with pytest.raises(ValueError):
        a.propose()


def test_entry_after_first_proposal_not_allowed():
    w, a, samples = entry()
    a.propose()
    with pytest.raises(ValueError, match="one pre-proposal"):
        a.install_measured_entry(samples, reference_frame=0)
    assert w.control_steps == 5
