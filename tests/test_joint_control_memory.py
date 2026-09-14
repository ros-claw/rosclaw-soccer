import numpy as np
import pytest

from rosclaw_soccer.sim.joint_control_memory import (
    JointControlMemory,
    JointMemoryTracker,
    JointMemoryTrackingConfig,
)


def memory(**overrides):
    args = dict(
        joint_position=np.ones((6, 2)),
        joint_velocity=np.zeros((6, 2)),
        pd_proposal=np.full((6, 2), 3.0),
        kp=np.full((6, 2), 10.0),
        kd=np.ones((6, 2)),
        period_sec=0.1,
        joint_names=("a", "b"),
    )
    return JointControlMemory(**(args | overrides))


def config(**overrides):
    return JointMemoryTrackingConfig(
        **(
            dict(
                feedback_scale=1.0,
                correction_cap_nm=2.0,
                release_delay_sec=0.1,
                release_fade_sec=0.2,
                initial_position_tolerance_rad=0.01,
                initial_velocity_tolerance_rad_s=0.01,
            )
            | overrides
        )
    )


def sample(tracker, t, **overrides):
    return tracker.sample(
        **(
            dict(
                timestamp_sec=t,
                joint_position=np.ones(2),
                joint_velocity=np.zeros(2),
                live_pd_proposal=np.full(2, 9.0),
                observed_contact=False,
            )
            | overrides
        )
    )


def test_exact_nominal_and_causal_release():
    tracker = JointMemoryTracker(memory(), config())
    np.testing.assert_array_equal(sample(tracker, 0).pd_proposal, [3, 3])
    assert sample(tracker, 0.1, observed_contact=True).memory_weight == 1
    assert sample(tracker, 0.2).memory_weight == 1
    out = sample(tracker, 0.3)
    assert out.memory_weight == pytest.approx(0.5)
    np.testing.assert_allclose(out.pd_proposal, [6, 6])
    np.testing.assert_array_equal(sample(tracker, 0.4).pd_proposal, [9, 9])
    assert sample(tracker, 0.5, observed_contact=True).observed_contact_timestamp_sec == 0.1


def test_correction_bound_is_distinct_from_total_proposal():
    tracker = JointMemoryTracker(memory(), config())
    sample(tracker, 0)
    out = sample(tracker, 0.1, joint_position=np.array([0.0, 2.0]))
    np.testing.assert_array_equal(out.correction_nm, [2, -2])
    np.testing.assert_array_equal(out.pd_proposal, [5, 1])
    assert not out.pd_proposal.flags.writeable and not out.correction_nm.flags.writeable


def test_joint_order_and_data_bind_hash():
    original = np.ones((6, 2))
    a = memory(joint_position=original)
    original[:] = 8
    np.testing.assert_array_equal(a.joint_position, np.ones((6, 2)))
    assert not a.joint_position.flags.writeable
    assert a.content_hash == memory().content_hash
    assert a.content_hash != memory(joint_names=("b", "a")).content_hash
    assert a.content_hash != memory(period_sec=0.2).content_hash
    assert a.content_hash != memory(pd_proposal=np.zeros((6, 2))).content_hash


@pytest.mark.parametrize(
    "override",
    [
        {"period_sec": 0},
        {"period_sec": True},
        {"period_sec": float("nan")},
        {"joint_names": ("a", "a")},
        {"joint_names": ("a", "")},
        {"joint_position": np.ones((0, 2))},
        {"kp": np.ones((6, 3))},
        {"kd": np.full((6, 2), -1.0)},
        {"pd_proposal": np.full((6, 2), float("inf"))},
        {"activation_ceiling": "REAL"},
    ],
)
def test_reject_invalid_memory(override):
    with pytest.raises(ValueError):
        memory(**override)


@pytest.mark.parametrize(
    "override",
    [
        {"feedback_scale": -1},
        {"feedback_scale": True},
        {"correction_cap_nm": 0},
        {"release_fade_sec": 0},
        {"initial_position_tolerance_rad": float("nan")},
    ],
)
def test_reject_invalid_config(override):
    with pytest.raises(ValueError):
        config(**override)


@pytest.mark.parametrize(
    "override",
    [
        {"timestamp_sec": 0.1},
        {"timestamp_sec": float("nan")},
        {"observed_contact": 1},
        {"joint_position": np.ones(3)},
        {"joint_position": np.array([1.0, float("nan")])},
        {"joint_position": np.zeros(2)},
        {"joint_velocity": np.ones(2)},
        {"live_pd_proposal": np.array([True, False])},
    ],
)
def test_invalid_sample_fault_latches(override):
    tracker = JointMemoryTracker(memory(), config())
    with pytest.raises(ValueError):
        sample(tracker, 0, **override)
    with pytest.raises(ValueError, match="fault-latched"):
        sample(tracker, 0)


def test_no_clock_reuse_or_auto_rearm():
    tracker = JointMemoryTracker(memory(), config())
    sample(tracker, 0)
    with pytest.raises(ValueError):
        sample(tracker, 0)
    with pytest.raises(ValueError, match="fault-latched"):
        sample(tracker, 0.1)


def test_roles_are_independent():
    a, b = (JointMemoryTracker(memory(), config()) for _ in range(2))
    sample(a, 0, observed_contact=True)
    sample(b, 0)
    for t in [0.1, 0.2, 0.3, 0.4]:
        x = sample(a, t)
        y = sample(b, t)
    assert x.memory_weight == 0 and y.memory_weight == 1


def test_arithmetic_overflow_faults():
    tracker = JointMemoryTracker(memory(kp=np.full((6, 2), 1e308)), config())
    sample(tracker, 0)
    with pytest.raises(ValueError, match="arithmetic"):
        sample(tracker, 0.1, joint_position=np.full(2, -1e308))
    with pytest.raises(ValueError, match="fault-latched"):
        sample(tracker, 0.1)


def test_end_of_memory_rejected():
    tracker = JointMemoryTracker(memory(), config())
    for i in range(6):
        sample(tracker, i * 0.1)
    with pytest.raises(ValueError):
        sample(tracker, 0.6)
