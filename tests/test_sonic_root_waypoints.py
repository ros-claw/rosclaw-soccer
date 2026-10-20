"""Mocked input binding and default compatibility, not native motion proof."""

from dataclasses import replace

import numpy as np
import pytest
from test_s304_sonic_navigation import Backend, observation

from rosclaw_soccer.providers.g1 import sonic_root_waypoints as module
from rosclaw_soccer.providers.g1.sonic_navigation import SonicNavigationConfig, _StreamingBackend
from rosclaw_soccer.providers.g1.sonic_root_waypoint_inputs import RootWaypointIntent


def config():
    return SonicNavigationConfig(
        maximum_frames=300,
        model_variant="low_latency",
        inference_threads=1,
        record_default_planner_events=True,
    )


def test_default_feed_hook_is_exact_identity_not_a_new_default_goal():
    feed = {"fixture": np.array([1.0])}
    assert _StreamingBackend._transform_planner_feed(None, feed, 0) is feed


def motor(monkeypatch):
    class MockBackend(Backend):
        def __init__(self, *args):
            super().__init__(*args)
            self.active_goal = None
            self.waypoint_events = []
            self.received = []

        def navigation_tick(self, state, frame):
            assert self.active_observation_hash is not None
            self.received.append(self.active_goal)
            return super().navigation_tick(state, frame)

    monkeypatch.setattr(module, "_RootWaypointBackend", MockBackend)
    return module.G1SonicRootWaypointNavigation(
        None,
        "red.defender",
        config(),
        fixed_world_goal_xy_m=(0.03, -0.04),
        goal_basis_hash="sha256:" + "a" * 64,
        goal_deadline_frame=90,
        native_joint_limits=((-3.2, 3.2),) * 29,
    )


def test_goal_is_bound_to_each_actual_observation_and_cleared_after_tick(monkeypatch):
    m = motor(monkeypatch)
    for f in range(2):
        m.propose(observation(f))
        assert m.backend.active_goal is None
    assert (
        m.backend.received[0].measured_observation_hash
        != m.backend.received[1].measured_observation_hash
    )
    assert m.backend.received[0].world_target_xyz_m == (0.03, -0.04, 0.78)
    with pytest.raises(ValueError):
        m.propose(observation(1))
    assert m._faulted and m.backend.active_goal is None


@pytest.mark.parametrize(
    "change",
    [
        dict(maximum_frames=600),
        dict(model_variant="sonic_v1_1"),
        dict(experimental_command_replanning=True),
        dict(record_default_planner_events=False),
    ],
)
def test_unqualified_mixed_config_rejected_before_nn_allocation(monkeypatch, change):
    def allocate(*args):
        pytest.fail("NN allocation before profile validation")

    monkeypatch.setattr(module, "_RootWaypointBackend", allocate)
    with pytest.raises(ValueError):
        module.G1SonicRootWaypointNavigation(
            None,
            "red.defender",
            replace(config(), **change),
            fixed_world_goal_xy_m=(0.03, -0.04),
            goal_basis_hash="sha256:" + "a" * 64,
            goal_deadline_frame=90,
            native_joint_limits=((-3.2, 3.2),) * 29,
        )


def test_new_option_is_not_silently_snapshot_clone_qualified(monkeypatch):
    from rosclaw_soccer.physics.sonic_navigation_forecast import navigation_state_hash

    m = motor(monkeypatch)
    with pytest.raises(ValueError, match="exact qualified"):
        navigation_state_hash(m)


def test_fixed_goal_expires_at_original_deadline_without_history_reset(monkeypatch):
    m = motor(monkeypatch)
    for f in range(92):
        m.propose(observation(f))
    assert m.backend.received[90] is not None
    assert m.backend.received[91] is None
    assert m.backend.resets == 1 and m.backend.observations == 91
    assert m.backend.active_goal is None and m.backend.active_observation_hash is None


def test_root_backend_records_actual_flag_and_keeps_original_feed_owned():
    backend = object.__new__(module._RootWaypointBackend)
    backend.active_goal = RootWaypointIntent(
        (0.03, -0.04, 0.78), 0.0, "sha256:" + "a" * 64, "sha256:" + "b" * 64
    )
    backend.waypoint_events = []
    backend.active_observation_hash = "sha256:" + "c" * 64
    context = np.zeros((1, 4, 36), dtype=np.float32)
    context[:, :, 3] = 1.0
    context[:, :, 2] = 0.78
    f = dict(
        context_mujoco_qpos=context,
        has_specific_target=np.zeros((1, 1), dtype=np.int64),
        specific_target_positions=np.zeros((1, 4, 3), dtype=np.float32),
        specific_target_headings=np.zeros((1, 4), dtype=np.float32),
    )
    actual = backend._transform_planner_feed(f, 20)
    assert actual["has_specific_target"].item() == 1 and not f["has_specific_target"].any()
    assert backend.waypoint_events[0]["actual_planner_inputs"]["has_specific_target"] == [[1]]
    assert backend.waypoint_events[0]["frame"] == 20


def test_waypoint_generated_reference_never_silently_clips_native_joint_violation():
    backend = object.__new__(module._RootWaypointBackend)
    backend.native_joint_limits = ((-1.0, 1.0),) * 29
    backend.waypoint_events = [{}]
    segment = np.zeros((16, 36))
    segment[:, 3] = 1.0
    segment[5, 7] = 1.01
    old = segment.copy()
    with pytest.raises(ValueError, match="no clipping"):
        backend._validate_planner_segment(segment)
    np.testing.assert_array_equal(old, segment)
    assert backend.waypoint_events[0]["maximum_native_joint_limit_violation_rad"] == pytest.approx(
        0.01
    )


def test_expired_goal_reverts_to_unmodified_flag0_on_next_scheduled_plan():
    backend = object.__new__(module._RootWaypointBackend)
    backend.active_goal = None
    backend.active_observation_hash = "sha256:" + "c" * 64
    backend.waypoint_events = []
    feed = dict(has_specific_target=np.zeros((1, 1), dtype=np.int64))
    assert backend._transform_planner_feed(feed, 100) is feed
    assert backend.waypoint_events[0]["fixed_goal_expired"]
