import numpy as np
import pytest

from rosclaw_soccer.rsi.measured_receiving_intent import receiving_intent


def state():
    return dict(
        root_xy=np.array([4.0, 0.5]),
        yaw=float(np.pi),
        ball_xy=np.zeros(2),
        ball_velocity_xy=np.zeros(2),
        home_xy=np.array([4.0, 0.5]),
        pass_direction=np.array([1.0, 0.0]),
        launch_seen=False,
    )


def test_receiver_waits_for_actual_pass_not_frame_index():
    s = state()
    before = {k: v.copy() for k, v in s.items() if isinstance(v, np.ndarray)}
    intent = receiving_intent(**s)
    assert intent.role_state == "WAIT_PASS" and not intent.launch_seen
    assert intent.command_body[:2] == (0.0, 0.0)
    for k, value in before.items():
        np.testing.assert_array_equal(s[k], value)


def test_incoming_measured_velocity_changes_intercept_and_history_survives_stop():
    s = state()
    s["ball_velocity_xy"] = np.array([5.0, 0.2])
    s["ball_xy"] = np.array([2.0, 0.0])
    intent = receiving_intent(**s)
    assert intent.role_state == "INTERCEPT" and intent.launch_seen
    assert intent.target_xy[0] > s["ball_xy"][0]
    assert -0.5 <= intent.command_body[0] <= 1
    assert abs(intent.command_body[1]) <= 0.5 and abs(intent.command_body[2]) <= 1
    s["launch_seen"] = True
    s["ball_velocity_xy"][:] = 0
    assert receiving_intent(**s).role_state == "INTERCEPT"


def test_backward_ball_does_not_fabricate_pass_and_nonfinite_rejected():
    s = state()
    s["ball_velocity_xy"] = np.array([-5.0, 0.0])
    assert not receiving_intent(**s).launch_seen
    s["ball_xy"][0] = np.nan
    with pytest.raises(ValueError):
        receiving_intent(**s)


def test_communicated_anchor_activates_support_without_fabricating_ball_launch():
    s = state()
    anchor = np.array([4.0, 0.12])
    intent = receiving_intent(**s, ready_anchor_xy=anchor)
    assert intent.role_state == "MOVE_SUPPORT" and not intent.launch_seen
    assert intent.target_xy == (4.0, 0.12)
    assert abs(intent.command_body[1]) > 0.1
    s["root_xy"] = anchor.copy()
    ready = receiving_intent(**s, ready_anchor_xy=anchor)
    assert ready.role_state == "WAIT_PASS" and ready.command_body[:2] == (0.0, 0.0)
    with pytest.raises(ValueError):
        receiving_intent(**s, ready_anchor_xy=np.array([float("nan"), 0.0]))


def test_plane_role_keeps_communicated_longitudinal_destination_and_original_lateral_prediction():
    s = state()
    s.update(ball_xy=np.array([1.0, 0.1]), ball_velocity_xy=np.array([1.5, 0.2]))
    anchor = np.array([4.25, 0.12])
    old = receiving_intent(**s, ready_anchor_xy=anchor)
    new = receiving_intent(**s, ready_anchor_xy=anchor, mode="anchor-plane-v2")
    assert old.target_xy[0] < 3.0 and new.target_xy[0] == 4.25
    assert old.target_xy[1] == new.target_xy[1]
    assert new.role_state == "TRACK_RECEIVING_PLANE" and new.launch_seen
    assert new.command_body[2] == old.command_body[2]
    for mode in ("pursuit-v1", "anchor-plane-v2"):
        s["ball_velocity_xy"][:] = 0
        s["launch_seen"] = False
        before = receiving_intent(**s, ready_anchor_xy=anchor, mode=mode)
        assert before.target_xy == tuple(anchor) and not before.launch_seen


@pytest.mark.parametrize("mode", [None, True, "unknown", "anchor-plane-v2"])
def test_unknown_mode_or_missing_plane_anchor_rejected(mode):
    with pytest.raises(ValueError):
        receiving_intent(**state(), mode=mode)


def test_rotated_plane_projection_uses_pass_axis_not_world_x():
    s = state()
    s.update(
        pass_direction=np.array([0.0, 1.0]),
        ball_xy=np.array([0.0, 1.0]),
        ball_velocity_xy=np.array([0.1, 2.0]),
    )
    anchor = np.array([0.12, 4.25])
    new = receiving_intent(**s, ready_anchor_xy=anchor, mode="anchor-plane-v2")
    assert new.target_xy[1] == 4.25


@pytest.mark.parametrize("name", ["rsi_pass_transfer_trial", "rsi_pass_transfer_replay_trial"])
def test_receiver_role_study_cannot_stack_sampling_or_replay_implicitly(name):
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module("scripts." + name)
    setup = SimpleNamespace(prepare=lambda *a, **k: pytest.fail("preallocation rejection required"))
    kwargs = dict(
        output=None,
        index=0,
        counts={},
        receiver_intent_mode="anchor-plane-v2",
        sampling_seed=1,
        sampling_profile="contact-v1",
    )
    with pytest.raises(ValueError):
        if name.endswith("replay_trial"):
            module.replay_trial(setup, None, dict(assisted=False), {}, raw_root=None, **kwargs)
        else:
            module.collect_trial(setup, None, dict(assisted=False), **kwargs)
