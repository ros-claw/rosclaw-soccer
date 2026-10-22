import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_handoff_opportunity import body_contact_kind, handoff_windows


def fixture():
    q, v = np.zeros((30, 10, 79)), np.zeros((30, 10, 76))
    q[:, :, 0:3] = [0.25, 0.1, 0.11]
    q[:, :, 3] = 1
    q[:, :, 45] = 0.8
    q[:, :, 46] = 1
    return dict(
        qpos=q,
        qvel=v,
        body_failed=np.zeros((30, 2), bool),
        root_qposadr=43,
        root_dofadr=41,
        ball_qposadr=0,
        ball_dofadr=0,
        contacts={"b/right_knee_link": dict(first_substep=4, peak_force_n=120.0)},
        prefix="b/",
    )


def test_leg_control_yields_only_candidate_boundary_not_a_clean_foot_grade():
    data = fixture()
    windows, excluded = handoff_windows(**data)
    assert not excluded and len(windows) == 1
    assert windows[0].earliest_next_control == 15
    assert windows[0].last_control == 29
    assert body_contact_kind("b/right_knee_link", prefix="b/") == "leg"
    assert body_contact_kind("b/right_ankle_roll_link", prefix="b/") == "foot"


def test_hand_unknown_body_or_failed_actor_does_not_produce_ready_boundary():
    data = fixture()
    data["contacts"]["b/left_wrist_roll_link"] = dict(first_substep=3, peak_force_n=2.0)
    windows, excluded = handoff_windows(**data)
    assert not windows and excluded == ("b/left_wrist_roll_link",)
    data = fixture()
    data["body_failed"][-1, 0] = True
    assert not handoff_windows(**data)[0]
    assert body_contact_kind("b/unknown_glove", prefix="b/") == "excluded_or_unknown"


def test_fast_or_behind_body_ball_and_missing_contact_do_not_fake_control():
    data = fixture()
    data["qvel"][:, :, 0] = 2.0
    assert not handoff_windows(**data)[0]
    data = fixture()
    data["qpos"][:, :, 0] = -0.25
    assert not handoff_windows(**data)[0]
    data = fixture()
    data["contacts"] = {}
    assert not handoff_windows(**data)[0]
    data = fixture()
    data["qpos"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        handoff_windows(**data)
