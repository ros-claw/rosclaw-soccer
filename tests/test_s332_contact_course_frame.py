from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from rosclaw_soccer.training.contact_course_frame import bind_contact_course_frame


def case():
    q = np.zeros(43)
    q[:3] = (3.4, 0, 0.75)
    q[3] = q[39] = 1
    q[36:39] = (4.0, -0.04, 0.115)
    return dict(
        source_qpos=q,
        training_qpos=q.copy(),
        source_goal_center_m=(7.5, 0.0, 1.0),
        training_goal_center_m=(7.5, 0.0, 1.0),
        source_reward_target_m=(7.5, 0.8, 1.5),
        training_reward_target_m=(7.5, 0.8, 1.5),
    )


def test_identity_binding_replays_and_does_not_mutate_inputs():
    c = case()
    original = c["source_qpos"].copy()
    result = bind_contact_course_frame(**c)
    assert result == bind_contact_course_frame(**c)
    assert result.translation_xy_m == (0.0, 0.0)
    assert result.activation_ceiling == "SIM_ONLY"
    np.testing.assert_array_equal(c["source_qpos"], original)
    np.testing.assert_array_equal(c["training_qpos"], original)
    with pytest.raises(FrozenInstanceError):
        result.binding_hash = "changed"


def test_translating_every_physical_and_reward_component_is_bound():
    c = case()
    before = bind_contact_course_frame(**c)
    delta = np.array([0.25, -1.4, 0.0])
    c["training_qpos"][:3] += delta
    c["training_qpos"][36:39] += delta
    for k in ("training_goal_center_m", "training_reward_target_m"):
        c[k] = tuple(float(x) for x in np.asarray(c[k]) + delta)
    after = bind_contact_course_frame(**c)
    assert after.translation_xy_m == (0.25, -1.4)
    assert before.binding_hash != after.binding_hash


@pytest.mark.parametrize("unmoved", ["ball", "goal", "reward", "goal_and_reward"])
def test_partial_translation_is_rejected(unmoved):
    c = case()
    delta = np.array([0.0, -1.4, 0.0])
    c["training_qpos"][:3] += delta
    if unmoved != "ball":
        c["training_qpos"][36:39] += delta
    for name in ("goal", "reward"):
        if name in unmoved:
            continue
        k = f"training_{name}_center_m" if name == "goal" else "training_reward_target_m"
        c[k] = tuple(float(x) for x in np.asarray(c[k]) + delta)
    with pytest.raises(ValueError, match="frames differ"):
        bind_contact_course_frame(**c)


@pytest.mark.parametrize("index", [2, 7, 38])
def test_no_hidden_pose_or_joint_changes(index):
    c = case()
    c["training_qpos"][index] += 0.01
    with pytest.raises(ValueError, match="preserve"):
        bind_contact_course_frame(**c)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, complex(1, 1)])
def test_invalid_goal_coordinates_rejected(bad):
    c = case()
    c["training_goal_center_m"] = (7.5, bad, 1.0)
    with pytest.raises(ValueError):
        bind_contact_course_frame(**c)


def test_nonunit_body_and_ball_quaternions_rejected():
    for index in (3, 39):
        c = case()
        c["training_qpos"][index] = 0.5
        with pytest.raises(ValueError, match="unit"):
            bind_contact_course_frame(**c)


@pytest.mark.parametrize(
    "state",
    [np.zeros(42), np.ones(43, dtype=bool), np.ones(43, dtype=complex), np.full(43, np.nan)],
)
def test_invalid_state_arrays_rejected(state):
    c = case()
    c["training_qpos"] = state
    with pytest.raises(ValueError, match="initial poses"):
        bind_contact_course_frame(**c)


@pytest.mark.parametrize("point", [[7.5, 0, 1], (7.5, 0), (7.5, 0, 1, 2)])
def test_coordinate_vectors_require_immutable_three_axes(point):
    c = case()
    c["source_goal_center_m"] = point
    with pytest.raises(ValueError, match="immutable"):
        bind_contact_course_frame(**c)
