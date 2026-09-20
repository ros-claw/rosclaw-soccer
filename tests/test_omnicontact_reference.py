import numpy as np
import pytest

from rosclaw_soccer.training.omnicontact_reference import (
    BODY_INDICES,
    LAB_TO_MUJOCO,
    omnicontact_pose_reference,
)


def inputs():
    quaternion = np.zeros((3, 39, 4))
    quaternion[..., 0] = 1
    return dict(
        joint_pos=np.tile(np.arange(29, dtype=np.float32), (3, 1)),
        body_pos_w=np.broadcast_to(np.arange(39)[None, :, None], (3, 39, 3)).copy(),
        body_quat_w=quaternion,
        fps=50.0,
        source_hash="sha256:" + "a" * 64,
    )


def test_mapping_is_explicit_and_does_not_invent_physics():
    reference = omnicontact_pose_reference(**inputs())
    assert sorted(LAB_TO_MUJOCO) == list(range(29))
    np.testing.assert_array_equal(reference["joint_position_mujoco_rad"][0], LAB_TO_MUJOCO)
    for name, index in BODY_INDICES.items():
        assert np.all(reference["body_position_m"][name] == index)
    assert "left_wrist_yaw_link" not in reference["body_position_m"]
    assert reference["support_state"] is reference["contact_force"] is None
    assert reference["receiving_success"] is None
    assert not reference["source_authenticated_here"]
    assert not reference["promotion_authorized"]


def test_reference_copies_are_readonly_and_source_is_unchanged():
    source = inputs()
    reference = omnicontact_pose_reference(**source)
    with pytest.raises(ValueError):
        reference["joint_position_mujoco_rad"][0, 0] = 9
    source["joint_pos"][0, 0] = 42
    assert reference["joint_position_mujoco_rad"][0, 0] == 0


@pytest.mark.parametrize("fps", [True, 0, -1, float("nan"), float("inf"), 1001])
def test_invalid_sample_rate(fps):
    source = inputs()
    source["fps"] = fps
    with pytest.raises(ValueError):
        omnicontact_pose_reference(**source)


@pytest.mark.parametrize("key", ["joint_pos", "body_pos_w", "body_quat_w"])
def test_nonfinite_arrays_rejected(key):
    source = inputs()
    source[key] = source[key].astype(float)
    source[key].flat[0] = np.nan
    with pytest.raises(ValueError):
        omnicontact_pose_reference(**source)


def test_wrong_body_shape_quaternion_and_missing_hash():
    for key, value in (
        ("body_pos_w", np.zeros((3, 38, 3))),
        ("body_quat_w", np.zeros((3, 39, 4))),
        ("source_hash", "unknown"),
    ):
        source = inputs()
        source[key] = value
        with pytest.raises(ValueError):
            omnicontact_pose_reference(**source)
