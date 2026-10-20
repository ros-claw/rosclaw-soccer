import numpy as np
import pytest

from rosclaw_soccer.rsi.native_motion_reference import reference_inputs70


def inputs():
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return dict(
        root_rotation=rotation,
        anchor_position=np.zeros(3),
        anchor_rotation=rotation.copy(),
        joint_position=np.arange(29, dtype=float) * 0.01,
        joint_velocity=np.arange(29, dtype=float) * 0.02,
        reference_position=np.array([1.0, 0.0, 0.0]),
        reference_quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        reference_linear_velocity=np.array([0.2, 0.0, 0.0]),
        reference_angular_velocity=np.array([0.0, 0.0, 0.3]),
    )


def test_native_reference_transform_and_exact_channel_semantics():
    source = inputs()
    before = {k: v.copy() for k, v in source.items()}
    command, suffix = reference_inputs70(**source)
    np.testing.assert_allclose(command, [0, -0.2, 0.3], atol=1e-7)
    assert suffix.dtype == command.dtype == np.float32
    np.testing.assert_array_equal(suffix[:29], source["joint_position"].astype(np.float32))
    np.testing.assert_array_equal(suffix[29:58], source["joint_velocity"].astype(np.float32))
    np.testing.assert_allclose(suffix[58:61], [0, -1, 0], atol=1e-7)
    np.testing.assert_allclose(suffix[61:67], [0, -1, 1, 0, 0, 0], atol=1e-7)
    np.testing.assert_allclose(suffix[67:], [0, -0.2, 0], atol=1e-7)
    assert all(np.array_equal(source[k], v) for k, v in before.items())


@pytest.mark.parametrize("kind", ["shape", "nan", "rotation", "quaternion"])
def test_unqualified_reference_rejected_without_repair(kind):
    source = inputs()
    if kind == "shape":
        source["joint_position"] = np.zeros(28)
    elif kind == "nan":
        source["joint_velocity"][0] = np.nan
    elif kind == "rotation":
        source["anchor_rotation"][0] *= 2
    else:
        source["reference_quaternion"][:] = 0
    with pytest.raises(ValueError):
        reference_inputs70(**source)


def test_position_feedback_only_changes_bounded_velocity_command():
    source = inputs()
    original, suffix = reference_inputs70(**source)
    command, with_feedback = reference_inputs70(**source, position_gain=0.1)
    np.testing.assert_allclose(command - original, [0, -0.1, 0], atol=1e-7)
    np.testing.assert_array_equal(suffix, with_feedback)
    bounded, _ = reference_inputs70(**source, position_gain=1.0)
    assert bounded[1] == -0.5
    with pytest.raises(ValueError, match="gain"):
        reference_inputs70(**source, position_gain=-1.0)
