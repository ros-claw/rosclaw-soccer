import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_runup import (
    ISAACLAB_TO_MUJOCO,
    G1SonicRunupController,
    _sonic_control_parameters,
)
from rosclaw_soccer.rsi.contact_reference_labels import reference_labels


def trace():
    raw = np.random.default_rng(4).normal(size=(300, 1, 29)).astype(np.float32)
    scale = _sonic_control_parameters(1.0, (1.0,) * 29)[2]
    target = G1SonicRunupController.default_angles + raw[:, :, ISAACLAB_TO_MUJOCO] * scale
    pre = target.copy()
    pre[40:50, :, 0] += 0.2
    return dict(
        foundation_neural_raw_action_isaac=raw,
        pre_motor_joint_target_rad=pre,
        navigation_command=np.zeros((300, 1, 3), dtype=np.float64),
        foundation_navigation_joint_target_rad=target,
        taskspace_joint_increment_rad=pre - target,
    )


def test_exact_same_state_labels_without_fitting_or_future_inputs():
    source = trace()
    result = reference_labels(source)
    assert (
        result["navigation_joint_target_rad"].tobytes()
        == source["foundation_navigation_joint_target_rad"].tobytes()
    )
    assert (
        result["pre_parent_reference_increment_rad"].tobytes()
        == source["taskspace_joint_increment_rad"][:, :, :12].tobytes()
    )
    assert result["pre_parent_reference_increment_rad"].shape == (300, 1, 12)
    result["navigation_command"][:] = 1
    assert not source["navigation_command"].any()


def test_old_none_capture_can_reconstruct_without_new_taskspace_fields():
    source = trace()
    source.pop("foundation_navigation_joint_target_rad")
    source.pop("taskspace_joint_increment_rad")
    assert reference_labels(source)["pre_parent_reference_increment_rad"].any()


def test_float32_foundation_target_is_not_the_actual_committed_navigation_target():
    source = trace()
    source["foundation_navigation_joint_target_rad"] = source[
        "foundation_navigation_joint_target_rad"
    ].astype(np.float32)
    with pytest.raises(ValueError, match="byte-exact"):
        reference_labels(source)


@pytest.mark.parametrize("tamper", ["shape", "nan", "dtype", "delta", "upper", "bound"])
def test_no_silent_coercion_clipping_or_substitution(tamper):
    source = trace()
    if tamper == "shape":
        source["navigation_command"] = source["navigation_command"][:299]
    elif tamper == "nan":
        source["pre_motor_joint_target_rad"][0, 0, 0] = np.nan
    elif tamper == "dtype":
        source["foundation_neural_raw_action_isaac"] = source[
            "foundation_neural_raw_action_isaac"
        ].astype(np.float64)
    elif tamper == "delta":
        source["taskspace_joint_increment_rad"][0, 0, 0] = 0.1
    else:
        source.pop("taskspace_joint_increment_rad")
        source["pre_motor_joint_target_rad"][0, 0, 12 if tamper == "upper" else 0] += 0.4
    with pytest.raises(ValueError):
        reference_labels(source)
