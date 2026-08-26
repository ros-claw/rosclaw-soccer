from __future__ import annotations

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.training.recovery_baseline_adapter import (
    HOST_G1_PRONE_ACTION_CONTRACT,
    HUMANUP_G1_ACTION_CONTRACT,
    adapt_recovery_baseline_action,
)


def test_humanup_expands_default_relative_23dof_action_and_holds_wrists() -> None:
    current = np.linspace(-0.1, 0.1, 29)
    default = np.zeros(29)
    source = np.full(23, 0.2)
    result = adapt_recovery_baseline_action(
        contract=HUMANUP_G1_ACTION_CONTRACT,
        source_action=source,
        current_joint_position_rad=current,
        default_joint_position_rad=default,
    )
    assert result.accepted
    omitted = [
        G1_DDS_JOINT_NAMES.index(name) for name in HUMANUP_G1_ACTION_CONTRACT.omitted_joint_names
    ]
    included = [
        G1_DDS_JOINT_NAMES.index(name) for name in HUMANUP_G1_ACTION_CONTRACT.source_joint_names
    ]
    assert np.allclose(result.target_joint_position_rad[included], 0.1)
    assert np.array_equal(result.target_joint_position_rad[omitted], current[omitted])
    assert set(HUMANUP_G1_ACTION_CONTRACT.omitted_joint_names) == {
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    }


def test_host_mapping_keeps_only_pitch_yaw_wrist_joints_uncontrolled() -> None:
    current = np.zeros(29)
    source = np.full(23, 0.1)
    result = adapt_recovery_baseline_action(
        contract=HOST_G1_PRONE_ACTION_CONTRACT,
        source_action=source,
        current_joint_position_rad=current,
        default_joint_position_rad=np.full(29, 9.0),
    )
    assert result.accepted
    assert set(HOST_G1_PRONE_ACTION_CONTRACT.omitted_joint_names) == {
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    }
    assert np.count_nonzero(result.target_joint_position_rad) == 23


def test_baseline_adapter_fails_closed_without_silent_clipping() -> None:
    current = np.linspace(-0.2, 0.2, 29)
    result = adapt_recovery_baseline_action(
        contract=HOST_G1_PRONE_ACTION_CONTRACT,
        source_action=np.ones(23),
        current_joint_position_rad=current,
        default_joint_position_rad=np.zeros(29),
    )
    assert not result.accepted
    assert result.reasons == ("TARGET_DELTA_EXCEEDED",)
    assert np.array_equal(result.target_joint_position_rad, current)
    assert result.activation_ceiling == "SIM_ONLY"
    assert not result.hardware_authorized
