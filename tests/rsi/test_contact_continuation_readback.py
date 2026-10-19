"""Synthetic readback corruption tests, not G1 performance evidence."""

import copy

import numpy as np
import pytest

from scripts.rsi_read_contact_continuation import verify_arrays


def fixture():
    arrays = dict(
        canonical_qpos_after_substep=np.zeros((20, 43)),
        canonical_qvel_after_substep=np.zeros((20, 41)),
        requested_pd_torque_nm=np.zeros((20, 29)),
        applied_control_nm=np.zeros((20, 29)),
        actual_actuator_force_nm=np.zeros((20, 29)),
        pelvis_height_solver_epoch_m=np.full(20, 0.7),
        robot_force_on_ball_world_n=np.zeros((20, 3)),
        completed_foot_ground_normal_n=np.zeros((20, 2)),
    )
    trace = dict(
        canonical_qpos=np.zeros((300, 1, 43)),
        canonical_qvel=np.zeros((300, 1, 41)),
        pd_requested_torque_nm=np.zeros((300, 1, 10, 29)),
        torque_nm=np.zeros((300, 1, 10, 29)),
        actual_actuator_force_nm=np.zeros((300, 1, 10, 29)),
        pelvis_z_per_substep_m=np.full((300, 1, 10), 0.7),
    )
    geometry = dict(
        contact=dict(
            first_foot_substep=9,
            robot_impulse_initial_window_world_n_s=[0.0, 0.0, 0.0],
            ball_velocity_after_initial_window_world_m_s=[0.0, 0.0, 0.0],
        )
    )
    return arrays, copy.deepcopy(arrays), copy.deepcopy(arrays), trace, geometry


def test_raw_native_and_dense_clock_parity():
    result = verify_arrays(*fixture())
    assert result["frozen_limits_satisfied"]


@pytest.mark.parametrize(
    "bad",
    [
        "repeat",
        "prefix",
        "native_q",
        "native_pd",
        "native_torque",
        "native_pelvis",
        "dense_impulse",
        "dense_velocity",
        "missing",
        "nan",
    ],
)
def test_every_original_clock_and_raw_prediction_is_checked(bad):
    held, repeated, oracle, trace, geometry = fixture()
    if bad == "repeat":
        repeated["canonical_qpos_after_substep"][19, 0] = 1
    elif bad == "prefix":
        oracle["completed_foot_ground_normal_n"][0, 0] = 1
    elif bad == "native_q":
        trace["canonical_qpos"][2, 0, 7] = 1
    elif bad == "native_pd":
        trace["pd_requested_torque_nm"][1, 0, 5, 0] = 1
    elif bad == "native_torque":
        trace["torque_nm"][1, 0, 5, 0] = 1
    elif bad == "native_pelvis":
        trace["pelvis_z_per_substep_m"][1, 0, 9] = 1
    elif bad == "dense_impulse":
        geometry["contact"]["robot_impulse_initial_window_world_n_s"][0] = 1
    elif bad == "dense_velocity":
        geometry["contact"]["ball_velocity_after_initial_window_world_m_s"][0] = 1
    elif bad == "missing":
        held.pop("completed_foot_ground_normal_n")
    else:
        oracle["robot_force_on_ball_world_n"][19, 0] = np.nan
    with pytest.raises(ValueError):
        verify_arrays(held, repeated, oracle, trace, geometry)
