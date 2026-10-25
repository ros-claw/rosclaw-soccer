import numpy as np
import pytest

from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays


def trace():
    n = 3000
    q = np.zeros((n, 43))
    q[:, 9] = 0.8
    q[:, 10] = 1.0
    ball = np.zeros((300, 3))
    ball[240:, 0] = np.arange(1, 61) / 50
    q[:, :3] = np.repeat(ball, 10, axis=0)
    force = np.zeros((n, 1, 3))
    force[2390, 0, 1] = 2
    return dict(
        qpos=q,
        qvel=np.zeros((n, 41)),
        torque=np.zeros((n, 29)),
        ball_forces=force,
        joint_targets=np.zeros((300, 29)),
        commands=np.zeros((300, 3)),
        ball_position_after_step_m=ball,
        pelvis_z_per_substep_m=np.full((300, 10), 0.8),
    )


def test_actual_native_indices_and_239_boundary_reconstruct_outcome():
    arrays = trace()
    r = relabel_arrays(arrays, controls=300, ball_address=0, root_address=7)
    assert r["task_pass"] and r["first_foot_contact_frame"] == 239
    assert r["forward_60_m"] == pytest.approx(1.2)
    arrays["ball_forces"][2390, 0, 1] = 0
    arrays["ball_forces"][2400, 0, 1] = 2
    r = relabel_arrays(arrays, controls=300, ball_address=0, root_address=7)
    assert not r["task_pass"] and r["forward_60_m"] is None


@pytest.mark.parametrize("defect", ["fakeball", "fakeheight", "nan", "wrongaddress"])
def test_labels_do_not_trust_saved_measurement_fields(defect):
    arrays = trace()
    address = 0
    if defect == "fakeball":
        arrays["ball_position_after_step_m"][0, 0] += 1
    elif defect == "fakeheight":
        arrays["pelvis_z_per_substep_m"][0, 0] -= 0.1
    elif defect == "nan":
        arrays["qvel"][0, 0] = np.nan
    else:
        address = 7
    with pytest.raises(ValueError):
        relabel_arrays(arrays, controls=300, ball_address=address, root_address=7)
