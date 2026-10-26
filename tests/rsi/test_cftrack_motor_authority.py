import numpy as np
import pytest

from rosclaw_soccer.rsi.cftrack_motor_authority import summarize_motor_future


def fixture():
    q = np.zeros((1600, 43))
    q[:, 9] = 0.77
    q[1000:, 0] = 1.2
    q[1000:, 1] = 0.24
    forces = np.zeros((1600, 1, 3))
    forces[400, 0, 1] = 10
    return dict(
        controls=160,
        qpos=q,
        ball_forces=forces,
        joint_targets=np.zeros((160, 29)),
        nn_decoded_targets=np.zeros((160, 29)),
        body_failed=False,
        runtime_failure=None,
    )


def test_native_ball_first_addresses_and_local_outcome_never_claim_full_exam():
    m = summarize_motor_future(fixture(), ball_address=0, root_address=7)
    assert m["first_foot_control"] == 40 and m["forward_60_m"] == 1.2
    assert m["minimum_root_z_m"] == 0.77 and m["lateral_forward_ratio"] == 0.2
    assert m["local_post60_conditions_met"] and not m["official_h300_exam"]
    assert m["full_task_pass_not_evaluated"]


@pytest.mark.parametrize(
    "defect", ["late", "unsafe", "body", "runtime", "side", "nan", "wrong_address"]
)
def test_bad_or_unobserved_private_future_is_not_local_rescue(defect):
    f = fixture()
    if defect == "late":
        f["ball_forces"][:] = 0
        f["ball_forces"][1000, 0, 0] = 3
    elif defect == "unsafe":
        f["ball_forces"][700, 0, 2] = 2
    elif defect == "body":
        f["body_failed"] = True
    elif defect == "runtime":
        f["runtime_failure"] = {"message": "failed"}
    elif defect == "side":
        f["qpos"][1000:, 1] = 1
    elif defect == "nan":
        f["qpos"][0, 2] = np.nan
    if defect in ("nan", "wrong_address"):
        with pytest.raises(ValueError):
            summarize_motor_future(
                f, ball_address=0, root_address=0 if defect == "wrong_address" else 7
            )
    else:
        assert not summarize_motor_future(f, ball_address=0, root_address=7)[
            "local_post60_conditions_met"
        ]
