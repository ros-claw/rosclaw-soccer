import numpy as np
import pytest

from rosclaw_soccer.rsi.cftrack_motor_authority import summarize_motor_future


def nomination():
    from dataclasses import asdict

    from rosclaw_soccer.rsi.cftrack_motor_authority import MOTOR_AXES
    from rosclaw_soccer.rsi.cftrack_motor_residual_plan import CFtrackMotorResidualPlan

    rows = [dict(name="parent", metrics=dict(first_foot_control=71))]
    for i in range(8):
        delta = [0.0] * 29
        delta[MOTOR_AXES[i // 2][1]] = 0.15 if i % 2 else -0.15
        p = CFtrackMotorResidualPlan(0, 46, 24, tuple(delta), "sha256:" + "a" * 64)
        rows.append(
            dict(
                name=f"arm-{i:02d}",
                motor_plan=asdict(p),
                metrics=dict(local_post60_conditions_met=i == 3),
                future=dict(controls=160, body_failed=False, runtime_failure=None),
            )
        )
    return dict(
        status="CFTRACK_MOTOR_AUTHORITY_LOCAL_RESCUE",
        private_post60_is_not_full_h300_exam=True,
        actual_native_steps=0,
        neural_optimizer_updates=0,
        records=rows,
        selected_local_candidate="arm-03",
    )


def test_decode_frozen_nominee_preserves_only_predeclared_axis_window():
    from rosclaw_soccer.rsi.cftrack_motor_authority import selected_motor_plan

    p = selected_motor_plan(nomination())
    assert p.start_control == 46 and p.delta_hg[7] == 0.15
    assert sum(v != 0 for v in p.delta_hg) == 1


@pytest.mark.parametrize(
    "defect",
    ["unclosed", "partial", "parent", "unsafe", "axis", "window", "amplitude", "lateorigin"],
)
def test_invalid_nomination_rejected_without_execution(defect):
    from rosclaw_soccer.rsi.cftrack_motor_authority import selected_motor_plan

    report = nomination()
    row = report["records"][4]
    if defect == "unclosed":
        report["status"] = "RUNNING"
    elif defect == "partial":
        report["records"].pop()
    elif defect == "parent":
        report["selected_local_candidate"] = "parent"
    elif defect == "unsafe":
        row["future"]["body_failed"] = True
    elif defect == "axis":
        row["motor_plan"]["delta_hg"] = tuple([0.15] + [0.0] * 28)
    elif defect == "window":
        row["motor_plan"]["start_control"] = 45
    elif defect == "amplitude":
        row["motor_plan"]["delta_hg"] = (0.3,) * 29
    elif defect == "lateorigin":
        row["motor_plan"]["origin_counter"] = 1
    with pytest.raises(ValueError):
        selected_motor_plan(report)


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
