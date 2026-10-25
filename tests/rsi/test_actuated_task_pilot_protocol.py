"""Synthetic admission and independent solver replay; not physical task evidence."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.actuated_response_kick_player import ActuatedResponseKickPlayer
from scripts.rsi_actuated_task_pilot_common import ACTUATED_INDICES, verify_actuated_response


def synthetic_search(monkeypatch, observed):
    import rosclaw_soccer.rsi.actuated_response_kick_player as module

    def prediction(delta):
        force = np.zeros((1280, 1, 3))
        force[:, 0, 2] = 1.5 - 50 * delta[2]
        return dict(
            forward=0.8 + 10 * delta[0],
            lateral=0.4 + 10 * delta[1],
            ball_forces=force,
            controls=128,
            donor_control_frame=70,
        )

    def window(prediction, **kwargs):
        return dict(
            predicted_body_healthy=True,
            quality_observed=True,
            forward_60_m=float(prediction["forward"]),
            lateral_60_m=float(prediction["lateral"]),
        )

    class Forecaster:
        def forecast(self, *, controls, before_proposal, target_plan):
            assert controls == 128 and before_proposal
            return prediction(np.array(target_plan.basis_hg[0]) * target_plan.coefficients[0][0])

    monkeypatch.setattr(module, "predicted_touch_window", window)
    ctrl = object.__new__(ActuatedResponseKickPlayer)
    ctrl.world = SimpleNamespace(control_steps=70)
    ctrl.layout = SimpleNamespace(bq=0, rq=7)
    ctrl.forecaster = Forecaster()
    parent = prediction(np.zeros(29))
    parent_window = window(parent) | dict(predicted_body_healthy=observed)
    candidates = [("parent", None, parent, parent_window)] * 37
    additional = ctrl._search_additional_candidates(candidates, 128)
    windows = [
        dict(name=c[0], coefficients=None if c[1] is None else c[1].coefficients, outcome=c[3])
        for c in candidates
    ]
    return (
        windows,
        [c[2] for c in candidates],
        dict(additional_candidate_search=additional, joint_task_response=None),
    )


@pytest.mark.parametrize("observed", [True, False])
def test_full_raw_derived_signed_response_reconstructs(monkeypatch, observed):
    assert ACTUATED_INDICES == (0, 2, 6, 8, 21, 30)
    windows, forecasts, event = synthetic_search(monkeypatch, observed)
    verify_actuated_response(windows, forecasts, event, 70)
    assert len(windows) == (96 if observed else 95)


@pytest.mark.parametrize("violation", ["axis", "basis", "clock", "fit", "authority", "count"])
def test_reject_edited_plan_solver_or_claim(monkeypatch, violation):
    windows, forecasts, original = synthetic_search(monkeypatch, True)
    event = deepcopy(original)
    extra = event["additional_candidate_search"]
    plans = extra["candidate_plans"]
    if violation == "axis":
        plans[0]["first_lobe_offset_hg_rad"][0] *= -1
    elif violation == "basis":
        plans[0]["basis_hg"] = ((0.0,) * 29,)
    elif violation == "clock":
        plans[0]["start_control"] += 1
    elif violation == "fit":
        extra["solver_report"]["status"] = "FABRICATED_SUCCESS"
    elif violation == "authority":
        extra["new_action_or_torque_limits"] = True
    elif violation == "count":
        plans.pop()
    with pytest.raises(ValueError):
        verify_actuated_response(windows, forecasts, event, 70)


@pytest.mark.parametrize(
    "flags",
    [
        ["--actuated-task-response"],
        ["--actuated-task-response", "--target-feasible-axes"],
        ["--task-qualification-audit-hash", "orphan"],
        ["--target-feasible-axes"],
    ],
)
def test_new_protocol_rejects_incomplete_flags_before_source_or_assets(monkeypatch, flags):
    from scripts import rsi_run_joint_response_pilot as runner

    def never_clean(head):
        pytest.fail("invalid protocol must fail before source or model loading")

    monkeypatch.setattr("scripts.rsi_native_fresh_common.clean", never_clean)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot",
            "--champion-audit",
            "unused",
            "--champion-root",
            "unused",
            "--qualification",
            "unused",
            "--backbone-model",
            "unused",
            "--output",
            "not-created",
            "--champion-audit-hash",
            "unused",
            "--qualification-hash",
            "unused",
            "--expected-head",
            "unused",
            *flags,
        ],
    )
    with pytest.raises(ValueError):
        runner.main()


@pytest.mark.parametrize("other", [None, "--aim-command-search", "--executed-target-search"])
def test_complete_new_mode_is_admitted_but_cannot_mix_other_families(monkeypatch, other):
    from scripts import rsi_run_joint_response_pilot as runner

    class AdmissionReached(Exception):
        pass

    def stop_before_assets(head):
        raise AdmissionReached

    monkeypatch.setattr("scripts.rsi_native_fresh_common.clean", stop_before_assets)
    argv = ["pilot"]
    for name in (
        "champion-audit",
        "champion-root",
        "qualification",
        "backbone-model",
        "output",
        "champion-audit-hash",
        "qualification-hash",
        "expected-head",
        "target-qualification",
        "target-qualification-hash",
        "task-qualification-audit",
        "task-qualification-audit-hash",
    ):
        argv.extend(("--" + name, "unused"))
    argv.extend(("--actuated-task-response", "--target-feasible-axes"))
    if other:
        argv.append(other)
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(ValueError if other else AdmissionReached):
        runner.main()
