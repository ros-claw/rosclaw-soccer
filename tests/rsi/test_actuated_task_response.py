"""Synthetic actual-target direction algebra/finite-search tests, not G1 evidence."""

from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.actuated_response_kick_player import ActuatedResponseKickPlayer
from rosclaw_soccer.rsi.actuated_task_response import actuated_task_plan


def test_every_signed_actual_joint_direction_uses_existing_rank_one_codec():
    for axis in range(29):
        for sign in (-1, 1):
            delta = np.zeros(29, np.float64)
            delta[axis] = sign * 0.03
            plan = actuated_task_plan(70, delta)
            assert plan.projection_mode == "per_axis"
            assert np.asarray(plan.basis_hg).shape == (1, 29)
            assert plan.basis_hg[0][axis] == 1
            assert plan.coefficients == ((sign * 0.03,), (0.0,))
            offsets = plan.offsets()
            np.testing.assert_array_equal(offsets[12], delta)
            assert not np.any(offsets[24:])
            assert not np.any(offsets[0])


def test_task_direction_is_not_restricted_to_fixed_success_pca_axes():
    delta = np.linspace(-0.003, 0.003, 29)
    assert abs(delta).sum() < 0.06
    plan = actuated_task_plan(80, delta)
    np.testing.assert_allclose(plan.offsets()[12], delta, rtol=0, atol=1e-17)
    assert np.max(abs(plan.offsets())) <= 0.03000001
    np.testing.assert_array_equal(
        actuated_task_plan(80, np.zeros(29)).offsets(), np.zeros((49, 29))
    )


@pytest.mark.parametrize(
    "delta",
    [
        np.zeros(28),
        np.zeros(29, np.float32),
        np.full(29, np.nan),
        np.full(29, np.inf),
        np.full(29, 0.03),
        np.eye(29)[0] * 0.031,
    ],
)
def test_no_new_axis_or_combined_trust_authority(delta):
    with pytest.raises(ValueError):
        actuated_task_plan(70, delta)


@pytest.mark.parametrize("start", [-1, 240, True, 1.0])
def test_only_actual_formal_task_entry_clock(start):
    with pytest.raises(ValueError):
        actuated_task_plan(start, np.zeros(29))


@pytest.mark.parametrize("baseline_observed", [True, False])
def test_exact_signed_probe_budget_one_solver_and_at_most_one_proposal(
    monkeypatch, baseline_observed
):
    import rosclaw_soccer.rsi.actuated_response_kick_player as module

    def prediction(delta):
        f = np.array([0.8 + 10 * delta[0], 0.4 + 10 * delta[1], 1.5 - 50 * delta[2]])
        force = np.zeros((1280, 1, 3))
        force[:, 0, 2] = f[2]
        return dict(features=f, ball_forces=force, controls=128, donor_control_frame=70)

    def window(prediction, **kwargs):
        f = prediction["features"]
        return dict(
            predicted_body_healthy=True,
            quality_observed=True,
            forward_60_m=float(f[0]),
            lateral_60_m=float(f[1]),
        )

    class Forecaster:
        def __init__(self):
            self.plans = []

        def forecast(self, *, controls, before_proposal, target_plan):
            assert controls == 128 and before_proposal
            self.plans.append(target_plan)
            return prediction(np.asarray(target_plan.basis_hg[0]) * target_plan.coefficients[0][0])

    monkeypatch.setattr(module, "predicted_touch_window", window)
    controller = object.__new__(ActuatedResponseKickPlayer)
    controller.world = SimpleNamespace(control_steps=70)
    controller.layout = SimpleNamespace(bq=0, rq=7)
    controller.forecaster = Forecaster()
    parent = prediction(np.zeros(29))
    parent_window = window(parent) | dict(predicted_body_healthy=baseline_observed)
    candidates = [("parent", None, parent, parent_window)]
    report = controller._search_additional_candidates(candidates, 128)
    expected = 59 if baseline_observed else 58
    assert len(controller.forecaster.plans) == len(report["candidate_plans"]) == expected
    assert len(candidates) == 1 + expected
    assert not report["fitted_pca_basis_used_for_new_actions"]
    assert not report["policy_reference_modified"]
    assert not report["new_action_or_torque_limits"]
    assert report["neural_optimizer_updates"] == 0
    if baseline_observed:
        delta = np.array(report["candidate_plans"][-1]["first_lobe_offset_hg_rad"])
        assert abs(delta).sum() <= 0.06000001
        assert abs(delta).max() <= 0.03000001
        assert candidates[-1][0] == "actual-task-response"
    else:
        assert report["solver_report"]["status"] == "ABSTAIN_BASELINE_CENSORED_OR_BODY_FAILED"
