from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw_soccer.growth.dynamic_lead_pass import (
    LeadPassCalibrationSample,
    fit_dynamic_lead_pass_policy,
)
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.dynamic_lead_pass_evidence import (
    DynamicLeadPassEvidenceConfig,
    DynamicLeadPassHoldout,
)


def _sample(*, sample_id: str, phase: float, yaw: float) -> LeadPassCalibrationSample:
    return LeadPassCalibrationSample(
        sample_id=sample_id,
        receiver_phase_start_sec=phase,
        passer_yaw_delta_rad=yaw,
        delivery_position_m=(3.50 - 1.15 * phase, -0.02 - 3.80 * yaw, 0.115),
        trajectory_hash=str(hash_json({"sample": sample_id})),
        safe=True,
    )


def _samples() -> tuple[LeadPassCalibrationSample, ...]:
    return tuple(
        [
            _sample(sample_id=f"phase-{index}", phase=phase, yaw=0.0)
            for index, phase in enumerate((1.80, 1.96, 2.04))
        ]
        + [
            _sample(sample_id=f"yaw-{index}", phase=1.96, yaw=yaw)
            for index, yaw in enumerate((-0.06, -0.03, 0.02, 0.03, 0.06))
        ]
    )


def test_lead_pass_fit_predicts_future_pocket_and_executes_yaw_action() -> None:
    policy = fit_dynamic_lead_pass_policy(_samples())

    target = policy.reception_target(
        receiver_phase_start_sec=1.92,
        receiver_lateral_lane_m=0.10,
    )

    assert target == pytest.approx((1.292, 0.10, 0.115))
    assert policy.longitudinal_fit_r2 == pytest.approx(1.0)
    assert policy.lateral_fit_r2 == pytest.approx(1.0)
    assert policy.passer_yaw_delta(target_lateral_m=0.10) == pytest.approx(-0.03157894736842106)
    assert policy.passer_world_yaw(target_lateral_m=0.10) < 3.141592653589793
    assert policy.artifact_hash.startswith("sha256:")


def test_lead_pass_wraps_positive_yaw_delta_inside_simulator_domain() -> None:
    policy = fit_dynamic_lead_pass_policy(_samples())

    yaw = policy.passer_world_yaw(target_lateral_m=-0.10)

    assert -3.141592653589793 < yaw < 0.0


def test_lead_pass_rejects_unsafe_discovery_sample() -> None:
    samples = list(_samples())
    samples[0] = replace(samples[0], safe=False)

    with pytest.raises(ValueError, match="safe samples"):
        fit_dynamic_lead_pass_policy(tuple(samples))


def test_lead_pass_action_fails_closed_outside_calibrated_envelope() -> None:
    policy = fit_dynamic_lead_pass_policy(_samples())

    with pytest.raises(ValueError, match="yaw envelope"):
        policy.passer_yaw_delta(target_lateral_m=0.30)


def test_holdout_phase_cannot_leak_into_discovery() -> None:
    with pytest.raises(ValueError, match="sealed from discovery"):
        DynamicLeadPassEvidenceConfig(
            holdouts=(DynamicLeadPassHoldout("leaked-left", 1.96, -0.10),)
        )


def test_default_round_is_one_plastic_role_and_sealed() -> None:
    config = DynamicLeadPassEvidenceConfig()

    assert {case.receiver_phase_start_sec for case in config.holdouts}.isdisjoint(
        config.discovery_receiver_phase_starts_sec
    )
    assert config.activation_ceiling == "SIM_ONLY"
    assert not config.hardware_authorized
