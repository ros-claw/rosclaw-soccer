from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw_soccer.growth.contextual_finish_target import (
    FinishTargetCalibrationSample,
    FinishTargetFailureMemory,
    fit_contextual_finish_target_actor,
    load_contextual_finish_target_actor,
    save_contextual_finish_target_actor,
)
from rosclaw_soccer.growth.physical_option_router import PhysicalSoccerOption
from rosclaw_soccer.growth.role_option_backend import (
    RoleOptionBackend,
    RoleOptionBackendCandidate,
)
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.contextual_finish_target_growth import (
    ContextualFinishTargetGrowthConfig,
    _selection_key,
)


def _hash(label: str) -> str:
    return str(hash_json({"s203": label}))


def _sample(index: int) -> FinishTargetCalibrationSample:
    target = (7.5, 0.89, 0.115)
    return FinishTargetCalibrationSample(
        context_hash=_hash(f"context-{index}"),
        trajectory_hash=_hash(f"trajectory-{index}"),
        control_envelope_hash=_hash("control-envelope"),
        features=(
            0.08 + 0.01 * index,
            1.29 + 0.005 * index,
            1.205 + 0.001 * index,
            -0.160 + 0.001 * index,
            0.10,
            (3.13 if index % 2 == 0 else -3.13),
            0.001 * index,
            -0.001 * index,
            0.80,
        ),
        requested_physical_target_m=target,
        executed_policy_target_m=(7.5, -0.41, 0.515),
        executed_foot_yaw_offset_rad=0.085,
        observed_crossing_m=(7.5, 0.89 + 0.005 * index, 0.115),
        target_error_m=0.005 * index,
        safe=True,
        exact_replay=True,
    )


def _actor(sample_count: int = 4):
    return fit_contextual_finish_target_actor(
        body_hash=_hash("body"),
        kick_prior_hash=_hash("kick"),
        roster_hash=_hash("roster"),
        finisher_self_model_hash=_hash("self"),
        control_envelope_hash=_hash("control-envelope"),
        source_evidence_hashes=(_hash("evidence"),),
        samples=tuple(_sample(index) for index in range(sample_count)),
    )


def _failure(index: int = 0) -> FinishTargetFailureMemory:
    return FinishTargetFailureMemory(
        context_hash=_hash(f"failed-context-{index}"),
        search_hash=_hash(f"failed-search-{index}"),
        control_envelope_hash=_hash("control-envelope"),
        features=_sample(index).features,
        failure_code="NO_PRECISE_SAFE_ACTION_IN_BOUNDED_SEARCH",
        candidate_count=30,
        safe_candidate_count=29,
        best_safe_target_error_m=0.228,
        exact_replay=True,
    )


def test_seed_with_one_context_is_explicitly_non_deployable() -> None:
    actor = _actor(1)

    decision = actor.decide(_sample(0).features, (7.5, 0.89, 0.115))

    assert not actor.evidence_ready
    assert not decision.accepted
    assert decision.route == "INSUFFICIENT_DISTINCT_PHYSICAL_SUPPORT"
    assert decision.policy_target_m is None
    assert decision.foot_yaw_offset_rad is None


def test_four_context_actor_uses_periodic_yaw_and_robust_residual() -> None:
    actor = _actor()
    query = replace(
        _sample(0), features=(*_sample(0).features[:5], -3.1531853071795863, 0.0, 0.0, 0.8)
    )

    decision = actor.decide(query.features, (7.5, 0.89, 0.115))

    assert actor.evidence_ready
    assert decision.accepted
    assert decision.route == "VERIFIED_CONTEXTUAL_FINISH_TARGET"
    assert decision.nearest_support_distance is not None
    assert decision.nearest_support_distance < 1.0
    assert decision.policy_target_m == pytest.approx((7.5, -0.41, 0.515))
    assert decision.foot_yaw_offset_rad == pytest.approx(0.085)


def test_out_of_distribution_target_context_fails_closed() -> None:
    actor = _actor()
    features = list(_sample(0).features)
    features[0] = 1.0

    decision = actor.decide(tuple(features), (7.5, 0.89, 0.115))

    assert not decision.accepted
    assert decision.route == "CONTEXTUAL_FINISH_TARGET_OOD_FALLBACK"
    assert decision.policy_target_m is None
    assert decision.foot_yaw_offset_rad is None


def test_known_failure_basin_vetoes_smooth_interpolation() -> None:
    actor = replace(_actor(), failure_memories=(_failure(),))

    decision = actor.decide(_sample(0).features, (7.5, 0.89, 0.115))

    assert not decision.accepted
    assert decision.route == "KNOWN_FINISH_FAILURE_BASIN_FALLBACK"
    assert decision.nearest_failure_distance == pytest.approx(0.0)
    assert decision.supporting_context_hashes == (_failure().context_hash,)


def test_failure_memory_cannot_hide_a_precise_action() -> None:
    with pytest.raises(ValueError, match="failure memory is invalid"):
        replace(_failure(), best_safe_target_error_m=0.10)


def test_inconsistent_physical_error_is_rejected() -> None:
    with pytest.raises(ValueError, match="sample is invalid"):
        replace(_sample(0), target_error_m=0.09)


def test_actor_rejects_samples_from_another_control_envelope() -> None:
    samples = tuple(
        replace(_sample(index), control_envelope_hash=_hash("other-envelope"))
        if index == 3
        else _sample(index)
        for index in range(4)
    )

    with pytest.raises(ValueError, match="violates its contract"):
        fit_contextual_finish_target_actor(
            body_hash=_hash("body"),
            kick_prior_hash=_hash("kick"),
            roster_hash=_hash("roster"),
            finisher_self_model_hash=_hash("self"),
            control_envelope_hash=_hash("control-envelope"),
            source_evidence_hashes=(_hash("evidence"),),
            samples=samples,
        )


def test_contextual_target_is_a_role_qualified_shoot_backend() -> None:
    candidate = RoleOptionBackendCandidate(
        backend=RoleOptionBackend.CONTEXTUAL_FINISH_TARGET,
        option=PhysicalSoccerOption.SHOOT,
        artifact_hash=_hash("actor"),
        evidence_hash=_hash("evidence"),
        distinct_context_count=4,
        distinct_trajectory_count=4,
        strict_replay=True,
        holdout_passed=True,
        parent_retention_passed=True,
    )

    assert candidate.evidence_ready


def test_single_context_cannot_fake_contextual_backend_coverage() -> None:
    candidate = RoleOptionBackendCandidate(
        backend=RoleOptionBackend.CONTEXTUAL_FINISH_TARGET,
        option=PhysicalSoccerOption.SHOOT,
        artifact_hash=_hash("actor"),
        evidence_hash=_hash("evidence"),
        distinct_context_count=1,
        distinct_trajectory_count=100,
        strict_replay=True,
        holdout_passed=True,
        parent_retention_passed=True,
    )

    assert not candidate.evidence_ready


def test_actor_artifact_round_trip_and_tamper_detection(tmp_path) -> None:
    path = tmp_path / "actor.json"
    actor = replace(_actor(), failure_memories=(_failure(),))
    save_contextual_finish_target_actor(actor, path)

    assert load_contextual_finish_target_actor(path) == actor

    path.write_text(path.read_text().replace("0.515", "0.516"))
    with pytest.raises(ValueError, match="integrity"):
        load_contextual_finish_target_actor(path)


def test_growth_config_bounds_joint_search_budget() -> None:
    with pytest.raises(ValueError, match="config is invalid"):
        ContextualFinishTargetGrowthConfig(
            policy_target_y_candidates_m=tuple(index / 100 for index in range(8)),
            foot_yaw_offset_candidates_rad=tuple(index / 100 for index in range(8)),
        )


def test_selection_never_prefers_precise_but_unsafe_candidate() -> None:
    unsafe = {
        "safe": False,
        "policy_target_m": [7.5, 0.2, 0.5],
        "foot_yaw_offset_rad": 0.08,
        "result": {
            "pass_contact_time_sec": 5.0,
            "shot_contact_time_sec": 7.0,
            "target_error_m": 0.01,
            "shooter_min_pelvis_height_m": 0.1,
            "shooter_post_contact_support_foot_slip_m": 0.01,
        },
    }
    safe = {
        "safe": True,
        "policy_target_m": [7.5, 0.1, 0.5],
        "foot_yaw_offset_rad": 0.08,
        "result": {
            "pass_contact_time_sec": 5.0,
            "shot_contact_time_sec": 7.0,
            "target_error_m": 0.20,
            "shooter_min_pelvis_height_m": 0.65,
            "shooter_post_contact_support_foot_slip_m": 0.05,
        },
    }

    assert min((unsafe, safe), key=_selection_key) is safe
