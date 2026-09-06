from __future__ import annotations

import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from rosclaw_soccer.growth.contextual_finish_intent import (
    DEFAULT_CONTEXTUAL_FINISH_INTENT_SCALE,
    ContextualFinishIntentAction,
    ContextualFinishIntentActor,
    ContextualFinishIntentSample,
    contextual_finish_intent_features,
    load_contextual_finish_intent_actor,
    save_contextual_finish_intent_actor,
)
from rosclaw_soccer.growth.physical_option_router import PhysicalSoccerOption
from rosclaw_soccer.growth.role_option_backend import (
    RoleOptionBackend,
    RoleOptionBackendCandidate,
)
from rosclaw_soccer.growth.runtime_finish_plan_actor import (
    prepared_finish_plan_features,
)
from rosclaw_soccer.media.contextual_finish_intent_video import (
    validate_contextual_finish_intent_video_manifest,
)
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.contextual_finish_intent_portfolio import (
    ContextualFinishIntentPortfolioConfig,
    _config_from_dict,
    validate_contextual_finish_intent_portfolio,
)


def _hash(label: str) -> str:
    return str(hash_json({"label": label}))


def _features(phase: float, lane: float) -> tuple[float, ...]:
    prepared = prepared_finish_plan_features(
        receiver_lane_m=lane,
        reception_target_x_m=1.33 - (phase - 1.90),
        passer_ball_local_xy_m=(1.205, -0.16),
        ball_ground_friction=0.10,
        passer_yaw_rad=3.115 - lane * 0.01,
        passer_stance_offset_xy_m=(0.0, 0.0),
        passer_swing_speed_scale=0.80,
    )
    return contextual_finish_intent_features(
        receiver_phase_start_sec=phase,
        prepared_features=prepared,
    )


def _actor() -> ContextualFinishIntentActor:
    source_hash = _hash("source")
    control_hash = _hash("control")
    samples = tuple(
        ContextualFinishIntentSample(
            context_hash=_hash(f"context-{index}"),
            trajectory_hash=_hash(f"trajectory-{index}"),
            control_envelope_hash=control_hash,
            source_evidence_hash=source_hash,
            features=_features(1.90, 0.0788 + index * 0.0002),
            action=ContextualFinishIntentAction(
                policy_target_y_m=0.30 + index * 0.01,
                foot_yaw_offset_rad=0.04,
                stance_offset_y_m=-0.02,
                foot_pitch_offset_rad=0.01,
            ),
            requested_physical_target_m=(7.5, 0.80, 0.115),
            observed_crossing_m=(7.5, 0.76, 0.115),
            target_error_m=0.04,
            safe=True,
            stability_retained=True,
            exact_replay=True,
        )
        for index in range(4)
    )
    return ContextualFinishIntentActor(
        body_hash=_hash("body"),
        kick_prior_hash=_hash("kick"),
        roster_hash=_hash("roster"),
        finisher_self_model_hash=_hash("self"),
        control_envelope_hash=control_hash,
        source_evidence_hashes=(source_hash,),
        feature_scale=DEFAULT_CONTEXTUAL_FINISH_INTENT_SCALE,
        samples=samples,
    )


def test_actor_selects_exact_local_expert_without_interpolation() -> None:
    actor = _actor()
    query = _features(1.90, 0.07882)

    decision = actor.decide(query)

    assert decision.accepted
    assert decision.route == "VERIFIED_LOCAL_FINISH_INTENT_EXPERT"
    assert decision.action == actor.samples[0].action
    assert decision.supporting_context_hash == actor.samples[0].context_hash
    assert decision.nearest_support_distance is not None
    assert decision.nearest_support_distance < actor.maximum_support_distance


def test_actor_rejects_contact_phase_boundary_without_action() -> None:
    actor = _actor()

    decision = actor.decide(_features(1.901, 0.07882))

    assert not decision.accepted
    assert decision.route == "CONTEXTUAL_FINISH_INTENT_OOD_FALLBACK"
    assert decision.action is None
    assert decision.supporting_context_hash is None


def test_actor_artifact_round_trip_is_content_bound(tmp_path: Path) -> None:
    actor = _actor()
    path = tmp_path / "actor.json"

    save_contextual_finish_intent_actor(actor, path)
    restored = load_contextual_finish_intent_actor(path)

    assert restored == actor
    assert restored.actor_hash == actor.actor_hash


def test_actor_cannot_expand_support_scale_or_gain_torque_authority() -> None:
    actor = _actor()

    with pytest.raises(ValueError, match="violates its contract"):
        replace(actor, feature_scale=tuple(value * 2.0 for value in actor.feature_scale))
    with pytest.raises(ValueError, match="violates its contract"):
        replace(actor, direct_joint_torque_output=True)


def test_existing_finish_backend_accepts_ready_extended_intent_actor() -> None:
    candidate = RoleOptionBackendCandidate(
        backend=RoleOptionBackend.CONTEXTUAL_FINISH_TARGET,
        option=PhysicalSoccerOption.SHOOT,
        artifact_hash=_hash("actor"),
        evidence_hash=_hash("evidence"),
        distinct_context_count=8,
        distinct_trajectory_count=8,
        strict_replay=True,
        holdout_passed=True,
        parent_retention_passed=True,
    )

    assert candidate.evidence_ready


def test_s206_search_and_holdout_plan_is_fixed_and_disjoint() -> None:
    config = ContextualFinishIntentPortfolioConfig()
    coarse = config.coarse_plan.local_candidates(config.repair_warm_start)
    refinement = config.refinement_plan.local_candidates(config.repair_warm_start)
    holdouts = (*config.success_holdouts, *config.rejection_holdouts)

    assert len(coarse) == 65
    assert len(refinement) == 33
    assert config.coarse_plan.plan_hash != config.refinement_plan.plan_hash
    assert len({candidate.candidate_hash for candidate in coarse}) == 65
    assert len({case.case_id for case in holdouts}) == 4
    assert (
        len({(case.receiver_phase_start_sec, case.receiver_lateral_lane_m) for case in holdouts})
        == 4
    )
    assert all(candidate.activation_ceiling == "SIM_ONLY" for candidate in coarse)


def test_s206_config_round_trip_and_authority_are_fail_closed() -> None:
    config = ContextualFinishIntentPortfolioConfig()

    restored = _config_from_dict(asdict(config))

    assert restored == config
    assert restored.config_hash == config.config_hash
    with pytest.raises(ValueError, match="portfolio config is invalid"):
        replace(config, hardware_authorized=True)
    with pytest.raises(ValueError, match="portfolio config is invalid"):
        replace(config, coarse_candidate_count=128)


def test_current_s206_evidence_reconstructs_when_mounted() -> None:
    root = os.environ.get("ROSCLAW_SOCCER_EVIDENCE")
    if root is None:
        pytest.skip("external soccer evidence is not mounted")
    path = (
        Path(root)
        / "s206-contextual-finish-intent-portfolio-v1"
        / "contextual-finish-intent-portfolio.json"
    )
    if not path.is_file():
        pytest.skip("S206 evidence is not mounted")

    report = validate_contextual_finish_intent_portfolio(path)

    assert report["status"] == "PASS_CONTEXTUAL_FINISH_INTENT_PORTFOLIO"
    assert report["gates"]["fresh_success_holdouts_passed"]
    assert report["gates"]["fresh_ood_holdouts_rejected"]

    video_manifest = path.parent / "s206-contextual-finish-growth.json"
    if video_manifest.is_file():
        video = validate_contextual_finish_intent_video_manifest(video_manifest)
        assert video["source_report_hash"] == report["report_hash"]
        assert video["pixels_used_for_scoring"] is False
