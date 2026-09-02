from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw_soccer.growth.tactical_2v1 import TacticalAction
from rosclaw_soccer.growth.tactical_2v1_actor import (
    fit_two_vs_one_tactical_actor,
    load_two_vs_one_tactical_actor,
    save_two_vs_one_tactical_actor,
)
from rosclaw_soccer.media.tactical_2v1_growth_video import (
    validate_two_vs_one_growth_video_manifest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.tactical_2v1_growth import (
    TwoVsOneRetentionManifest,
    collect_two_vs_one_acquisition,
    run_two_vs_one_growth_round,
    validate_two_vs_one_growth_stage,
)
from rosclaw_soccer.training.tactical_2v1_physics import (
    FrozenTacticalSkillBundle,
    TwoVsOnePhysicsConfig,
    TwoVsOnePhysicsScenario,
    matched_two_vs_one_decision,
    persist_matched_two_vs_one_decision,
    simulate_two_vs_one_physics,
)


def _hash(label: str) -> str:
    return str(hash_json({"fixture": label}))


def _bundle() -> FrozenTacticalSkillBundle:
    return FrozenTacticalSkillBundle(
        body_hash=_hash("body"),
        athlete_foundation_hash=_hash("foundation"),
        first_touch_actor_hash=_hash("first-touch"),
        pass_skill_hash=_hash("pass"),
        shoot_skill_hash=_hash("shoot"),
    )


def _scenario(index: int, commitment: float, lateral: float) -> TwoVsOnePhysicsScenario:
    return TwoVsOnePhysicsScenario(
        scenario_id=f"test.s119.{index}",
        seed=119 + index,
        defender_commitment=commitment,
        finisher_lateral_m=lateral,
    )


def _training_scenarios() -> tuple[TwoVsOnePhysicsScenario, ...]:
    return tuple(
        _scenario(index, commitment, lateral)
        for index, (commitment, lateral) in enumerate(
            (
                (0.0, 1.4),
                (0.2, -1.5),
                (0.3, 1.6),
                (0.4, -1.7),
                (0.72, 1.4),
                (0.8, -1.5),
                (0.9, 1.6),
                (1.0, -1.7),
            )
        )
    )


def test_two_vs_one_contracts_reject_unsafe_or_unsealed_values() -> None:
    with pytest.raises(ValueError, match="hierarchy or safety"):
        TwoVsOnePhysicsConfig(hardware_authorized=True)
    with pytest.raises(ValueError, match="curriculum"):
        _scenario(0, 1.1, 1.4)
    with pytest.raises(ValueError, match="sealed"):
        TwoVsOneRetentionManifest(
            scenarios=tuple(_scenario(index, 0.2, 1.4) for index in range(16)),
            training_access_allowed=True,
        )


def test_physics_changes_the_best_action_with_defender_commitment() -> None:
    bundle = _bundle()
    cover = _scenario(0, 0.0, 1.4)
    press = _scenario(1, 1.0, 1.4)
    cover_pass, _ = simulate_two_vs_one_physics(
        scenario=cover,
        action=TacticalAction.PASS,
        skill_bundle=bundle,
    )
    cover_shoot, _ = simulate_two_vs_one_physics(
        scenario=cover,
        action=TacticalAction.SHOOT,
        skill_bundle=bundle,
    )
    press_pass, _ = simulate_two_vs_one_physics(
        scenario=press,
        action=TacticalAction.PASS,
        skill_bundle=bundle,
    )
    press_shoot, _ = simulate_two_vs_one_physics(
        scenario=press,
        action=TacticalAction.SHOOT,
        skill_bundle=bundle,
    )
    assert cover_pass.intercepted is True
    assert cover_shoot.goal_scored is True
    assert press_pass.pass_completed is True
    assert press_shoot.intercepted is True
    assert all(result.safe for result in (cover_pass, cover_shoot, press_pass, press_shoot))


def test_matched_pass_has_positive_physical_counterfactual_credit() -> None:
    evidence, primary, ablated, primary_trace, ablated_trace = matched_two_vs_one_decision(
        scenario=_scenario(0, 1.0, 1.4),
        action=TacticalAction.PASS,
        policy_hash=_hash("policy"),
        skill_bundle=_bundle(),
    )
    assert primary.pass_completed is True
    assert ablated.pass_completed is False
    assert evidence.rollout.difference_reward > 0.0
    assert evidence.promotion_eligible is True
    assert evidence.rollout.trajectory_hash != evidence.rollout.ablated_trajectory_hash
    assert primary_trace["focal_agent_present"].all()
    assert not ablated_trace["focal_agent_present"].any()


def test_actor_learns_both_actions_and_fails_closed_on_tamper(tmp_path: Path) -> None:
    bundle = _bundle()
    evidence, _ = collect_two_vs_one_acquisition(
        scenarios=_training_scenarios(),
        skill_bundle=bundle,
    )
    actor = fit_two_vs_one_tactical_actor(evidence)
    assert (
        actor.decide(
            _scenario(20, 0.1, 1.55).state(
                skill_bundle=bundle,
                config=TwoVsOnePhysicsConfig(),
            )
        ).action
        is TacticalAction.SHOOT
    )
    assert (
        actor.decide(
            _scenario(21, 0.9, -1.55).state(
                skill_bundle=bundle,
                config=TwoVsOnePhysicsConfig(),
            )
        ).action
        is TacticalAction.PASS
    )
    ood_state = replace(
        _scenario(22, 0.5, 1.55).state(
            skill_bundle=bundle,
            config=TwoVsOnePhysicsConfig(),
        ),
        carrier_pressure=1.0,
    )
    assert actor.decide(ood_state).action is TacticalAction.HOLD

    path = tmp_path / "actor.json"
    save_two_vs_one_tactical_actor(actor, path)
    assert load_two_vs_one_tactical_actor(path) == actor
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["action_weights"][0][0] += 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_two_vs_one_tactical_actor(path)


def test_persisted_decision_binds_both_trajectories(tmp_path: Path) -> None:
    report = persist_matched_two_vs_one_decision(
        output_dir=tmp_path / "evidence",
        source_checkout=Path(__file__).parents[1],
        scenario=_scenario(0, 1.0, 1.4),
        action=TacticalAction.PASS,
        policy_hash=_hash("policy"),
        skill_bundle=_bundle(),
    )
    assert report["status"] == "PASS_MATCHED_TACTICAL_ROLLOUT"
    assert report["decision_evidence"]["rollout"]["difference_reward"] > 0.0
    assert report["evidence_boundary"]["g1_whole_body_rollout_claimed"] is False
    assert (tmp_path / "evidence/primary-trajectory.npz").is_file()
    assert (tmp_path / "evidence/ablated-trajectory.npz").is_file()


def test_complete_growth_stage_is_validated_and_tamper_evident(tmp_path: Path) -> None:
    retention = TwoVsOneRetentionManifest(
        scenarios=tuple(
            _scenario(
                100 + index,
                0.2 if index < 8 else 0.9,
                (1.45 if index % 2 else -1.65),
            )
            for index in range(16)
        )
    )
    output = tmp_path / "growth"
    stage = run_two_vs_one_growth_round(
        output_dir=output,
        source_checkout=Path(__file__).parents[1],
        skill_bundle=_bundle(),
        retention_manifest=retention,
    )
    assert stage["status"] == "PASS_BOUNDED_TACTICAL_RETENTION"
    validated = validate_two_vs_one_growth_stage(
        output / "stage-summary.json",
        source_checkout=Path(__file__).parents[1],
    )
    assert validated["retention_metrics"]["task_success_rate"] == 1.0
    artifact = output / "retention/case-000/selected-primary.npz"
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        validate_two_vs_one_growth_stage(output / "stage-summary.json")


def test_two_vs_one_video_manifest_is_content_bound(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture-video")
    manifest = {
        "schema_version": "rosclaw_soccer.two_vs_one_growth_video.v1",
        "claim": "BOUNDED_TWO_VS_ONE_TACTICAL_GROWTH_VISUALIZATION",
        "source_stage_hash": _hash("stage"),
        "source_stage_passed": True,
        "source_files": {str(source): hash_bytes(source.read_bytes())},
        "video_path": str(video),
        "video_hash": hash_bytes(video.read_bytes()),
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "frame_count": 420,
        "duration_sec": 14.0,
        "visualization_only": True,
        "tactical_plane_only": True,
        "g1_bodies_rendered": False,
        "pixels_used_for_scoring": False,
        "commercial_use_allowed": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    manifest["manifest_hash"] = hash_json(manifest)
    path = tmp_path / "video.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_two_vs_one_growth_video_manifest(path)["video_hash"] == manifest["video_hash"]
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="video changed"):
        validate_two_vs_one_growth_video_manifest(path)
