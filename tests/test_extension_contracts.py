from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from rosclaw_soccer.data.sources import (
    G1_RETARGETED_MOTIONS_SOURCE,
    MOTIONDECODE_SOURCE,
    OMNICONTACT_SOURCE,
    SOCCER_MOTION_SOURCES,
)
from rosclaw_soccer.growth.adapter import SOCCER_GROWTH_ADAPTER
from rosclaw_soccer.sim.tasks import SOCCER_TASK_PROVIDER


def _core_contract_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def test_extension_objects_import_without_core_runtime_side_effects() -> None:
    assert SOCCER_GROWTH_ADAPTER.adapter_id == "soccer.growth"
    assert SOCCER_GROWTH_ADAPTER.skill_ids == (
        "soccer.first_touch",
        "soccer.free_kick",
        "soccer.passing",
        "soccer.shooting",
        "soccer.goalkeeping",
    )
    assert SOCCER_TASK_PROVIDER.provider_id == "soccer.academy"
    assert SOCCER_TASK_PROVIDER.task_ids == (
        "soccer.age04_regulation",
        "soccer.first_touch",
        "soccer.continuous_match",
        "soccer.two_vs_one_decision",
        "soccer.three_role_league",
    )
    with pytest.raises(KeyError, match="unknown soccer task"):
        SOCCER_TASK_PROVIDER.task_spec("soccer.unknown")


def test_motion_source_classifies_relative_names_only() -> None:
    assert OMNICONTACT_SOURCE.classify_file(
        "OmniContact",
        "npz/soccer/train/clip_with_contact.npz",
    ) == (
        "contact.annotation",
        "motion.reference",
        "split.train",
    )
    assert MOTIONDECODE_SOURCE.classify_file("MotionDecode", "README.md") == ("license.material",)
    assert G1_RETARGETED_MOTIONS_SOURCE.classify_file(
        "g1-retargeted-motions",
        "ACCAD_retargeted/C3_Run_stageii.pkl",
    ) == ("motion.reference",)
    assert MOTIONDECODE_SOURCE.classify_file("unrelated", "train/data.csv") == ()
    with pytest.raises(ValueError, match="safe relative path"):
        MOTIONDECODE_SOURCE.classify_file("MotionDecode", "../secret")
    with pytest.raises(ValueError, match="safe relative path"):
        MOTIONDECODE_SOURCE.classify_file("MotionDecode", "train/bad\nname.csv")


def test_motion_sources_keep_upstream_provenance_separate() -> None:
    assert len({source.source_uri for source in SOCCER_MOTION_SOURCES}) == 3
    assert len({source.revision for source in SOCCER_MOTION_SOURCES}) == 3
    assert len({source.dataset_id for source in SOCCER_MOTION_SOURCES}) == 3
    assert all(len(source.revision) == 40 for source in SOCCER_MOTION_SOURCES)


@pytest.mark.skipif(
    not _core_contract_available("rosclaw.growth.registry"),
    reason="requires stacked ROSClaw Core PR #307",
)
def test_growth_entry_point_normalizes_fail_closed_core_experience() -> None:
    from rosclaw.growth import GrowthExtensionRegistry

    registry = GrowthExtensionRegistry()
    registry.register_adapter(SOCCER_GROWTH_ADAPTER)
    adapter = registry.adapter("soccer.growth")
    payload = {
        "segment_id": "segment.free_kick.001",
        "episode_id": "episode.age04.001",
        "skill_id": "soccer.free_kick",
        "phase": "post_contact_recovery",
        "start_time_sec": 3.0,
        "end_time_sec": 4.0,
        "body_hash": _hash("body"),
        "regime_hash": _hash("regime"),
        "source_evidence_level": "e1_physics_replay",
        "lineage": {
            "source_artifact_hash": _hash("artifact"),
            "source_event_hashes": [_hash("event")],
            "transform_hash": _hash("transform"),
            "clock_id": "mujoco.sim_time",
            "maximum_skew_sec": 0.001,
            "observed_skew_sec": 0.0,
            "synchronization_receipt_hash": None,
        },
        "base_policy_version": "age04.base.v1",
        "residual_policy_version": "age04.contact.v1",
        "state_start_hash": _hash("state-start"),
        "observation_sequence_hash": _hash("observations"),
        "self_state_hash": _hash("self"),
        "world_state_hash": _hash("world"),
        "action": {
            "commanded_action_hash": _hash("commanded"),
            "executed_action_hash": _hash("executed"),
            "safety_projected_action_hash": _hash("executed"),
            "policy_version": "age04.contact.v1",
            "controller_hash": _hash("controller"),
            "projection_applied": False,
        },
        "reward_vector": {
            "soccer.contact": 1.0,
            "soccer.precision": 0.8,
        },
        "cost_vector": {"soccer.balance": 0.2},
        "terminal_state_hash": _hash("terminal"),
        "advantage_label": "unsafe_negative",
        "label_confidence": 1.0,
    }

    segment = adapter.normalize_experience(payload)

    assert segment.failure_signature is not None
    assert segment.failure_signature.primary_type == "soccer.post_contact_instability"
    assert segment.failure_signature.recommended_learner_ids == (
        "residual_sac",
        "system_identification",
    )
    assert segment.to_dict()["hardware_authorized"] is False


@pytest.mark.skipif(
    not _core_contract_available("rosclaw.dataset.registry"),
    reason="requires stacked ROSClaw Core PR #310",
)
def test_dataset_source_satisfies_core_minimum_authority_protocol() -> None:
    from rosclaw.dataset import DatasetSourceRegistry

    registry = DatasetSourceRegistry()
    for source in SOCCER_MOTION_SOURCES:
        registry.register_source(source)

    assert registry.source_ids == (
        "soccer.g1-retargeted-motions",
        "soccer.motiondecode",
        "soccer.omnicontact",
    )
    assert tuple(descriptor.dataset_ids for descriptor in registry.descriptors) == (
        ("g1-retargeted-motions",),
        ("MotionDecode",),
        ("OmniContact",),
    )


@pytest.mark.skipif(
    not _core_contract_available("rosclaw.simforge.registry"),
    reason="requires stacked ROSClaw Core PR #308",
)
def test_simforge_provider_registers_descriptions_without_running_physics() -> None:
    from rosclaw.simforge.registry import SimForgeTaskRegistry

    registry = SimForgeTaskRegistry()
    registry.register(SOCCER_TASK_PROVIDER)

    age04 = registry.task_spec("soccer.age04_regulation")
    age05 = registry.task_spec("soccer.first_touch")
    continuous = registry.task_spec("soccer.continuous_match")
    two_vs_one = registry.task_spec("soccer.two_vs_one_decision")
    league = registry.task_spec("soccer.three_role_league")
    assert age04.evidence_requirements.minimum_seeds == 8
    assert age05.evidence_requirements.minimum_seeds == 20
    assert age05.scenario_distribution_ref == "soccer://age05/first-touch-v1"
    assert continuous.success_spec[0] == ("continuous_episode_sec.min", 60.0)
    assert continuous.success_spec[1] == ("terminate_on_goal", False)
    assert two_vs_one.evidence_requirements.minimum_seeds == 32
    assert two_vs_one.candidate_allowed_paths == ("/tactical_policy",)
    assert league.evidence_requirements.minimum_seeds == 8
    assert league.candidate_allowed_paths == (
        "/roles/passer/policy",
        "/roles/shooter/policy",
        "/roles/goalkeeper/policy",
    )


def test_migrated_growth_modules_have_no_checkout_absolute_paths() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "rosclaw_soccer" / "growth"
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "/code/rosclaw" not in text
