from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import asdict

import pytest


def _legacy_growth_available() -> bool:
    try:
        return (
            importlib.util.find_spec("rosclaw.growth.ballistic_contact_impulse_actor") is not None
        )
    except ModuleNotFoundError:
        return False


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


@pytest.mark.skipif(
    not _legacy_growth_available(),
    reason="legacy #271 implementation is needed only for extraction parity",
)
def test_migrated_actor_and_residual_contracts_preserve_legacy_artifacts() -> None:
    from rosclaw.growth.approach_strike_residual import (
        G1ApproachStrikeResidualConfig as LegacyApproachConfig,
    )
    from rosclaw.growth.ballistic_contact_impulse_actor import (
        G1BallisticContactImpulseActor as LegacyActor,
    )
    from rosclaw.growth.ballistic_contact_impulse_actor import (
        g1_ballistic_contact_impulse_context_hash as legacy_context_hash,
    )
    from rosclaw.growth.phase_conditioned_residual import (
        G1PhaseConditionedResidualConfig as LegacyPhaseConfig,
    )

    from rosclaw_soccer.growth.approach_strike_residual import (
        G1ApproachStrikeResidualConfig as MigratedApproachConfig,
    )
    from rosclaw_soccer.growth.ballistic_contact_impulse_actor import (
        G1BallisticContactImpulseActor as MigratedActor,
    )
    from rosclaw_soccer.growth.ballistic_contact_impulse_actor import (
        g1_ballistic_contact_impulse_context_hash as migrated_context_hash,
    )
    from rosclaw_soccer.growth.phase_conditioned_residual import (
        G1PhaseConditionedResidualConfig as MigratedPhaseConfig,
    )

    hashes = tuple(_hash(f"probe-{index}") for index in range(8))
    actor_values = {
        "body_hash": _hash("body"),
        "implementation_hash": _hash("implementation"),
        "experiment_context_hash": _hash("context"),
        "source_evidence_hashes": hashes,
        "selected_evidence_hash": hashes[0],
        "selected_goal_plane_target_error_m": 0.05,
        "precision_success_count": 2,
        "rejected_probe_count": 6,
        "task_space_actor_weight_matrix": ((400.0, -40.0, 0.0), (350.0, 0.0, -50.0)),
        "maximum_lateral_force_n": 250.0,
        "maximum_vertical_force_n": 250.0,
        "maximum_foot_ball_distance_m": 0.18,
        "start_policy_frame": 230,
        "end_policy_frame": 335,
        "foot_strike_point_offset_m": (0.13, 0.0, -0.025),
        "qualified_error_max_m": 0.10,
    }
    legacy_actor = LegacyActor(**actor_values)
    migrated_actor = MigratedActor(**actor_values)
    context = {
        "flow_config": {"schema_version": "flow.v1", "gain": 0.2},
        "goal_spec": {"target_y_m": 1.0, "target_z_m": 2.0},
        "runup_config": {"speed": 1.2},
        "sonic_runup_config": None,
        "approach_strike_candidate_hash": _hash("candidate"),
    }
    phase_delta = [0.0] * 29
    phase_delta[3] = -1.2

    assert migrated_actor.to_dict() == legacy_actor.to_dict()
    assert migrated_actor.actor_hash == legacy_actor.actor_hash
    assert migrated_context_hash(**context) == legacy_context_hash(**context)
    assert MigratedApproachConfig().config_hash == LegacyApproachConfig().config_hash
    assert asdict(
        MigratedPhaseConfig(
            event_phase_id=4,
            joint_delta_nm=tuple(phase_delta),
        )
    ) == asdict(
        LegacyPhaseConfig(
            event_phase_id=4,
            joint_delta_nm=tuple(phase_delta),
        )
    )
