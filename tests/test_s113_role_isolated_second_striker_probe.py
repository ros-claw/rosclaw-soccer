from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.training.role_isolated_second_striker_probe import (
    RoleIsolatedSecondStrikerProbeConfig,
    _candidate_diagnostics,
    _derive_probe_gates,
    validate_role_isolated_second_striker_probe,
)

_EVIDENCE = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "s113-role-isolated-target-actor-control-v3/evidence.json"
)


def _diagnostic_trajectory(*, candidate_selected: bool) -> dict[str, np.ndarray]:
    return {
        "second_striker_ballistic_actor_target_conditioned": np.asarray(
            (False, True, True), dtype=np.bool_
        ),
        "second_striker_ballistic_actor_launch_envelope_supported": np.asarray(
            (False, candidate_selected, candidate_selected), dtype=np.bool_
        ),
        "second_striker_ballistic_actor_candidate_selected": np.asarray(
            (False, candidate_selected, candidate_selected), dtype=np.bool_
        ),
        "second_striker_ballistic_actor_active": np.asarray((False, True, True), dtype=np.bool_),
        "second_striker_ballistic_actor_desired_launch_velocity_yz_mps": np.asarray(
            ((np.nan, np.nan), (-0.14, 3.14), (-0.14, 3.14)), dtype=np.float64
        ),
    }


def test_role_isolated_probe_contract_rejects_hardware_and_bad_ball_physics() -> None:
    config = RoleIsolatedSecondStrikerProbeConfig()
    assert config.activation_ceiling == "SIM_ONLY"
    assert config.replay_count == 2
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
    with pytest.raises(ValueError, match="mass"):
        replace(config, second_ball_mass_kg=0.8)
    with pytest.raises(ValueError, match="friction"):
        replace(config, second_ball_ground_friction=0.01)


def test_candidate_diagnostics_separate_parent_fallback_from_plasticity() -> None:
    fallback = _candidate_diagnostics(_diagnostic_trajectory(candidate_selected=False))
    plastic = _candidate_diagnostics(_diagnostic_trajectory(candidate_selected=True))

    assert fallback["conditioned_frame_count"] == 2
    assert fallback["supported_frame_count"] == 0
    assert fallback["candidate_selected_frame_count"] == 0
    assert fallback["frozen_parent_selected_frame_count"] == 2
    assert plastic["candidate_selected_frame_count"] == 2
    assert plastic["frozen_parent_selected_frame_count"] == 0


def test_gate_derivation_does_not_confuse_retention_with_growth() -> None:
    diagnostics = _candidate_diagnostics(_diagnostic_trajectory(candidate_selected=False))
    replay = {
        "result": {"finite_state": True},
        "evaluation": {
            "passed": True,
            "first_takeoff_exam": {"passed": True},
            "gates": {"whole_world_safety": True},
        },
        "candidate_diagnostics": diagnostics,
        "trajectory_digest": "sha256:digest",
        "trajectory_hash": "sha256:trajectory",
    }

    evidence, plasticity = _derive_probe_gates([dict(replay), dict(replay)])

    assert all(evidence.values())
    assert plasticity["complete_chain_passed"] is True
    assert plasticity["candidate_envelope_supported"] is False
    assert plasticity["candidate_selected"] is False


def test_external_s113_evidence_if_available() -> None:
    if not _EVIDENCE.is_file():
        pytest.skip("external S113 role-isolated evidence is not present")
    report = validate_role_isolated_second_striker_probe(_EVIDENCE)
    assert report["evidence_passed"] is True
    assert report["candidate_promoted"] is False
    assert report["candidate_status"] == "REJECTED_NO_SUPPORTED_PLASTICITY"
