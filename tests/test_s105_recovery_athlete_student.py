from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.skills.team.shared_world import G1GoalkeeperConfig
from rosclaw_soccer.training.recovery_athlete_integration_exam import (
    RecoveryAthleteIntegrationConfig,
    _recovery_command_metrics,
)
from rosclaw_soccer.training.recovery_athlete_student import (
    RecoveryAthleteStudentConfig,
    build_recovery_athlete_actor,
    decode_recovery_athlete_command,
    load_recovery_athlete_student,
    recovery_athlete_features_numpy,
    recovery_teacher_numpy,
)
from rosclaw_soccer.training.save_to_ready_successor import SaveToReadySuccessorConfig


def _mirrored_features() -> np.ndarray:
    return recovery_athlete_features_numpy(
        depth_error_m=np.asarray((0.30, 0.30)),
        lateral_position_m=np.asarray((0.25, -0.25)),
        yaw_error_rad=np.asarray((0.20, -0.20)),
        root_velocity=np.asarray(
            (
                (0.10, 0.20, -0.05, 0.0, 0.0, 0.30),
                (0.10, -0.20, -0.05, 0.0, 0.0, -0.30),
            )
        ),
        pelvis_height_m=np.asarray((0.76, 0.76)),
        upright_projection=np.asarray((0.98, 0.98)),
        foot_contact=np.asarray(((True, False), (False, True))),
        elapsed_since_contact_sec=np.asarray((4.0, 4.0)),
    )


def test_recovery_student_contract_is_low_authority_sim_only() -> None:
    config = RecoveryAthleteStudentConfig()
    assert config.required_world_size == 4
    assert config.activation_ceiling == "SIM_ONLY"
    assert not config.hardware_authorized
    assert config.depth_speed_limit_mps <= 0.30
    assert config.lateral_speed_limit_mps <= 0.20
    assert config.yaw_rate_limit_rad_s <= 0.20
    with pytest.raises(ValueError, match="four GPUs"):
        replace(config, required_world_size=1)
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
    with pytest.raises(ValueError, match="authority"):
        replace(config, depth_speed_limit_mps=0.31)


def test_features_and_teacher_have_exact_sagittal_mapping() -> None:
    features = _mirrored_features()
    feature_sign = np.asarray(
        (1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0)
    )
    np.testing.assert_allclose(features[1], features[0] * feature_sign, atol=1.0e-7)
    target = recovery_teacher_numpy(
        depth_error_m=np.asarray((0.30, 0.30)),
        lateral_position_m=np.asarray((0.25, -0.25)),
        yaw_error_rad=np.asarray((0.20, -0.20)),
        config=RecoveryAthleteStudentConfig(),
    )
    np.testing.assert_allclose(target[1], target[0] * (1.0, -1.0, -1.0))
    assert np.max(np.abs(target)) <= 1.0


def test_decoder_enforces_exact_vector_equivariance() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    torch.manual_seed(105)
    model = build_recovery_athlete_actor(torch, nn, hidden_size=32)
    features = torch.as_tensor(_mirrored_features(), dtype=torch.float32)
    decoded = decode_recovery_athlete_command(torch=torch, model=model, features=features)
    torch.testing.assert_close(
        decoded[1],
        decoded[0] * torch.tensor((1.0, -1.0, -1.0)),
        rtol=0.0,
        atol=1.0e-7,
    )
    assert torch.max(torch.abs(decoded)).item() <= 1.0
    with pytest.raises(ValueError, match="shape"):
        decode_recovery_athlete_command(torch=torch, model=model, features=torch.zeros((1, 12)))


def test_checkpoint_is_bound_to_frozen_locomotion_prior(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    locomotion = tmp_path / "policy.pt"
    locomotion.write_bytes(b"qualified-locomotion")
    checkpoint = tmp_path / "student.pt"
    model = build_recovery_athlete_actor(torch, nn, hidden_size=32)
    torch.save(
        {
            "schema_version": "rosclaw_soccer.recovery_athlete_student.v1",
            "model_state_dict": model.state_dict(),
            "input_size": 13,
            "output_size": 3,
            "hidden_size": 32,
            "feature_mirror_sign": [
                1.0,
                -1.0,
                -1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
            ],
            "output_mirror_sign": [1.0, -1.0, -1.0],
            "output_scale": np.asarray((0.30, 0.12, 0.12), dtype=np.float32).tolist(),
            "output_representation": "BOUNDED_WORLD_DEPTH_LATERAL_YAW_LOCOMOTION_COMMAND",
            "symmetry_enforcement": "EXACT_SAGITTAL_EQUIVARIANCE_V1",
            "locomotion_policy_hash": hash_bytes(locomotion.read_bytes()),
            "activation_ceiling": "SIM_ONLY",
            "hardware_authorized": False,
            "commercial_use_allowed": False,
        },
        checkpoint,
    )
    loaded, contract = load_recovery_athlete_student(
        checkpoint_path=checkpoint,
        locomotion_policy_path=locomotion,
        device=torch.device("cpu"),
    )
    assert loaded.training is False
    assert contract["hardware_authorized"] is False
    locomotion.write_bytes(b"changed")
    with pytest.raises(ValueError, match="boundary"):
        load_recovery_athlete_student(
            checkpoint_path=checkpoint,
            locomotion_policy_path=locomotion,
            device=torch.device("cpu"),
        )


def test_shared_world_recovery_actor_contract_fails_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "student.pt"
    exam = tmp_path / "cpu-exam.json"
    checkpoint.write_bytes(b"checkpoint")
    exam.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint, exam"):
        replace(G1GoalkeeperConfig(), recovery_athlete_checkpoint_path=checkpoint)
    with pytest.raises(ValueError, match="ready recovery"):
        replace(
            G1GoalkeeperConfig(),
            recovery_athlete_checkpoint_path=checkpoint,
            recovery_athlete_exam_path=exam,
            recovery_athlete_blend=1.0,
        )
    active = replace(
        G1GoalkeeperConfig(),
        post_contact_stabilization_enabled=True,
        post_contact_ready_recovery_enabled=True,
        recovery_athlete_checkpoint_path=checkpoint,
        recovery_athlete_exam_path=exam,
        recovery_athlete_blend=1.0,
    )
    assert active.schema_version == "rosclaw_soccer.g1_goalkeeper_config.v31"
    assert replace(active, recovery_athlete_blend=0.5).recovery_athlete_blend == 0.5
    with pytest.raises(ValueError, match="blend"):
        replace(active, recovery_athlete_blend=1.1)


def test_integration_metrics_measure_only_recovery_segments() -> None:
    config = SaveToReadySuccessorConfig()
    time = np.arange(0.02, 25.01, 0.02)
    command = np.zeros_like(time)
    recovery = (time >= 10.2) & (time < 18.0)
    command[recovery] = np.linspace(-0.08, 0.0, np.count_nonzero(recovery))
    probe = (time >= 18.0) & (time < 18.8)
    command[probe] = 0.14
    active = recovery | (time >= 18.9)
    learned = np.zeros((time.size, 3), dtype=np.float64)
    learned[active, 1] = command[active]
    metrics = _recovery_command_metrics(
        {
            "time": time,
            "goalkeeper_command_mps": command,
            "goalkeeper_recovery_athlete_active": active,
            "goalkeeper_recovery_athlete_world_command": learned,
        },
        contact_time_sec=8.0,
        config=config,
    )
    assert metrics["recovery_actor_active_frame_count"] == int(np.count_nonzero(active))
    assert 0.07 <= metrics["lateral_command_total_variation_mps"] <= 0.09
    assert metrics["lateral_command_peak_step_mps"] < 0.01


def test_integration_contract_remains_sim_only() -> None:
    config = RecoveryAthleteIntegrationConfig()
    assert config.candidate_blend == 1.0
    assert config.activation_ceiling == "SIM_ONLY"
    assert not config.hardware_authorized
    with pytest.raises(ValueError, match="full candidate"):
        replace(config, candidate_blend=0.5)
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
