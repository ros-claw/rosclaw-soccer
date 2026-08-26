from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.media.lateral_athlete_video import (
    validate_lateral_athlete_video_manifest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.lateral_athlete_cpu_exam import LateralAthleteExamConfig
from rosclaw_soccer.training.lateral_athlete_expert import (
    LateralAthleteExpertConfig,
    build_lateral_athlete_actor,
    capture_point_teacher_numpy,
    decode_lateral_athlete_command,
    lateral_athlete_features_numpy,
    load_lateral_athlete_expert,
)


def _features() -> np.ndarray:
    return lateral_athlete_features_numpy(
        lateral_error_m=np.asarray((-1.0, 1.0)),
        lateral_velocity_mps=np.asarray((0.2, -0.2)),
        time_remaining_sec=np.asarray((5.0, 5.0)),
        pelvis_height_m=np.asarray((0.793, 0.793)),
        upright_projection=np.asarray((1.0, 1.0)),
        root_angular_velocity_rad_s=np.asarray(((0.3, 0.1, -0.2), (-0.3, 0.1, 0.2))),
        previous_command=np.asarray((0.4, -0.4)),
    )


def test_lateral_athlete_config_is_sim_only_and_bounded() -> None:
    config = LateralAthleteExpertConfig()
    assert config.activation_ceiling == "SIM_ONLY"
    assert not config.hardware_authorized
    assert config.lateral_speed_limit_mps <= 0.40
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
    with pytest.raises(ValueError, match="speed"):
        replace(config, lateral_speed_limit_mps=0.41)
    with pytest.raises(ValueError, match="mirrored pairs"):
        replace(config, samples_per_epoch=4_097)


def test_features_and_teacher_are_bilateral() -> None:
    features = _features()
    assert features.shape == (2, 11)
    np.testing.assert_allclose(features[0, (0, 1, 7, 9, 10)], -features[1, (0, 1, 7, 9, 10)])
    np.testing.assert_allclose(features[0, (2, 3, 4, 5, 6, 8)], features[1, (2, 3, 4, 5, 6, 8)])
    config = LateralAthleteExpertConfig()
    command = capture_point_teacher_numpy(
        lateral_error_m=np.asarray((-1.0, 1.0)),
        lateral_velocity_mps=np.asarray((0.2, -0.2)),
        config=config,
    )
    assert command[0] == pytest.approx(-command[1])
    assert np.all(np.abs(command) <= 1.0)


def test_neural_decoder_enforces_exact_odd_symmetry() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    torch.manual_seed(7)
    model = build_lateral_athlete_actor(torch, nn, hidden_size=32)
    features = torch.as_tensor(_features(), dtype=torch.float32)
    command = decode_lateral_athlete_command(torch=torch, model=model, features=features)
    assert command.shape == (2,)
    assert command[0].item() == pytest.approx(-command[1].item(), abs=1.0e-7)
    with pytest.raises(ValueError, match="shape"):
        decode_lateral_athlete_command(torch=torch, model=model, features=torch.zeros((1, 10)))


def test_checkpoint_is_bound_to_locomotion_prior(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    locomotion = tmp_path / "locomotion.pt"
    locomotion.write_bytes(b"qualified-locomotion")
    model = build_lateral_athlete_actor(torch, nn, hidden_size=32)
    checkpoint = {
        "schema_version": "rosclaw_soccer.lateral_athlete_expert.v1",
        "model_state_dict": model.state_dict(),
        "input_size": 11,
        "output_size": 1,
        "hidden_size": 32,
        "output_representation": "BOUNDED_LOCAL_LATERAL_VELOCITY_COMMAND",
        "symmetry_enforcement": "EXACT_ODD_SAGITTAL_EQUIVARIANCE_V1",
        "locomotion_policy_hash": "sha256:"
        + __import__("hashlib").sha256(locomotion.read_bytes()).hexdigest(),
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "commercial_use_allowed": False,
    }
    path = tmp_path / "expert.pt"
    torch.save(checkpoint, path)
    loaded, payload = load_lateral_athlete_expert(
        checkpoint_path=path, locomotion_policy_path=locomotion, device=torch.device("cpu")
    )
    assert loaded is not None
    assert payload["activation_ceiling"] == "SIM_ONLY"
    locomotion.write_bytes(b"different")
    with pytest.raises(ValueError, match="boundary"):
        load_lateral_athlete_expert(
            checkpoint_path=path,
            locomotion_policy_path=locomotion,
            device=torch.device("cpu"),
        )


def test_cpu_exam_requires_qualified_timing_and_has_successor_gate() -> None:
    config = LateralAthleteExamConfig()
    assert config.control_dt_sec / config.physics_substeps == pytest.approx(0.002)
    assert config.maximum_endpoint_error_m == pytest.approx(0.10)
    assert config.maximum_successor_lateral_speed_mps < config.settle_speed_mps
    assert config.maximum_successor_root_angular_speed_rad_s < 0.25
    with pytest.raises(ValueError, match="2 ms physics"):
        replace(config, physics_substeps=8)
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)


def test_lateral_athlete_video_manifest_is_content_bound(tmp_path: Path) -> None:
    source = tmp_path / "exam.json"
    source.write_text("evidence\n", encoding="utf-8")
    video = tmp_path / "reel.mp4"
    video.write_bytes(b"video")
    manifest = {
        "schema_version": "rosclaw_soccer.lateral_athlete_video.v1",
        "video_path": str(video),
        "video_hash": hash_bytes(video.read_bytes()),
        "source_files": {str(source): hash_bytes(source.read_bytes())},
        "claim": "S101_BILATERAL_LATERAL_ATHLETE_EXPERT_CPU_MUJOCO_REPLAY",
        "source_evidence_passed": True,
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
    }
    manifest["manifest_hash"] = hash_json(manifest)
    path = tmp_path / "reel.json"
    path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    assert validate_lateral_athlete_video_manifest(path)["source_evidence_passed"]
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        validate_lateral_athlete_video_manifest(path)
