from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.growth.mosaic_g1_contract import (
    MOSAIC_G1_ISAACLAB_JOINT_NAMES,
    MOSAIC_G1_SEMANTIC_CONTRACT_HASH,
)
from rosclaw_soccer.growth.mosaic_gmt import (
    G1MosaicGMTOverheadSkill,
    MosaicGMTContract,
    MosaicGMTTorchController,
    load_g1_mosaic_gmt_overhead_skill,
    skill_to_dict,
)
from rosclaw_soccer.training.goalkeeper_targeted_dive_mjwarp import (
    GoalkeeperTargetedDiveRLConfig,
)
from rosclaw_soccer.training.goalkeeper_whole_body_reach import (
    G1WholeBodyReachAtlas,
    GoalkeeperWholeBodyReachConfig,
    load_g1_whole_body_reach_atlas,
    whole_body_reach_from_target_numpy,
    whole_body_reach_from_target_torch,
    write_g1_whole_body_reach_atlas,
)
from rosclaw_soccer.training.mosaic_gmt_goalkeeper_probe import (
    MosaicGMTGoalkeeperProbeConfig,
    _actor_training_exam_contract,
    _sanitize_nonfinite_evidence,
)

_HASH = "sha256:" + "1" * 64


def test_mosaic_probe_nonfinite_diagnostics_are_explicitly_nulled() -> None:
    sanitized, paths = _sanitize_nonfinite_evidence(
        {
            "safe": 1.0,
            "tail": [float("nan"), {"speed": float("inf")}],
            "tuple": (1.0, -float("inf")),
        }
    )

    assert sanitized == {
        "safe": 1.0,
        "tail": [None, {"speed": None}],
        "tuple": [1.0, None],
    }
    assert paths == ["tail[0]", "tail[1].speed", "tuple[1]"]


def _contract() -> MosaicGMTContract:
    return MosaicGMTContract(
        checkpoint_hash=_HASH,
        topology_hash=_HASH,
        semantic_contract_hash=MOSAIC_G1_SEMANTIC_CONTRACT_HASH,
        raw_joint_names=MOSAIC_G1_ISAACLAB_JOINT_NAMES,
        body_names=(
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ),
        anchor_body_name="torso_link",
        default_joint_position_rad=tuple(0.01 * index for index in range(29)),
        joint_stiffness=(40.0,) * 29,
        joint_damping=(2.0,) * 29,
        action_scale=(0.5,) * 29,
        observation_history_length=5,
    )


def _skill(contract: MosaicGMTContract) -> G1MosaicGMTOverheadSkill:
    times = np.arange(60, dtype=np.float64) / 50.0 - 0.56
    quaternion = np.zeros((60, 4), dtype=np.float64)
    quaternion[:, 0] = 1.0
    return G1MosaicGMTOverheadSkill(
        checkpoint_hash=contract.checkpoint_hash,
        checkpoint_contract_hash=contract.contract_hash,
        source_hash=_HASH,
        qualification_hash=_HASH,
        semantic_contract_hash=MOSAIC_G1_SEMANTIC_CONTRACT_HASH,
        center_frame=100,
        source_fps=50.0,
        relative_times_sec=tuple(float(value) for value in times),
        raw_joint_position_rad=tuple((0.0,) * 29 for _ in times),
        raw_joint_velocity_rad_s=tuple((0.0,) * 29 for _ in times),
        aligned_torso_quaternion_wxyz=tuple(
            tuple(float(value) for value in row) for row in quaternion
        ),
        official_minimum_pelvis_height_m=0.72,
        official_peak_minimum_bilateral_hand_height_m=1.32,
    )


def test_gmt_skill_loader_is_content_bound(tmp_path: Path) -> None:
    skill = _skill(_contract())
    path = tmp_path / "skill.json"
    path.write_text(json.dumps(skill_to_dict(skill)), encoding="utf-8")

    assert load_g1_mosaic_gmt_overhead_skill(path) == skill
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["raw_joint_position_rad"][20][3] = 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_g1_mosaic_gmt_overhead_skill(path)


def test_gmt_controller_builds_term_major_history_and_resets_inactive() -> None:
    torch = pytest.importorskip("torch")
    contract = _contract()
    skill = _skill(contract)

    class Policy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.last = None

        def forward(self, observation):
            self.last = observation.clone()
            return torch.zeros((observation.shape[0], 29), device=observation.device)

    policy = Policy()
    controller = MosaicGMTTorchController(
        policy=policy,
        contract=contract,
        skill=skill,
        environment_count=2,
        device="cpu",
    )
    position = torch.zeros((2, 29))
    quaternion = torch.zeros((2, 4))
    quaternion[:, 0] = 1.0
    target, action = controller.target(
        canonical_joint_position=position,
        canonical_joint_velocity=position,
        torso_quaternion_wxyz=quaternion,
        base_angular_velocity_body_rad_s=torch.zeros((2, 3)),
        heading_quaternion_wxyz=quaternion,
        relative_time_sec=torch.zeros(2),
        active=torch.tensor((True, False)),
    )

    assert tuple(target.shape) == (2, 29)
    assert torch.count_nonzero(action) == 0
    assert policy.last is not None
    assert tuple(policy.last.shape) == (2, 770)
    assert torch.count_nonzero(policy.last[0]) > 0
    assert torch.count_nonzero(policy.last[1]) == 0
    reference = controller.reference_target(torch.tensor((0.10, 0.20)))
    assert tuple(reference.shape) == (2, 29)
    assert torch.isfinite(reference).all()

    controller.target(
        canonical_joint_position=position,
        canonical_joint_velocity=position,
        torso_quaternion_wxyz=quaternion,
        base_angular_velocity_body_rad_s=torch.zeros((2, 3)),
        heading_quaternion_wxyz=quaternion,
        relative_time_sec=torch.zeros(2),
        active=torch.tensor((False, False)),
    )
    assert torch.count_nonzero(policy.last) == 0


def test_targeted_dive_gmt_contract_is_fail_closed(tmp_path: Path) -> None:
    model = tmp_path / "gmt.onnx"
    skill = tmp_path / "skill.json"
    model.write_bytes(b"model")
    skill.write_text("{}", encoding="utf-8")
    config = GoalkeeperTargetedDiveRLConfig(
        mosaic_gmt_model_path=str(model.resolve()),
        mosaic_gmt_skill_path=str(skill.resolve()),
        mosaic_gmt_blend=0.8,
    )

    assert config.mosaic_gmt_blend == 0.8
    with pytest.raises(ValueError, match="requires model and skill"):
        GoalkeeperTargetedDiveRLConfig(mosaic_gmt_blend=0.8)
    with pytest.raises(ValueError, match="disjoint joint-group authority"):
        GoalkeeperTargetedDiveRLConfig(
            mosaic_gmt_model_path=str(model.resolve()),
            mosaic_gmt_skill_path=str(skill.resolve()),
            mosaic_gmt_blend=0.8,
            runtime_reach_blend=0.2,
        )
    composed = GoalkeeperTargetedDiveRLConfig(
        mosaic_gmt_model_path=str(model.resolve()),
        mosaic_gmt_skill_path=str(skill.resolve()),
        mosaic_gmt_blend=0.8,
        mosaic_gmt_arm_scale=0.0,
        runtime_reach_blend=0.2,
        runtime_reach_contact_standoff_m=0.20,
        runtime_reach_vertical_lead_m=0.18,
        runtime_reach_low_vertical_lead_m=-0.20,
        runtime_reach_mid_vertical_lead_m=0.0,
        runtime_reach_high_vertical_lead_m=0.30,
        mosaic_gmt_minimum_target_height_m=0.40,
        mosaic_gmt_full_target_height_m=1.10,
    )
    assert composed.mosaic_gmt_arm_scale == 0.0
    assert composed.runtime_reach_blend == 0.2
    assert composed.runtime_reach_contact_standoff_m == 0.20
    assert composed.runtime_reach_vertical_lead_m == 0.18
    assert composed.runtime_reach_low_vertical_lead_m == -0.20
    assert composed.mosaic_gmt_minimum_target_height_m == 0.40
    stability_floor = GoalkeeperTargetedDiveRLConfig(
        mosaic_gmt_model_path=str(model.resolve()),
        mosaic_gmt_skill_path=str(skill.resolve()),
        mosaic_gmt_blend=0.8,
        mosaic_gmt_stability_floor=0.25,
        mosaic_gmt_arm_scale=0.0,
    )
    assert stability_floor.mosaic_gmt_stability_floor == 0.25
    with pytest.raises(ValueError, match="cannot own arm joints"):
        GoalkeeperTargetedDiveRLConfig(
            mosaic_gmt_model_path=str(model.resolve()),
            mosaic_gmt_skill_path=str(skill.resolve()),
            mosaic_gmt_blend=0.8,
            mosaic_gmt_stability_floor=0.25,
            mosaic_gmt_arm_scale=0.1,
        )
    with pytest.raises(ValueError, match="reach compensation requires runtime reach"):
        GoalkeeperTargetedDiveRLConfig(runtime_reach_contact_standoff_m=0.20)
    with pytest.raises(ValueError, match="must be complete"):
        GoalkeeperTargetedDiveRLConfig(
            runtime_reach_blend=0.20,
            runtime_reach_low_vertical_lead_m=-0.20,
        )
    with pytest.raises(ValueError, match="MOSAIC GMT settings"):
        GoalkeeperTargetedDiveRLConfig(
            mosaic_gmt_model_path=str(model.resolve()),
            mosaic_gmt_skill_path=str(skill.resolve()),
            mosaic_gmt_blend=0.8,
            mosaic_gmt_arm_scale=0.2,
            mosaic_gmt_minimum_target_height_m=0.40,
            mosaic_gmt_full_target_height_m=1.10,
        )


def test_gmt_probe_exposes_arm_rate_and_filter_stability_controls() -> None:
    config = MosaicGMTGoalkeeperProbeConfig(
        maximum_arm_target_step_rad=0.08,
        arm_target_filter_fraction=0.40,
        substep_option_lower_body_guard_enabled=True,
        substep_option_lower_body_guard_onset_rad_s=2.20,
        substep_option_lower_body_guard_ceiling_rad_s=3.20,
        substep_option_lower_body_minimum_scale=0.20,
        canonical_locomotion_mirror_enabled=True,
        contact_support_side_enabled=True,
        actor_contact_support_side_enabled=True,
    )

    assert config.maximum_arm_target_step_rad == 0.08
    assert config.arm_target_filter_fraction == 0.40
    assert config.substep_option_lower_body_guard_enabled
    assert config.substep_option_lower_body_minimum_scale == 0.20
    assert config.canonical_locomotion_mirror_enabled
    assert config.contact_support_side_enabled
    assert config.actor_contact_support_side_enabled
    with pytest.raises(ValueError, match="probe settings"):
        MosaicGMTGoalkeeperProbeConfig(actor_contact_support_side_enabled=True)
    with pytest.raises(ValueError, match="probe settings"):
        MosaicGMTGoalkeeperProbeConfig(arm_target_filter_fraction=1.01)
    with pytest.raises(ValueError, match="probe settings"):
        replace(config, substep_option_lower_body_guard_onset_rad_s=3.40)
    feedback = MosaicGMTGoalkeeperProbeConfig(
        task_space_reach_blend=0.80,
        task_space_reach_feedback_blend=0.65,
        gmt_arm_scale=0.0,
    )
    assert feedback.task_space_reach_feedback_blend == 0.65
    with pytest.raises(ValueError, match="probe settings"):
        replace(feedback, task_space_reach_blend=0.0)
    official = MosaicGMTGoalkeeperProbeConfig(
        official_goalkeeper_teacher_checkpoint="/tmp/goalkeeper.pt",
        official_goalkeeper_teacher_blend=0.75,
    )
    assert official.official_goalkeeper_teacher_blend == 0.75
    with pytest.raises(ValueError, match="probe settings"):
        replace(official, canonical_locomotion_mirror_enabled=True)


def test_gmt_probe_reports_training_exam_world_contract_mismatch() -> None:
    config = MosaicGMTGoalkeeperProbeConfig(
        first_shot_release_sec=0.90,
        hard_shot_height_mode="low",
        contact_support_side_enabled=True,
        actor_contact_support_side_enabled=True,
    )
    training = {
        "shot_difficulty_profile": "match",
        "training_first_shot_release_sec": 0.90,
        "training_hard_shot_fraction": 1.0,
        "training_hard_shot_height_mode": "low",
        "training_hard_shot_flight_time_range_sec": [0.40, 0.50],
        "targeted_dive_nominal_shot_flight_time_sec": 0.45,
        "targeted_dive_runtime_contact_support_side_enabled": True,
        "targeted_dive_actor_contact_support_side_enabled": True,
        "targeted_dive_post_save_counterstep_recenter_weight": 1.0,
    }

    matched = _actor_training_exam_contract(training, config)
    assert matched["status"] == "MATCHED"
    assert matched["mismatched_fields"] == []
    mismatched = _actor_training_exam_contract(
        {**training, "shot_difficulty_profile": "standard"}, config
    )
    assert mismatched["status"] == "OUT_OF_DISTRIBUTION"
    side_specialist = _actor_training_exam_contract(
        {**training, "training_hard_shot_side_mode": "negative"}, config
    )
    assert side_specialist["status"] == "OUT_OF_DISTRIBUTION"
    assert "training_hard_shot_side_mode" in side_specialist["mismatched_fields"]
    assert mismatched["mismatched_fields"] == ["shot_difficulty_profile"]
    mirrored_runtime = _actor_training_exam_contract(
        training,
        replace(config, canonical_locomotion_mirror_enabled=True),
    )
    assert mirrored_runtime["status"] == "OUT_OF_DISTRIBUTION"
    assert (
        "targeted_dive_canonical_locomotion_mirror_enabled" in mirrored_runtime["mismatched_fields"]
    )
    landing_brake = _actor_training_exam_contract(
        training,
        replace(config, post_save_counterstep_recenter_weight=0.0),
    )
    assert landing_brake["status"] == "OUT_OF_DISTRIBUTION"
    assert (
        "targeted_dive_post_save_counterstep_recenter_weight" in landing_brake["mismatched_fields"]
    )


def test_whole_body_reach_is_bounded_sim_only_and_exactly_reflected(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    config = GoalkeeperWholeBodyReachConfig()
    assert config.config_hash.startswith("sha256:")
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
    targets = tuple(
        (-0.08, lateral, height)
        for lateral in (0.0, 0.25, 0.50, 0.75)
        for height in (-0.35, -0.10, 0.15, 0.40)
    )
    canonical_delta = (
        (0.0, 0.20, 0.10)
        + (0.0,) * 7
        + (
            0.30,
            -0.20,
            0.10,
            0.05,
            -0.02,
            0.01,
            -0.01,
        )
    )
    model = G1WholeBodyReachAtlas(
        body_hash=_HASH,
        config_hash=_HASH,
        target_relative_m=targets,
        joint_delta_rad=(canonical_delta,) * len(targets),
        terminal_error_m=(0.01,) * len(targets),
        interpolation_distance_scales_m=(0.24, 0.20, 0.18),
        interpolation_neighbors=8,
        interpolation_temperature=0.65,
    )
    target = torch.tensor(((-0.08, 0.50, -0.10), (-0.08, -0.50, -0.10)))
    result = whole_body_reach_from_target_torch(torch=torch, target_relative=target, model=model)
    numpy_result = whole_body_reach_from_target_numpy(
        target_relative=target.numpy().astype(np.float64),
        model=model,
    )

    np.testing.assert_allclose(numpy_result, result.numpy(), atol=1e-7, rtol=0.0)
    assert torch.allclose(result[0, 10:17], torch.tensor(canonical_delta[10:17]))
    assert result[1, 1] == pytest.approx(-canonical_delta[1])
    assert torch.allclose(
        result[1, 3:10],
        torch.tensor(canonical_delta[10:17])
        * torch.tensor((1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0)),
    )
    path = tmp_path / "reach-atlas.json"
    write_g1_whole_body_reach_atlas(model, path)
    assert load_g1_whole_body_reach_atlas(path) == model
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["joint_delta_rad"][0][0] = 9.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        load_g1_whole_body_reach_atlas(path)


def test_gmt_probe_declares_balanced_height_routing() -> None:
    config = MosaicGMTGoalkeeperProbeConfig(
        hard_shot_height_mode="balanced",
        gmt_minimum_target_height_m=0.40,
        gmt_full_target_height_m=1.10,
        posture_exception_duration_sec=2.0,
        prediction_lead_sec=0.50,
        anchor_lower_body_scale=0.10,
        anchor_waist_scale=0.40,
    )

    assert config.hard_shot_height_mode == "balanced"
    assert config.gmt_minimum_target_height_m == 0.40
    assert config.posture_exception_duration_sec == 2.0
    assert config.prediction_lead_sec == 0.50
    assert config.anchor_lower_body_scale == 0.10
    crouch = replace(
        config,
        decoder_lower_body_residual_authority=1.0,
        decoder_lower_body_command_scale=0.75,
        decoder_waist_residual_authority=1.0,
    )
    assert crouch.decoder_lower_body_command_scale == 0.75
    assert crouch.decoder_waist_residual_authority == 1.0
    with pytest.raises(ValueError, match="probe settings"):
        MosaicGMTGoalkeeperProbeConfig(
            gmt_minimum_target_height_m=1.20,
            gmt_full_target_height_m=1.10,
        )
    with pytest.raises(ValueError, match="probe settings"):
        MosaicGMTGoalkeeperProbeConfig(prediction_lead_sec=1.01)
    with pytest.raises(ValueError, match="probe settings"):
        MosaicGMTGoalkeeperProbeConfig(
            anchor_lower_body_scale=0.50,
            anchor_waist_scale=0.40,
        )
    with pytest.raises(ValueError, match="probe settings"):
        replace(config, decoder_lower_body_command_scale=0.05)
    height_conditioned = MosaicGMTGoalkeeperProbeConfig(
        task_space_reach_blend=0.20,
        gmt_arm_scale=0.0,
        task_space_low_vertical_lead_m=-0.20,
        task_space_mid_vertical_lead_m=0.0,
        task_space_high_vertical_lead_m=0.30,
    )
    assert height_conditioned.task_space_low_vertical_lead_m == -0.20
    assert MosaicGMTGoalkeeperProbeConfig(hard_shot_height_mode="low")
