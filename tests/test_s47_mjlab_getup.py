from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.growth.mjlab_getup import (
    MJLabGetUpTorchController,
    MJLabRecoveryHandoff,
    MJLabRecoveryHandoffConfig,
    estimate_mjlab_getup_reference_frame_torch,
    load_mjlab_getup_torch,
)

_ROOT = Path("/code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy")
_MODEL = _ROOT / "policy/beyondmimic_mj/model/beyondmimic_mj.onnx"
_SOURCE = _ROOT / "policy/beyondmimic_mj/model/fallAndGetUp2_subject2_mj.npz"
_CONFIG = _ROOT / "policy/beyondmimic_mj/config/standup_mj.yaml"


@pytest.mark.skipif(not _MODEL.is_file(), reason="external RoboNaldo get-up model unavailable")
def test_mjlab_getup_torch_matches_onnx_and_is_sim_only() -> None:
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    policy, contract, references = load_mjlab_getup_torch(
        checkpoint_path=_MODEL,
        source_path=_SOURCE,
        config_path=_CONFIG,
        asset_root=_ROOT,
        device="cpu",
    )
    rng = np.random.default_rng(47)
    observation = rng.normal(0.0, 0.25, size=(1, 154)).astype(np.float32)
    expected = ort.InferenceSession(str(_MODEL)).run(
        ["actions"],
        {"obs": observation, "time_step": np.zeros((1, 1), dtype=np.float32)},
    )[0]
    actual = policy(torch.as_tensor(observation)).detach().numpy()
    assert np.max(np.abs(expected - actual)) < 2.0e-5
    assert contract.activation_ceiling == "SIM_ONLY"
    assert not contract.hardware_authorized
    assert contract.duration_sec == pytest.approx(8.98)
    assert tuple(references["joint_position"].shape) == (450, 29)


@pytest.mark.skipif(not _MODEL.is_file(), reason="external RoboNaldo get-up model unavailable")
def test_mjlab_getup_controller_has_causal_warmup() -> None:
    torch = pytest.importorskip("torch")
    policy, contract, references = load_mjlab_getup_torch(
        checkpoint_path=_MODEL,
        source_path=_SOURCE,
        config_path=_CONFIG,
        asset_root=_ROOT,
        device="cpu",
    )
    controller = MJLabGetUpTorchController(
        policy=policy,
        contract=contract,
        references=references,
        environment_count=2,
        device="cpu",
    )
    position = torch.zeros((2, 29))
    quaternion = references["torso_quaternion"][0].repeat(2, 1)
    target, action = controller.target(
        canonical_joint_position=position,
        canonical_joint_velocity=position,
        torso_quaternion_wxyz=quaternion,
        base_angular_velocity_body_rad_s=torch.zeros((2, 3)),
        relative_time_sec=torch.tensor((0.10, 0.30)),
        active=torch.tensor((True, False)),
    )
    assert torch.count_nonzero(action[0]) == 0
    assert torch.count_nonzero(action[1]) == 0
    assert torch.count_nonzero(target[0]) > 0
    assert torch.count_nonzero(target[1]) == 0


@pytest.mark.skipif(not _MODEL.is_file(), reason="external RoboNaldo get-up model unavailable")
def test_mjlab_getup_phase_estimator_recovers_exact_reference_frames() -> None:
    torch = pytest.importorskip("torch")
    policy, contract, references = load_mjlab_getup_torch(
        checkpoint_path=_MODEL,
        source_path=_SOURCE,
        config_path=_CONFIG,
        asset_root=_ROOT,
        device="cpu",
    )
    expected = torch.tensor((37, 244), dtype=torch.long)
    frame, weights = estimate_mjlab_getup_reference_frame_torch(
        references=references,
        canonical_joint_position=references["joint_position"][expected],
        canonical_joint_velocity=references["joint_velocity"][expected],
        pelvis_height_m=references["pelvis_position"][expected, 2],
        pelvis_quaternion_wxyz=references["pelvis_quaternion"][expected],
    )
    assert torch.equal(frame, expected)
    assert weights["joint_position_mse"] == 1.0
    controller = MJLabGetUpTorchController(
        policy=policy,
        contract=contract,
        references=references,
        environment_count=2,
        device="cpu",
        initial_reference_frame=frame,
    )
    assert torch.equal(controller.initial_reference_frame, expected)

    replacement = torch.tensor((51, 301), dtype=torch.long)
    controller.set_initial_reference_frame_before_start(
        replacement,
        mask=torch.tensor((False, True)),
    )
    assert controller.initial_reference_frame.tolist() == [37, 301]
    position = torch.zeros((2, 29))
    controller.target(
        canonical_joint_position=position,
        canonical_joint_velocity=position,
        torso_quaternion_wxyz=references["torso_quaternion"][expected],
        base_angular_velocity_body_rad_s=torch.zeros((2, 3)),
        relative_time_sec=torch.zeros(2),
        active=torch.tensor((True, False)),
    )
    with pytest.raises(RuntimeError, match="immutable"):
        controller.set_initial_reference_frame_before_start(replacement)


def test_recovery_handoff_requires_continuous_bilateral_low_momentum_hold() -> None:
    torch = pytest.importorskip("torch")
    config = MJLabRecoveryHandoffConfig(stable_hold_sec=0.50, blend_sec=0.50)
    handoff = MJLabRecoveryHandoff(config=config, environment_count=2, device="cpu")
    height = torch.tensor((0.75, 0.75))
    upright = torch.tensor((0.98, 0.98))
    linear = torch.tensor((0.10, 0.10))
    angular = torch.tensor((0.20, 0.20))
    support = torch.tensor((True, True))
    missing_left = torch.tensor((True, False))

    first = handoff.update(
        pelvis_height_m=height,
        upright_projection=upright,
        root_linear_speed_mps=linear,
        root_angular_speed_rad_s=angular,
        left_foot_supported=missing_left,
        right_foot_supported=support,
        active=support,
    )
    assert first.reset_locomotion.tolist() == [True, False]
    assert not bool(first.handoff_started.any())
    for _ in range(config.stable_hold_steps - 1):
        signals = handoff.update(
            pelvis_height_m=height,
            upright_projection=upright,
            root_linear_speed_mps=linear,
            root_angular_speed_rad_s=angular,
            left_foot_supported=support,
            right_foot_supported=support,
            active=support,
        )
    assert signals.handoff_started.tolist() == [True, False]
    final = handoff.update(
        pelvis_height_m=height,
        upright_projection=upright,
        root_linear_speed_mps=linear,
        root_angular_speed_rad_s=angular,
        left_foot_supported=support,
        right_foot_supported=support,
        active=support,
    )
    assert final.handoff_started.tolist() == [True, True]
    assert 0.0 < float(final.blend_fraction[0]) < 1.0
    assert float(final.blend_fraction[1]) == 0.0


def test_recovery_handoff_fails_closed_on_nonfinite_state() -> None:
    torch = pytest.importorskip("torch")
    handoff = MJLabRecoveryHandoff(
        config=MJLabRecoveryHandoffConfig(stable_hold_sec=0.50),
        environment_count=1,
        device="cpu",
    )
    signals = handoff.update(
        pelvis_height_m=torch.tensor((float("nan"),)),
        upright_projection=torch.ones(1),
        root_linear_speed_mps=torch.zeros(1),
        root_angular_speed_rad_s=torch.zeros(1),
        left_foot_supported=torch.ones(1, dtype=torch.bool),
        right_foot_supported=torch.ones(1, dtype=torch.bool),
        active=torch.ones(1, dtype=torch.bool),
    )
    assert not bool(signals.stable_candidate.item())
    assert not bool(signals.warm_locomotion.item())
    assert not bool(signals.handoff_started.item())
