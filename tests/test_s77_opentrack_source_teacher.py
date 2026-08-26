from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.growth.opentrack_tracking import (
    OPENTRACK_DEFAULT_JOINT_POSITION,
    load_opentrack_tracking_torch,
    opentrack_tracking_observation_torch,
)
from rosclaw_soccer.media.recovery_moe_video import (
    validate_recovery_moe_video_manifest,
)
from rosclaw_soccer.training.opentrack_teacher_source_exam import (
    validate_opentrack_teacher_source_exam_report,
)

_ROOT = Path(
    "/code/rosclaw/rosclaw_football/checkpoints/opentrack-official/"
    "specialist_trackers_lafan1_v2/"
    "05132118_G1TrackingGeneralDR_new_specialist2/checkpoints"
)
_POLICY = _ROOT / "002000158720/policy.onnx"
_CONFIG = _ROOT / "config.json"
_MOTION = Path(
    "/code/rosclaw/rosclaw_football/repos/OpenTrack/storage/data/mocap/"
    "lafan1s51/UnitreeG1/fallAndGetUp2_subject2.npz"
)
_SOURCE_EVIDENCE = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "opentrack-capture-moe-source-stadium-v3.json"
)
_VIDEO_MANIFEST = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/showcase/"
    "recovery-moe-source-stadium-v1.json"
)


@pytest.mark.skipif(not _POLICY.is_file(), reason="external OpenTrack expert unavailable")
def test_opentrack_torch_policy_matches_onnx_and_is_sim_only() -> None:
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    policy, contract, references = load_opentrack_tracking_torch(
        policy_path=_POLICY,
        config_path=_CONFIG,
        motion_path=_MOTION,
        device="cpu",
    )
    observation = np.random.default_rng(77).normal(size=(3, 156)).astype(np.float32)
    expected = ort.InferenceSession(str(_POLICY)).run(
        ["continuous_actions"], {"obs": observation}
    )[0]
    actual = policy(torch.as_tensor(observation)).detach().numpy()
    assert np.max(np.abs(expected - actual)) < 2.0e-5
    assert contract.activation_ceiling == "SIM_ONLY"
    assert not contract.hardware_authorized
    assert references["qpos"].shape[1] == 36
    assert references["feet_height"].shape[1] == 4


def test_opentrack_observation_has_exact_order_and_shape() -> None:
    torch = pytest.importorskip("torch")
    count = 2
    joint = torch.zeros((count, 29))
    reference = torch.ones((count, 29))
    observation = opentrack_tracking_observation_torch(
        canonical_joint_position=joint,
        canonical_joint_velocity=joint,
        pelvis_quaternion_wxyz=torch.tensor(((1.0, 0.0, 0.0, 0.0),) * count),
        root_angular_velocity_body_rad_s=torch.zeros((count, 3)),
        previous_motor_target=torch.full((count, 29), 0.25),
        reference_joint_position=reference,
        reference_joint_velocity=reference,
        reference_feet_height_m=torch.full((count, 4), 0.1),
        reference_root_height_m=torch.full((count,), 0.2),
    )
    assert tuple(observation.shape) == (count, 156)
    assert torch.allclose(observation[:, :29], torch.ones((count, 29)))
    assert torch.allclose(observation[:, 29:58], torch.full((count, 29), 0.05))
    assert observation[0, 60].item() == -1.0
    assert torch.allclose(
        observation[:, 64:93],
        -torch.as_tensor(OPENTRACK_DEFAULT_JOINT_POSITION).repeat(count, 1),
    )
    assert torch.allclose(observation[:, -5:-1], torch.full((count, 4), 0.1))
    assert torch.allclose(observation[:, -1], torch.full((count,), 0.2))


@pytest.mark.skipif(not _SOURCE_EVIDENCE.is_file(), reason="source-scene evidence unavailable")
def test_opentrack_source_moe_evidence_is_content_bound_and_complete(
    tmp_path: Path,
) -> None:
    report = validate_opentrack_teacher_source_exam_report(_SOURCE_EVIDENCE)
    assert report["decision"] == "SOURCE_SCENE_TEACHER_REACHABILITY_SUPPORTED"
    assert report["final_stable_count"] == report["environment_count"] == 9
    assert all(row["capture_handoff_completed"] for row in report["rows"])
    assert min(row["final_continuous_stable_sec"] for row in report["rows"]) >= 2.0

    tampered = tmp_path / "tampered.json"
    tampered.write_text(_SOURCE_EVIDENCE.read_text(encoding="utf-8"), encoding="utf-8")
    payload = json.loads(tampered.read_text(encoding="utf-8"))
    payload["final_stable_count"] = 8
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        validate_opentrack_teacher_source_exam_report(tampered)


@pytest.mark.skipif(not _VIDEO_MANIFEST.is_file(), reason="showcase manifest unavailable")
def test_recovery_moe_video_manifest_is_fail_closed(tmp_path: Path) -> None:
    manifest = validate_recovery_moe_video_manifest(_VIDEO_MANIFEST)
    assert manifest["visualization_only"] is True
    assert manifest["pixels_used_for_scoring"] is False
    assert manifest["promotion_eligible"] is False

    tampered = tmp_path / "tampered-video.json"
    payload = json.loads(_VIDEO_MANIFEST.read_text(encoding="utf-8"))
    payload["frame_count"] -= 1
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        validate_recovery_moe_video_manifest(tampered)
