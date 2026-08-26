"""Fail-closed qualification for external G1/RoboNaldo simulation assets."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json

_POLICY_REL = Path("policy/robonaldo/model/policy-obs-aic.onnx")
_MOTION_REL = Path("policy/robonaldo/model/freekick_motion.npz")
_SCENE_REL = Path("g1_description/scene_with_ball.xml")
_MODEL_REL = Path("g1_description/g1_liao.xml")
_FREEKICK_REL = Path("policy/robonaldo/FreeKick.py")
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class G1AssetQualification:
    eligible: bool
    asset_root: Path
    body_hash: str
    kick_prior_hash: str
    motion_hash: str
    backend_commit: str
    actuator_count: int
    joint_names: tuple[str, ...]
    policy_input_size: int
    policy_output_size: int
    errors: tuple[str, ...]
    schema_version: str = "rosclaw.g1_goalforge.asset_qualification.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["asset_root"] = str(self.asset_root)
        value["joint_names"] = list(self.joint_names)
        value["errors"] = list(self.errors)
        return value

    def require_eligible(self) -> None:
        if not self.eligible:
            raise ValueError("G1 assets are not eligible: " + "; ".join(self.errors))


def qualify_g1_assets(asset_root: Path) -> G1AssetQualification:
    """Qualify a bounded external deployment checkout without opening transports."""

    root = asset_root.expanduser().resolve()
    errors: list[str] = []
    required = (_POLICY_REL, _MOTION_REL, _SCENE_REL, _MODEL_REL, _FREEKICK_REL)
    missing = [str(path) for path in required if not (root / path).is_file()]
    if missing:
        errors.append("missing_assets=" + ",".join(missing))
        return _failed(root, errors)
    oversized = [
        str(path) for path in required if (root / path).stat().st_size > _MAX_ARTIFACT_BYTES
    ]
    if oversized:
        errors.append("oversized_assets=" + ",".join(oversized))
        return _failed(root, errors)

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(root / _SCENE_REL))
    joint_names = tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)) for index in range(1, 30)
    )
    if model.nu != 29:
        errors.append(f"actuator_count={model.nu},expected=29")
    if joint_names != G1_DDS_JOINT_NAMES:
        errors.append("joint_order_does_not_match_unitree_hg_dds")
    for body_name in (
        "pelvis",
        "torso_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) < 0:
            errors.append(f"missing_body={body_name}")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball") < 0:
        errors.append("missing_ball_body")

    input_size = 0
    output_size = 0
    try:
        import onnxruntime

        session = onnxruntime.InferenceSession(
            str(root / _POLICY_REL),
            providers=["CPUExecutionProvider"],
        )
        input_size = int(session.get_inputs()[0].shape[-1])
        output_size = int(session.get_outputs()[0].shape[-1])
        if input_size != 547 or output_size != 29:
            errors.append(f"policy_shape={input_size}->{output_size},expected=547->29")
    except Exception as exc:  # noqa: BLE001 - dependency failures are evidence
        errors.append(f"onnx_qualification={type(exc).__name__}:{exc}")

    try:
        with np.load(root / _MOTION_REL, allow_pickle=False) as motion:
            if motion["joint_pos"].ndim != 2 or motion["joint_pos"].shape[1] != 29:
                errors.append("motion_joint_shape_invalid")
            if motion["body_pos_w"].ndim != 3 or motion["body_quat_w"].shape[-1] != 4:
                errors.append("motion_body_shape_invalid")
            if not all(np.all(np.isfinite(motion[name])) for name in motion.files):
                errors.append("motion_contains_non_finite_values")
    except Exception as exc:  # noqa: BLE001 - malformed assets are evidence
        errors.append(f"motion_qualification={type(exc).__name__}:{exc}")

    model_hash = _bounded_file_hash(root / _MODEL_REL)
    scene_hash = _bounded_file_hash(root / _SCENE_REL)
    policy_hash = _bounded_file_hash(root / _POLICY_REL)
    motion_hash = _bounded_file_hash(root / _MOTION_REL)
    body_hash = hash_json(
        {
            "model_hash": model_hash,
            "scene_hash": scene_hash,
            "joint_names": joint_names,
            "hard_torque_limits": G1_HARD_TORQUE_LIMITS,
        }
    )
    return G1AssetQualification(
        eligible=not errors,
        asset_root=root,
        body_hash=body_hash,
        kick_prior_hash=policy_hash,
        motion_hash=motion_hash,
        backend_commit=_git_commit(root),
        actuator_count=int(model.nu),
        joint_names=joint_names,
        policy_input_size=input_size,
        policy_output_size=output_size,
        errors=tuple(errors),
    )


def g1_body_hash(asset_root: Path) -> str:
    """Bind learning artifacts to the G1 body without requiring ONNX runtime.

    A motion prior never executes the free-kick ONNX policy. Requiring that
    optional inference dependency in a GPU physics worker couples unrelated
    subsystems. This bounded helper keeps the same model/scene/joint/torque
    commitment used by :func:`qualify_g1_assets` while validating only assets
    relevant to a motion prior.
    """

    root = asset_root.expanduser().resolve()
    for relative in (_MODEL_REL, _SCENE_REL):
        path = root / relative
        if not path.is_file() or path.stat().st_size > _MAX_ARTIFACT_BYTES:
            raise ValueError(f"G1 Body asset is missing or oversized: {relative}")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(root / _SCENE_REL))
    joint_names = tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)) for index in range(1, 30)
    )
    if model.nu != 29 or joint_names != G1_DDS_JOINT_NAMES:
        raise ValueError("G1 Body joint/actuator contract mismatch")
    return str(
        hash_json(
            {
                "model_hash": _bounded_file_hash(root / _MODEL_REL),
                "scene_hash": _bounded_file_hash(root / _SCENE_REL),
                "joint_names": joint_names,
                "hard_torque_limits": G1_HARD_TORQUE_LIMITS,
            }
        )
    )


def trajectory_digest(trajectory: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(trajectory):
        value = np.ascontiguousarray(trajectory[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return "sha256:" + digest.hexdigest()


def _failed(root: Path, errors: list[str]) -> G1AssetQualification:
    zero = "sha256:" + "0" * 64
    return G1AssetQualification(
        eligible=False,
        asset_root=root,
        body_hash=zero,
        kick_prior_hash=zero,
        motion_hash=zero,
        backend_commit="unknown",
        actuator_count=0,
        joint_names=(),
        policy_input_size=0,
        policy_output_size=0,
        errors=tuple(errors),
    )


def _bounded_file_hash(path: Path) -> str:
    if path.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("asset exceeds bounded hashing limit")
    return hash_bytes(path.read_bytes())


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


__all__ = [
    "G1AssetQualification",
    "g1_body_hash",
    "qualify_g1_assets",
    "trajectory_digest",
]
