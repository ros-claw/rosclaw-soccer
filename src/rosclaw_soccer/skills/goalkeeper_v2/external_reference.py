"""License-isolated inference adapter for an external goalkeeper teacher.

This module deliberately does not vendor the checkpoint, motion tensors, or
adapted weights into ROSClaw Soccer.  A user may build a content-addressed,
SIM_ONLY NumPy bundle *outside* the source checkout from the separately
licensed Humanoid-Goalkeeper repository.  The bundle remains research-only
CC-BY-NC-SA-4.0 material and cannot become a ROSClaw champion artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_json

_REFERENCE_COMMIT = "976a81ff19b7306bafbe923d2890066b68a85271"
_REFERENCE_REPOSITORY = "https://github.com/InternRobotics/Humanoid-Goalkeeper"
_CHECKPOINT_RELATIVE_PATH = Path("legged_gym/resources/weight/goalkeeper.pt")
_LICENSE_RELATIVE_PATH = Path("LICENSE")
_EXPECTED_LICENSE_HASH = "sha256:6c8cd1cdbe7accec4f63b6c3afb45ce0ffae9ed6abc0ca55acf5900b37970a82"
_EXPECTED_CHECKPOINT_HASH = (
    "sha256:7ecdedff5de6e30a0a4d11742561a9be6c94d8faeefc4701f3e8788381b67b14"
)
_REFERENCE_DEFAULT_JOINT_POSITION = (
    -0.1,
    0.2,
    0.0,
    0.3,
    -0.2,
    -0.2,
    -0.1,
    -0.2,
    0.0,
    0.3,
    -0.2,
    0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.5,
    0.0,
    1.2,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.5,
    0.0,
    1.2,
    0.0,
    0.0,
    0.0,
)
_NETWORK_SHAPES = {
    "history_encoder.0.weight": (128, 960),
    "history_encoder.0.bias": (128,),
    "history_encoder.2.weight": (64, 128),
    "history_encoder.2.bias": (64,),
    "history_encoder.4.weight": (16, 64),
    "history_encoder.4.bias": (16,),
    "ball_estimator.0.weight": (128, 960),
    "ball_estimator.0.bias": (128,),
    "ball_estimator.2.weight": (32, 128),
    "ball_estimator.2.bias": (32,),
    "ball_estimator.4.weight": (6, 32),
    "ball_estimator.4.bias": (6,),
    "region_estimator.0.weight": (128, 960),
    "region_estimator.0.bias": (128,),
    "region_estimator.2.weight": (32, 128),
    "region_estimator.2.bias": (32,),
    "region_estimator.4.weight": (6, 32),
    "region_estimator.4.bias": (6,),
    "actor.0.weight": (512, 119),
    "actor.0.bias": (512,),
    "actor.2.weight": (256, 512),
    "actor.2.bias": (256,),
    "actor.4.weight": (256, 256),
    "actor.4.bias": (256,),
    "actor.6.weight": (29, 256),
    "actor.6.bias": (29,),
}


@dataclass(frozen=True)
class HumanoidGoalkeeperReferenceManifest:
    weights_file: str
    weights_file_hash: str
    source_checkpoint_hash: str
    source_license_hash: str
    source_commit: str
    joint_names: tuple[str, ...]
    default_joint_position_rad: tuple[float, ...]
    action_scale_rad: float = 0.25
    observation_history_steps: int = 10
    one_step_observation_size: int = 96
    attribution_required: bool = True
    commercial_use_allowed: bool = False
    share_alike_required: bool = True
    external_teacher_only: bool = True
    champion_eligible: bool = False
    activation_ceiling: str = "SIM_ONLY"
    repository: str = _REFERENCE_REPOSITORY
    schema_version: str = "rosclaw_soccer.humanoid_goalkeeper_reference_manifest.v1"

    def __post_init__(self) -> None:
        if Path(self.weights_file).name != self.weights_file or not self.weights_file.endswith(
            ".npz"
        ):
            raise ValueError("reference weights must be a sibling numeric NPZ file")
        for value in (
            self.weights_file_hash,
            self.source_checkpoint_hash,
            self.source_license_hash,
        ):
            if not value.startswith("sha256:"):
                raise ValueError("reference provenance requires content hashes")
        if self.source_commit != _REFERENCE_COMMIT or self.repository != _REFERENCE_REPOSITORY:
            raise ValueError("reference source identity changed")
        if self.joint_names != G1_DDS_JOINT_NAMES or len(self.default_joint_position_rad) != 29:
            raise ValueError("reference bundle requires the canonical 29-DoF G1 contract")
        if not math.isclose(self.action_scale_rad, 0.25, abs_tol=1e-12):
            raise ValueError("reference checkpoint action scale changed")
        if self.observation_history_steps != 10 or self.one_step_observation_size != 96:
            raise ValueError("reference checkpoint observation contract changed")
        if (
            not self.attribution_required
            or self.commercial_use_allowed
            or not self.share_alike_required
            or not self.external_teacher_only
            or self.champion_eligible
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("reference bundle violated its CC-BY-NC-SA research boundary")

    @property
    def manifest_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = asdict(self)
        value["joint_names"] = list(self.joint_names)
        value["default_joint_position_rad"] = list(self.default_joint_position_rad)
        if include_hash:
            value["manifest_hash"] = self.manifest_hash
        return value


@dataclass(frozen=True)
class HumanoidGoalkeeperReferenceAction:
    target_joint_position_rad: tuple[float, ...]
    normalized_action: tuple[float, ...]
    estimated_region: int
    manifest_hash: str


class NumpyHumanoidGoalkeeperReferenceActor:
    """Inference-only reproduction of the published teacher architecture."""

    def __init__(
        self,
        manifest: HumanoidGoalkeeperReferenceManifest,
        arrays: dict[str, NDArray[np.float64]],
    ) -> None:
        self.manifest = manifest
        self._arrays = arrays
        self._history: deque[NDArray[np.float64]] = deque(maxlen=manifest.observation_history_steps)
        self._previous_action = np.zeros(29, dtype=np.float64)

    def reset(self) -> None:
        self._history.clear()
        self._previous_action[:] = 0.0

    def action(
        self,
        *,
        ball_position_local_m: NDArray[np.float64],
        ball_visible: bool,
        angular_velocity_rad_s: NDArray[np.float64],
        projected_gravity: NDArray[np.float64],
        joint_position_rad: NDArray[np.float64],
        joint_velocity_rad_s: NDArray[np.float64],
        region_override: int | None = None,
    ) -> HumanoidGoalkeeperReferenceAction:
        vectors = (
            (ball_position_local_m, 3, "ball position"),
            (angular_velocity_rad_s, 3, "angular velocity"),
            (projected_gravity, 3, "projected gravity"),
            (joint_position_rad, 29, "joint position"),
            (joint_velocity_rad_s, 29, "joint velocity"),
        )
        checked: list[NDArray[np.float64]] = []
        for value, size, label in vectors:
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (size,) or not np.all(np.isfinite(array)):
                raise ValueError(f"reference actor {label} is invalid")
            checked.append(array)
        ball, angular, gravity, q, dq = checked
        visible_ball = ball if ball_visible else np.zeros(3, dtype=np.float64)
        default = np.asarray(self.manifest.default_joint_position_rad, dtype=np.float64)
        step = np.concatenate(
            (
                visible_ball,
                angular * 0.25,
                gravity,
                q - default,
                dq * 0.05,
                self._previous_action,
            )
        )
        if step.shape != (96,):
            raise RuntimeError("reference actor one-step observation changed")
        self._history.append(step)
        padded = [self._history[0]] * (self.manifest.observation_history_steps - len(self._history))
        history = np.concatenate((*padded, *self._history))[None, :]
        latent = self._network(history, "history_encoder", activation="relu")
        ball_estimate = self._network(history, "ball_estimator", activation="relu")
        region_logits = self._network(history, "region_estimator", activation="relu")
        if region_override is not None and not 0 <= region_override < 6:
            raise ValueError("reference actor region override must be in [0, 5]")
        region = int(np.argmax(region_logits[0])) if region_override is None else region_override
        actor_input = np.concatenate(
            (history[:, -96:], latent, ball_estimate, np.asarray(((region,),), dtype=np.float64)),
            axis=1,
        )
        action = self._network(actor_input, "actor", activation="elu")[0]
        if action.shape != (29,) or not np.all(np.isfinite(action)):
            raise RuntimeError("reference actor produced an invalid action")
        self._previous_action = action.copy()
        target = default + self.manifest.action_scale_rad * action
        return HumanoidGoalkeeperReferenceAction(
            target_joint_position_rad=tuple(float(value) for value in target),
            normalized_action=tuple(float(value) for value in action),
            estimated_region=region,
            manifest_hash=self.manifest.manifest_hash,
        )

    def _network(
        self,
        value: NDArray[np.float64],
        prefix: str,
        *,
        activation: str,
    ) -> NDArray[np.float64]:
        indices = (0, 2, 4, 6) if prefix == "actor" else (0, 2, 4)
        for position, index in enumerate(indices):
            value = value @ self._arrays[f"{prefix}.{index}.weight"].T
            value += self._arrays[f"{prefix}.{index}.bias"]
            if position != len(indices) - 1:
                if activation == "relu":
                    value = np.maximum(value, 0.0)
                else:
                    value = np.where(value > 0.0, value, np.expm1(value))
        return value


def build_humanoid_goalkeeper_reference_bundle(
    *,
    reference_checkout: Path,
    output_manifest_path: Path,
    source_checkout: Path,
) -> HumanoidGoalkeeperReferenceManifest:
    """Convert the published checkpoint to non-pickle arrays outside Git."""

    root = reference_checkout.expanduser().resolve()
    output = output_manifest_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("external reference bundles must remain outside source checkout")
    if output.suffix != ".json" or output.exists():
        raise ValueError("external reference manifest requires a new JSON path")
    checkpoint = root / _CHECKPOINT_RELATIVE_PATH
    license_path = root / _LICENSE_RELATIVE_PATH
    if _file_hash(checkpoint) != _EXPECTED_CHECKPOINT_HASH:
        raise ValueError("external reference checkpoint hash changed")
    if _file_hash(license_path) != _EXPECTED_LICENSE_HASH:
        raise ValueError("external reference license hash changed")
    if "Attribution-NonCommercial-ShareAlike 4.0" not in license_path.read_text(encoding="utf-8"):
        raise ValueError("external reference license terms are unavailable")
    commit = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != _REFERENCE_COMMIT:
        raise ValueError("external reference checkout is not pinned")
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError("external reference checkpoint lacks a model state")
    arrays: dict[str, NDArray[np.float32]] = {}
    for key, shape in _NETWORK_SHAPES.items():
        tensor = state.get(key)
        if tensor is None or tuple(tensor.shape) != shape:
            raise ValueError(f"external reference tensor changed: {key}")
        array = np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"external reference tensor is non-finite: {key}")
        arrays[key] = array
    output.parent.mkdir(parents=True, exist_ok=True)
    weights_path = output.with_suffix(".npz")
    if weights_path.exists():
        raise ValueError("external reference weights output already exists")
    # NumPy's overload stubs enumerate archive-specific keyword arguments and
    # do not model arbitrary named arrays, although that is the public API.
    np.savez_compressed(weights_path, **arrays)  # type: ignore[arg-type]
    manifest = HumanoidGoalkeeperReferenceManifest(
        weights_file=weights_path.name,
        weights_file_hash=_file_hash(weights_path),
        source_checkpoint_hash=_EXPECTED_CHECKPOINT_HASH,
        source_license_hash=_EXPECTED_LICENSE_HASH,
        source_commit=_REFERENCE_COMMIT,
        joint_names=G1_DDS_JOINT_NAMES,
        default_joint_position_rad=_REFERENCE_DEFAULT_JOINT_POSITION,
    )
    output.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_humanoid_goalkeeper_reference_actor(
    manifest_path: Path,
) -> NumpyHumanoidGoalkeeperReferenceActor:
    path = manifest_path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed_hash = str(payload.pop("manifest_hash", ""))
    try:
        payload["joint_names"] = tuple(payload["joint_names"])
        payload["default_joint_position_rad"] = tuple(payload["default_joint_position_rad"])
        manifest = HumanoidGoalkeeperReferenceManifest(**payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("external reference manifest is invalid") from exc
    if claimed_hash != manifest.manifest_hash:
        raise ValueError("external reference manifest content hash mismatch")
    weights_path = path.parent / manifest.weights_file
    if _file_hash(weights_path) != manifest.weights_file_hash:
        raise ValueError("external reference weights content hash mismatch")
    with np.load(weights_path, allow_pickle=False) as archive:
        if set(archive.files) != set(_NETWORK_SHAPES):
            raise ValueError("external reference tensor set changed")
        arrays = {key: np.asarray(archive[key], dtype=np.float64) for key in _NETWORK_SHAPES}
    for key, shape in _NETWORK_SHAPES.items():
        if arrays[key].shape != shape or not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"external reference tensor invalid: {key}")
    return NumpyHumanoidGoalkeeperReferenceActor(manifest, arrays)


def _file_hash(path: Path) -> str:
    if not path.is_file() or path.stat().st_size > 1024 * 1024 * 1024:
        raise ValueError(f"external reference file is unavailable or oversized: {path}")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "HumanoidGoalkeeperReferenceAction",
    "HumanoidGoalkeeperReferenceManifest",
    "NumpyHumanoidGoalkeeperReferenceActor",
    "build_humanoid_goalkeeper_reference_bundle",
    "load_humanoid_goalkeeper_reference_actor",
]
