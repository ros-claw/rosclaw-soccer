"""Content-bound physical snapshots for failure-driven recovery curricula.

Reference motions answer *what a recovery should look like*.  They do not
describe the state distribution created by a preceding physical skill.  This
module records that missing boundary: the body state, contacts, momentum,
action history and causal event that enter recovery after a real simulation
rollout.

The archive is deliberately NumPy-only and pickle-free.  It grants no runtime
or hardware authority; consumers must separately pass their own promotion
gate before a learned recovery candidate can be activated.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

RecoveryStage = Literal[
    "SAVE_EVENT",
    "POST_SAVE_FLIGHT",
    "LANDING",
    "RECOVERY_ENTRY",
    "FAILURE_TERMINAL",
    "EPISODE_TERMINAL",
]
RecoveryPostureCluster = Literal[
    "STANDING",
    "AIRBORNE_OR_HIGH_MOMENTUM",
    "KNEELING_OR_SUPPORTED",
    "LEFT_SIDE",
    "RIGHT_SIDE",
    "PRONE",
    "SUPINE",
    "AMBIGUOUS_FALLEN",
]

_STAGES = {
    "SAVE_EVENT",
    "POST_SAVE_FLIGHT",
    "LANDING",
    "RECOVERY_ENTRY",
    "FAILURE_TERMINAL",
    "EPISODE_TERMINAL",
}
_CLUSTERS = {
    "STANDING",
    "AIRBORNE_OR_HIGH_MOMENTUM",
    "KNEELING_OR_SUPPORTED",
    "LEFT_SIDE",
    "RIGHT_SIDE",
    "PRONE",
    "SUPINE",
    "AMBIGUOUS_FALLEN",
}


def _valid_hash(value: str) -> bool:
    return value.startswith("sha256:") and len(value) == 71


def _finite_vector(
    value: NDArray[np.floating[Any]] | Sequence[float],
    *,
    width: int,
    name: str,
) -> NDArray[np.float64]:
    # The corpus archive is float32 to match the MJWarp state tensors.  Round
    # here as well so row hashes bind exactly to the serialized representation.
    array = np.asarray(value, dtype=np.float32).astype(np.float64)
    if array.shape != (width,) or not np.all(np.isfinite(array)):
        raise ValueError(f"recovery snapshot {name} must contain {width} finite values")
    return array


def classify_recovery_posture(
    *,
    root_quaternion_wxyz: NDArray[np.floating[Any]] | Sequence[float],
    pelvis_height_m: float,
    root_linear_speed_mps: float,
    root_angular_speed_rad_s: float,
    left_foot_supported: bool,
    right_foot_supported: bool,
) -> RecoveryPostureCluster:
    """Classify one causal body state without using a reference trajectory.

    The cluster intentionally routes high-momentum states before geometric
    posture.  A visually side-lying body that is still airborne or spinning
    is not yet a safe entry point for a contact-rich get-up expert.
    """

    quaternion = _finite_vector(
        root_quaternion_wxyz,
        width=4,
        name="root quaternion",
    )
    scalars = (pelvis_height_m, root_linear_speed_mps, root_angular_speed_rad_s)
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("recovery snapshot posture scalars must be finite")
    if pelvis_height_m < 0.0 or root_linear_speed_mps < 0.0 or root_angular_speed_rad_s < 0.0:
        raise ValueError("recovery snapshot posture scalars must be non-negative")
    norm = float(np.linalg.norm(quaternion))
    if not 0.95 <= norm <= 1.05:
        raise ValueError("recovery snapshot root quaternion must be normalized")
    w, x, y, z = quaternion / norm
    upright = 1.0 - 2.0 * (x * x + y * y)
    # World-z projections of the body x (forward) and y (left) axes.
    forward_up = 2.0 * (x * z - w * y)
    lateral_up = 2.0 * (y * z + w * x)
    bilateral = bool(left_foot_supported and right_foot_supported)
    any_foot = bool(left_foot_supported or right_foot_supported)

    if (
        pelvis_height_m >= 0.68
        and upright >= 0.85
        and root_linear_speed_mps <= 0.50
        and root_angular_speed_rad_s <= 1.00
        and bilateral
    ):
        return "STANDING"
    if (
        root_linear_speed_mps > 1.00
        or root_angular_speed_rad_s > 2.00
        or (pelvis_height_m > 0.38 and not any_foot)
    ):
        return "AIRBORNE_OR_HIGH_MOMENTUM"
    if pelvis_height_m >= 0.42 or (upright > 0.25 and any_foot):
        return "KNEELING_OR_SUPPORTED"
    if abs(lateral_up) >= 0.55:
        # G1 body +y points left.  If +y points upward, its right side is down.
        return "RIGHT_SIDE" if lateral_up > 0.0 else "LEFT_SIDE"
    if abs(forward_up) >= 0.55:
        # Body +x points forward.  +x down is face/front down (prone).
        return "SUPINE" if forward_up > 0.0 else "PRONE"
    return "AMBIGUOUS_FALLEN"


@dataclass(frozen=True)
class RecoverySnapshot:
    """One simulator-confirmed post-event recovery state."""

    episode_seed: int
    environment_index: int
    control_step: int
    stage: RecoveryStage
    save_kind: Literal["HAND", "BODY"]
    posture_cluster: RecoveryPostureCluster
    qpos: NDArray[np.float64]
    qvel: NDArray[np.float64]
    applied_action: NDArray[np.float64]
    ball_position_m: NDArray[np.float64]
    ball_velocity_mps: NDArray[np.float64]
    target_position_m: NDArray[np.float64]
    left_foot_supported: bool
    right_foot_supported: bool
    failed: bool
    body_hash: str
    physics_scene_hash: str
    source_policy_hash: str
    source_config_hash: str
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.recovery_snapshot.v1"

    def __post_init__(self) -> None:
        if (
            self.episode_seed < 0
            or self.environment_index < 0
            or self.control_step < 0
            or self.stage not in _STAGES
            or self.save_kind not in {"HAND", "BODY"}
            or self.posture_cluster not in _CLUSTERS
            or not all(
                _valid_hash(value)
                for value in (
                    self.body_hash,
                    self.physics_scene_hash,
                    self.source_policy_hash,
                    self.source_config_hash,
                )
            )
            or not isinstance(self.left_foot_supported, bool)
            or not isinstance(self.right_foot_supported, bool)
            or not isinstance(self.failed, bool)
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery snapshot contract is invalid")
        object.__setattr__(self, "qpos", _finite_vector(self.qpos, width=36, name="qpos"))
        object.__setattr__(self, "qvel", _finite_vector(self.qvel, width=35, name="qvel"))
        action = np.asarray(self.applied_action, dtype=np.float32).astype(np.float64)
        if action.ndim != 1 or not 1 <= action.size <= 64 or not np.all(np.isfinite(action)):
            raise ValueError("recovery snapshot applied action is invalid")
        object.__setattr__(self, "applied_action", action)
        for name in ("ball_position_m", "ball_velocity_mps", "target_position_m"):
            object.__setattr__(
                self,
                name,
                _finite_vector(getattr(self, name), width=3, name=name),
            )

    @property
    def snapshot_hash(self) -> str:
        payload = self.metadata()
        payload.update(
            {
                "qpos": self.qpos.tolist(),
                "qvel": self.qvel.tolist(),
                "applied_action": self.applied_action.tolist(),
                "ball_position_m": self.ball_position_m.tolist(),
                "ball_velocity_mps": self.ball_velocity_mps.tolist(),
                "target_position_m": self.target_position_m.tolist(),
            }
        )
        return str(hash_json(payload))

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "qpos",
            "qvel",
            "applied_action",
            "ball_position_m",
            "ball_velocity_mps",
            "target_position_m",
        ):
            payload.pop(key)
        return payload


def write_recovery_snapshot_corpus(
    *,
    snapshots: Sequence[RecoverySnapshot],
    output_dir: Path,
    corpus_name: str = "recovery-snapshots",
) -> dict[str, Any]:
    """Atomically write a pickle-free archive and its content-bound manifest."""

    if not snapshots:
        raise ValueError("recovery snapshot corpus cannot be empty")
    if not corpus_name or len(corpus_name) > 64 or not corpus_name.replace("-", "").isalnum():
        raise ValueError("recovery snapshot corpus name is invalid")
    bindings = {
        (
            item.body_hash,
            item.physics_scene_hash,
            item.source_policy_hash,
            item.source_config_hash,
            item.applied_action.size,
        )
        for item in snapshots
    }
    if len(bindings) != 1:
        raise ValueError("recovery snapshot corpus mixes incompatible source contracts")
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / f"{corpus_name}.npz"
    archive_tmp = destination / f".{corpus_name}.npz.tmp"
    arrays = {
        "qpos": np.stack([item.qpos for item in snapshots]).astype(np.float32),
        "qvel": np.stack([item.qvel for item in snapshots]).astype(np.float32),
        "applied_action": np.stack([item.applied_action for item in snapshots]).astype(np.float32),
        "ball_position_m": np.stack([item.ball_position_m for item in snapshots]).astype(
            np.float32
        ),
        "ball_velocity_mps": np.stack([item.ball_velocity_mps for item in snapshots]).astype(
            np.float32
        ),
        "target_position_m": np.stack([item.target_position_m for item in snapshots]).astype(
            np.float32
        ),
    }
    with archive_tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
    archive_tmp.replace(archive_path)
    archive_hash = hash_bytes(archive_path.read_bytes())
    rows = []
    for index, snapshot in enumerate(snapshots):
        row = snapshot.metadata()
        row["archive_row"] = index
        row["snapshot_hash"] = snapshot.snapshot_hash
        rows.append(row)
    body_hash, scene_hash, policy_hash, config_hash, action_width = next(iter(bindings))
    cluster_counts = {
        cluster: sum(item.posture_cluster == cluster for item in snapshots)
        for cluster in sorted(_CLUSTERS)
        if any(item.posture_cluster == cluster for item in snapshots)
    }
    stage_counts = {
        stage: sum(item.stage == stage for item in snapshots)
        for stage in sorted(_STAGES)
        if any(item.stage == stage for item in snapshots)
    }
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw.recovery_snapshot_corpus.v1",
        "corpus_name": corpus_name,
        "archive": archive_path.name,
        "archive_hash": archive_hash,
        "snapshot_count": len(snapshots),
        "action_width": action_width,
        "body_hash": body_hash,
        "physics_scene_hash": scene_hash,
        "source_policy_hash": policy_hash,
        "source_config_hash": config_hash,
        "cluster_counts": cluster_counts,
        "stage_counts": stage_counts,
        "rows": rows,
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
    }
    manifest["corpus_hash"] = hash_json(manifest)
    manifest_path = destination / f"{corpus_name}.json"
    manifest_tmp = destination / f".{corpus_name}.json.tmp"
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest_tmp.replace(manifest_path)
    return manifest


def load_recovery_snapshot_corpus(manifest_path: Path) -> tuple[RecoverySnapshot, ...]:
    """Load and fully verify a recovery corpus before it can seed training."""

    resolved = manifest_path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    declared_hash = payload.pop("corpus_hash", None)
    if (
        payload.get("schema_version") != "rosclaw.recovery_snapshot_corpus.v1"
        or declared_hash != hash_json(payload)
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_authorized") is not False
    ):
        raise ValueError("recovery snapshot corpus manifest is invalid")
    archive_path = resolved.parent / str(payload["archive"])
    if hash_bytes(archive_path.read_bytes()) != payload.get("archive_hash"):
        raise ValueError("recovery snapshot corpus archive hash mismatch")
    required = {
        "qpos",
        "qvel",
        "applied_action",
        "ball_position_m",
        "ball_velocity_mps",
        "target_position_m",
    }
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("recovery snapshot corpus arrays are incomplete")
        arrays = {name: np.asarray(archive[name], dtype=np.float64) for name in required}
    count = int(payload.get("snapshot_count", -1))
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("recovery snapshot corpus row count is invalid")
    if any(value.shape[0] != count or not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("recovery snapshot corpus array shape or finiteness is invalid")
    snapshots: list[RecoverySnapshot] = []
    for index, raw_row in enumerate(rows):
        row = dict(raw_row)
        snapshot_hash = row.pop("snapshot_hash", None)
        if row.pop("archive_row", None) != index:
            raise ValueError("recovery snapshot corpus row order is invalid")
        snapshot = RecoverySnapshot(
            **row,
            qpos=arrays["qpos"][index],
            qvel=arrays["qvel"][index],
            applied_action=arrays["applied_action"][index],
            ball_position_m=arrays["ball_position_m"][index],
            ball_velocity_mps=arrays["ball_velocity_mps"][index],
            target_position_m=arrays["target_position_m"][index],
        )
        # The archive stores float32 deliberately; bind the hash to that exact
        # serialized precision rather than a pre-serialization float64 input.
        if snapshot.snapshot_hash != snapshot_hash:
            raise ValueError("recovery snapshot row hash mismatch")
        snapshots.append(snapshot)
    return tuple(snapshots)


__all__ = [
    "RecoveryPostureCluster",
    "RecoverySnapshot",
    "RecoveryStage",
    "classify_recovery_posture",
    "load_recovery_snapshot_corpus",
    "write_recovery_snapshot_corpus",
]
