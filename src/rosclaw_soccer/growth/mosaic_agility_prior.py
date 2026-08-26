"""Qualified MOSAIC motion teachers for generic humanoid agility research."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.mosaic_g1_contract import (
    MOSAIC_G1_SEMANTIC_CONTRACT_HASH,
    canonicalize_mosaic_g1_joints,
)
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_json

_ALLOWED_SKILLS = {
    "SE52": "agility_ladder",
    "SE56": "shadow_boxing",
    "SE57": "baseball_swing",
    "SE63": "soccer_taps",
}


@dataclass(frozen=True)
class G1MosaicAgilityEvent:
    skill_id: str
    skill_name: str
    relative_path: str
    source_hash: str
    fps: float
    frame_count: int
    selected_center_frame: int
    waist_velocity_rms_rad_s: float
    arm_velocity_rms_rad_s: float
    root_planar_speed_mps: float

    def __post_init__(self) -> None:
        if (
            self.skill_id not in _ALLOWED_SKILLS
            or self.skill_name != _ALLOWED_SKILLS[self.skill_id]
        ):
            raise ValueError("MOSAIC agility event skill identity is invalid")
        if not self.source_hash.startswith("sha256:") or not self.relative_path:
            raise ValueError("MOSAIC agility event source identity is invalid")
        values = (
            self.fps,
            self.waist_velocity_rms_rad_s,
            self.arm_velocity_rms_rad_s,
            self.root_planar_speed_mps,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("MOSAIC agility event metrics must be finite and non-negative")
        if self.frame_count < 21 or not 10 <= self.selected_center_frame < self.frame_count - 10:
            raise ValueError("MOSAIC agility event frame contract is invalid")


@dataclass(frozen=True)
class G1MosaicAgilityPrior:
    dataset_readme_hash: str
    source_partition_hash: str
    joint_order_hash: str
    joint_names: tuple[str, ...]
    reference_times_sec: tuple[float, ...]
    whole_body_velocity_reference_rad_s: tuple[tuple[float, ...], ...]
    maximum_velocity_correction_rad_s: tuple[float, ...]
    selected_events: tuple[G1MosaicAgilityEvent, ...]
    teacher_skill_id: str | None = None
    source_dataset: str = "MOSAIC"
    dataset_license: str = "CDLA-Permissive-2.0"
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.growth.g1_mosaic_agility_prior.v1"

    def __post_init__(self) -> None:
        for value in (
            self.dataset_readme_hash,
            self.source_partition_hash,
            self.joint_order_hash,
        ):
            if not value.startswith("sha256:") or len(value) != 71:
                raise ValueError("MOSAIC agility prior hashes must be sha256 content hashes")
        if self.joint_names != G1_DDS_JOINT_NAMES:
            raise ValueError("MOSAIC agility prior requires canonical G1 joint order")
        expected = (len(self.reference_times_sec), len(self.joint_names))
        velocity = np.asarray(self.whole_body_velocity_reference_rad_s, dtype=np.float64)
        maximum = np.asarray(self.maximum_velocity_correction_rad_s, dtype=np.float64)
        if velocity.shape != expected or not np.isfinite(velocity).all():
            raise ValueError("MOSAIC agility velocity teacher is malformed")
        if maximum.shape != (29,) or not np.isfinite(maximum).all() or np.any(maximum <= 0.0):
            raise ValueError("MOSAIC agility velocity bounds are malformed")
        if len(self.selected_events) != len(_ALLOWED_SKILLS):
            raise ValueError("MOSAIC agility prior requires one event per declared skill")
        if self.teacher_skill_id is not None and self.teacher_skill_id not in _ALLOWED_SKILLS:
            raise ValueError("MOSAIC agility teacher skill is not declared")
        if self.schema_version == "rosclaw.growth.g1_mosaic_agility_prior.v1":
            if self.teacher_skill_id is not None:
                raise ValueError("MOSAIC v1 prior requires the cross-skill median teacher")
        elif self.schema_version in {
            "rosclaw.growth.g1_mosaic_agility_prior.v2",
            "rosclaw.growth.g1_mosaic_agility_prior.v3",
        }:
            if self.teacher_skill_id is None:
                raise ValueError("MOSAIC v2 prior requires a semantic teacher skill")
        else:
            raise ValueError("MOSAIC agility prior schema is unsupported")
        if self.activation_ceiling != "SIM_ONLY" or self.promotion_authorized:
            raise ValueError("MOSAIC agility prior must remain SIM_ONLY and unpromoted")

    @property
    def prior_hash(self) -> str:
        payload = asdict(self)
        if self.schema_version == "rosclaw.growth.g1_mosaic_agility_prior.v1":
            payload.pop("teacher_skill_id")
        return str(hash_json(payload))

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "prior_hash": self.prior_hash}


def load_g1_mosaic_agility_prior(path: Path) -> G1MosaicAgilityPrior:
    """Load one no-pickle, content-bound MOSAIC agility artifact."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file() or not 1 <= resolved.stat().st_size <= 8 * 1024 * 1024:
        raise ValueError("MOSAIC agility prior is missing, empty, or too large")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("MOSAIC agility prior root must be an object")
    claimed_hash = payload.pop("prior_hash", None)
    try:
        payload["joint_names"] = tuple(payload["joint_names"])
        payload["reference_times_sec"] = tuple(payload["reference_times_sec"])
        payload["whole_body_velocity_reference_rad_s"] = tuple(
            tuple(row) for row in payload["whole_body_velocity_reference_rad_s"]
        )
        payload["maximum_velocity_correction_rad_s"] = tuple(
            payload["maximum_velocity_correction_rad_s"]
        )
        payload["selected_events"] = tuple(
            G1MosaicAgilityEvent(**event) for event in payload["selected_events"]
        )
        prior = G1MosaicAgilityPrior(**payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("MOSAIC agility prior payload is invalid") from exc
    if claimed_hash != prior.prior_hash:
        raise ValueError("MOSAIC agility prior content hash mismatch")
    return prior


def blend_g1_mosaic_agility_velocity(
    *,
    target_velocity: NDArray[np.float64],
    prior: G1MosaicAgilityPrior,
    policy_frame: int,
    contact_policy_frame: int,
    control_dt_sec: float,
    blend: float,
    joint_scales: tuple[float, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """Apply bounded MOSAIC velocity only through the existing PD target."""

    if target_velocity.shape != (29,) or not np.isfinite(target_velocity).all():
        raise ValueError("MOSAIC agility target velocity must contain 29 finite joints")
    if not 0.0 <= blend <= 0.10 or not math.isfinite(blend):
        raise ValueError("MOSAIC agility blend must be in [0, 0.10]")
    if len(joint_scales) != 29 or not all(
        math.isfinite(value) and 0.0 <= value <= 1.0 for value in joint_scales
    ):
        raise ValueError("MOSAIC agility joint scales must contain 29 values in [0, 1]")
    if control_dt_sec <= 0.0 or not math.isfinite(control_dt_sec):
        raise ValueError("MOSAIC agility control clock must be positive")
    delta = np.zeros(29, dtype=np.float64)
    relative_time = (policy_frame - contact_policy_frame) * control_dt_sec
    times = np.asarray(prior.reference_times_sec, dtype=np.float64)
    if blend == 0.0 or relative_time < times[0] or relative_time > times[-1]:
        return target_velocity.copy(), delta, False
    reference = np.asarray(prior.whole_body_velocity_reference_rad_s, dtype=np.float64)
    desired = np.asarray(
        [
            np.interp(relative_time, times, reference[:, joint])
            for joint in range(reference.shape[1])
        ],
        dtype=np.float64,
    )
    progress = (relative_time - times[0]) / max(times[-1] - times[0], 1e-9)
    envelope = math.sin(math.pi * min(1.0, max(0.0, progress))) ** 2
    maximum = np.asarray(prior.maximum_velocity_correction_rad_s, dtype=np.float64)
    bounded = np.clip(desired - target_velocity, -maximum, maximum)
    delta = blend * envelope * bounded * np.asarray(joint_scales, dtype=np.float64)
    return target_velocity + delta, delta, bool(np.any(np.abs(delta) > 1e-12))


def blend_g1_mosaic_agility_target(
    *,
    target: NDArray[np.float64],
    prior: G1MosaicAgilityPrior,
    policy_frame: int,
    contact_policy_frame: int,
    control_dt_sec: float,
    blend: float,
    joint_scales: tuple[float, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """Apply a bounded, endpoint-neutral MOSAIC pose residual through PD."""

    if target.shape != (29,) or not np.isfinite(target).all():
        raise ValueError("MOSAIC agility target must contain 29 finite joints")
    if not 0.0 <= blend <= 0.50 or not math.isfinite(blend):
        raise ValueError("MOSAIC agility position blend must be in [0, 0.50]")
    if len(joint_scales) != 29 or not all(
        math.isfinite(value) and 0.0 <= value <= 1.0 for value in joint_scales
    ):
        raise ValueError("MOSAIC agility joint scales must contain 29 values in [0, 1]")
    if control_dt_sec <= 0.0 or not math.isfinite(control_dt_sec):
        raise ValueError("MOSAIC agility control clock must be positive")
    delta = np.zeros(29, dtype=np.float64)
    relative_time = (policy_frame - contact_policy_frame) * control_dt_sec
    times = np.asarray(prior.reference_times_sec, dtype=np.float64)
    if blend == 0.0 or relative_time < times[0] or relative_time > times[-1]:
        return target.copy(), delta, False
    velocity = np.asarray(prior.whole_body_velocity_reference_rad_s, dtype=np.float64)
    dt = np.diff(times)[:, None]
    integrated = np.vstack(
        (
            np.zeros((1, velocity.shape[1]), dtype=np.float64),
            np.cumsum(0.5 * (velocity[:-1] + velocity[1:]) * dt, axis=0),
        )
    )
    progress = (times - times[0]) / max(times[-1] - times[0], 1e-9)
    endpoint_neutral = integrated - progress[:, None] * integrated[-1]
    desired = np.asarray(
        [
            np.interp(relative_time, times, endpoint_neutral[:, joint])
            for joint in range(endpoint_neutral.shape[1])
        ],
        dtype=np.float64,
    )
    bounded = np.clip(desired, -0.25, 0.25)
    delta = blend * bounded * np.asarray(joint_scales, dtype=np.float64)
    return target + delta, delta, bool(np.any(np.abs(delta) > 1e-12))


def derive_g1_mosaic_agility_prior(
    *,
    mosaic_root: Path,
    output_path: Path,
    source_checkout: Path,
    joint_order_contract: Path | None = None,
    teacher_skill_id: str | None = None,
) -> G1MosaicAgilityPrior:
    """Distil four high-motion windows without accessing a test metric."""

    root = mosaic_root.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    contract = None if joint_order_contract is None else joint_order_contract.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("MOSAIC agility prior must be outside the source checkout")
    if output.exists():
        raise ValueError("MOSAIC agility prior output already exists")
    if teacher_skill_id is not None and teacher_skill_id not in _ALLOWED_SKILLS:
        raise ValueError("MOSAIC agility teacher skill is not declared")
    readme = root / "README.md"
    if (
        not readme.is_file()
        or "license: cdla-permissive-2.0" not in readme.read_text(encoding="utf-8").lower()
    ):
        raise ValueError("MOSAIC README with CDLA-Permissive-2.0 declaration is required")
    if contract is not None:
        contract_text = contract.read_text(encoding="utf-8")
        if not all(f'"{name}"' in contract_text for name in G1_DDS_JOINT_NAMES):
            raise ValueError("MOSAIC conversion contract lacks canonical G1 joint names")
    times = tuple(float(value) for value in np.linspace(-0.20, 0.20, 21))
    sequences: list[NDArray[np.float64]] = []
    events: list[G1MosaicAgilityEvent] = []
    source_hashes: dict[str, str] = {}
    for skill_id, skill_name in _ALLOWED_SKILLS.items():
        path = root / "G1" / "optical_mocap" / f"g1_{skill_id}_stageii.npz"
        if not path.is_file():
            raise ValueError(f"MOSAIC agility source is missing: {path}")
        with np.load(path, allow_pickle=False) as data:
            required = {"fps", "joint_pos", "joint_vel", "body_lin_vel_w"}
            if not required.issubset(data.files):
                raise ValueError(f"MOSAIC agility source lacks required arrays: {path}")
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            raw_position = np.asarray(data["joint_pos"], dtype=np.float64)
            raw_velocity = np.asarray(data["joint_vel"], dtype=np.float64)
            body_velocity = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
        frame_count = raw_position.shape[0]
        if (
            raw_position.shape != (frame_count, 29)
            or raw_velocity.shape != raw_position.shape
            or body_velocity.shape != (frame_count, 30, 3)
            or not all(
                np.isfinite(value).all()
                for value in (raw_position, raw_velocity, body_velocity)
            )
            or not 30.0 <= fps <= 240.0
        ):
            raise ValueError(f"MOSAIC agility source shape or clock is invalid: {path}")
        # Validate the source ordering even though this prior uses velocity.
        _ = canonicalize_mosaic_g1_joints(raw_position)
        velocity = canonicalize_mosaic_g1_joints(raw_velocity)
        radius = max(10, int(round(0.20 * fps)))
        energy = np.sqrt(np.mean(np.square(velocity[:, 12:]), axis=1))
        kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
        smoothed = np.convolve(energy, kernel, mode="same")
        smoothed[:radius] = -np.inf
        smoothed[-radius:] = -np.inf
        center = int(np.argmax(smoothed))
        indices = np.clip(
            np.rint(center + np.asarray(times) * fps).astype(int),
            0,
            frame_count - 1,
        )
        sequences.append(velocity[indices])
        digest = _file_hash(path)
        relative = str(path.relative_to(root))
        source_hashes[relative] = digest
        events.append(
            G1MosaicAgilityEvent(
                skill_id=skill_id,
                skill_name=skill_name,
                relative_path=relative,
                source_hash=digest,
                fps=fps,
                frame_count=frame_count,
                selected_center_frame=center,
                waist_velocity_rms_rad_s=_rms(velocity[indices, 12:15]),
                arm_velocity_rms_rad_s=_rms(velocity[indices, 15:]),
                root_planar_speed_mps=_rms(body_velocity[indices, 0, :2]),
            )
        )
    reference = (
        np.median(np.asarray(sequences), axis=0)
        if teacher_skill_id is None
        else sequences[tuple(_ALLOWED_SKILLS).index(teacher_skill_id)]
    )
    maximum = np.maximum(0.25, np.quantile(np.abs(np.asarray(sequences)), 0.90, axis=(0, 1)))
    prior = G1MosaicAgilityPrior(
        dataset_readme_hash=_file_hash(readme),
        source_partition_hash=hash_json(source_hashes),
        joint_order_hash=hash_json(
            {
                "semantic_contract_hash": MOSAIC_G1_SEMANTIC_CONTRACT_HASH,
                "external_contract_hash": None if contract is None else _file_hash(contract),
            }
        ),
        joint_names=G1_DDS_JOINT_NAMES,
        reference_times_sec=times,
        whole_body_velocity_reference_rad_s=tuple(
            tuple(float(value) for value in row) for row in reference
        ),
        maximum_velocity_correction_rad_s=tuple(float(value) for value in maximum),
        selected_events=tuple(events),
        teacher_skill_id=teacher_skill_id,
        schema_version=(
            "rosclaw.growth.g1_mosaic_agility_prior.v1"
            if teacher_skill_id is None
            else "rosclaw.growth.g1_mosaic_agility_prior.v3"
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(prior.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return prior


def _rms(values: NDArray[np.generic]) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "G1MosaicAgilityEvent",
    "G1MosaicAgilityPrior",
    "blend_g1_mosaic_agility_target",
    "blend_g1_mosaic_agility_velocity",
    "derive_g1_mosaic_agility_prior",
    "load_g1_mosaic_agility_prior",
]
