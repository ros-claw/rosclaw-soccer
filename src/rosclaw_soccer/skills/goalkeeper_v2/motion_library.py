"""Auditable, position-conditioned motion library for Goalkeeper V2.

MotionDecode contains useful G1-retargeted movement, but it does not label
goalkeeper dives.  This module therefore preserves the distinction between a
true goalkeeper demonstration and a proxy (catch, lateral gait, recovery).
Proxy clips may bootstrap a train-only prior; they cannot support a claim that
the deployed actor learned a human goalkeeper motion.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_json

_MOTIONDECODE_REVISION = "f71451a3e3285e83f11fe8738fc1d4750cab84f2"
_EXPECTED_COLUMNS = (
    "root_pos_x(m)",
    "root_pos_y(m)",
    "root_pos_z(m)",
    "root_rot_w",
    "root_rot_x",
    "root_rot_y",
    "root_rot_z",
    *(f"dof_{name}(rad)" for name in G1_DDS_JOINT_NAMES),
)
_MAX_SOURCE_BYTES = 64 * 1024 * 1024


class GoalkeeperMotionFamily(StrEnum):
    READY = "ready"
    SPLIT_STEP = "split_step"
    SHUFFLE_LEFT = "shuffle_left"
    SHUFFLE_RIGHT = "shuffle_right"
    LOW_SAVE_LEFT = "low_save_left"
    LOW_SAVE_RIGHT = "low_save_right"
    HIGH_REACH_LEFT = "high_reach_left"
    HIGH_REACH_RIGHT = "high_reach_right"
    CENTER_BLOCK = "center_block"
    RECOVERY = "recovery"


_CONDITION_TO_FAMILY = {
    ("ready", "center"): GoalkeeperMotionFamily.READY,
    ("shuffle", "upper_left"): GoalkeeperMotionFamily.SHUFFLE_LEFT,
    ("shuffle", "lower_left"): GoalkeeperMotionFamily.SHUFFLE_LEFT,
    ("shuffle", "upper_right"): GoalkeeperMotionFamily.SHUFFLE_RIGHT,
    ("shuffle", "lower_right"): GoalkeeperMotionFamily.SHUFFLE_RIGHT,
    ("shuffle", "center"): GoalkeeperMotionFamily.SPLIT_STEP,
    ("save", "upper_left"): GoalkeeperMotionFamily.HIGH_REACH_LEFT,
    ("save", "upper_right"): GoalkeeperMotionFamily.HIGH_REACH_RIGHT,
    ("save", "lower_left"): GoalkeeperMotionFamily.LOW_SAVE_LEFT,
    ("save", "lower_right"): GoalkeeperMotionFamily.LOW_SAVE_RIGHT,
    ("save", "center"): GoalkeeperMotionFamily.CENTER_BLOCK,
    ("landing", "upper_left"): GoalkeeperMotionFamily.HIGH_REACH_LEFT,
    ("landing", "upper_right"): GoalkeeperMotionFamily.HIGH_REACH_RIGHT,
    ("landing", "lower_left"): GoalkeeperMotionFamily.LOW_SAVE_LEFT,
    ("landing", "lower_right"): GoalkeeperMotionFamily.LOW_SAVE_RIGHT,
    ("landing", "center"): GoalkeeperMotionFamily.CENTER_BLOCK,
    ("recovery", "center"): GoalkeeperMotionFamily.RECOVERY,
}


@dataclass(frozen=True)
class GoalkeeperMotionClip:
    clip_id: str
    family: GoalkeeperMotionFamily
    source_relative_path: str
    source_hash: str
    license_terms_hash: str
    source_fps: float
    frame_count: int
    segment_start_frame: int
    segment_end_frame: int
    mirrored_left_right: bool
    proxy_kind: str
    quality_score: float
    root_height_min_m: float
    root_height_max_m: float
    joint_velocity_rms_rad_s: float
    recovery_posture: str | None = None
    schema_version: str = "rosclaw_soccer.goalkeeper_motion_clip.v2"

    def __post_init__(self) -> None:
        if not self.clip_id or self.clip_id != self.clip_id.strip():
            raise ValueError("goalkeeper motion clip requires a stable id")
        if (
            Path(self.source_relative_path).is_absolute()
            or ".." in Path(self.source_relative_path).parts
        ):
            raise ValueError("goalkeeper motion source must be a safe relative path")
        for value in (self.source_hash, self.license_terms_hash):
            if not value.startswith("sha256:"):
                raise ValueError("goalkeeper motion provenance requires content hashes")
        if not 30.0 <= self.source_fps <= 240.0:
            raise ValueError("goalkeeper motion source fps is outside [30, 240]")
        if not (0 <= self.segment_start_frame < self.segment_end_frame <= self.frame_count):
            raise ValueError("goalkeeper motion segment is outside its source clip")
        values = (
            self.quality_score,
            self.root_height_min_m,
            self.root_height_max_m,
            self.joint_velocity_rms_rad_s,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("goalkeeper motion quality metrics must be finite and non-negative")
        if self.proxy_kind not in {
            "balance_proxy",
            "lateral_gait_proxy",
            "catch_proxy",
            "ground_recovery_proxy",
            "goalkeeper_demonstration",
        }:
            raise ValueError("goalkeeper motion clip has an unknown provenance kind")
        if self.recovery_posture not in {
            None,
            "LEFT_SIDE",
            "RIGHT_SIDE",
            "PRONE",
            "SUPINE",
            "AMBIGUOUS_FALLEN",
        }:
            raise ValueError("goalkeeper recovery clip posture is invalid")
        if (self.family is GoalkeeperMotionFamily.RECOVERY) != (
            self.recovery_posture is not None
        ):
            raise ValueError("goalkeeper recovery posture must match the recovery family")

    @property
    def true_goalkeeper_demonstration(self) -> bool:
        return self.proxy_kind == "goalkeeper_demonstration"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["family"] = self.family.value
        return value


@dataclass(frozen=True)
class GoalkeeperMotionLibrary:
    library_id: str
    dataset_id: str
    dataset_revision: str
    license_terms_hash: str
    readme_hash: str
    joint_order_hash: str
    body_hash: str
    clips: tuple[GoalkeeperMotionClip, ...]
    attribution_required: bool = True
    commercial_use_allowed: bool = False
    training_use_only: bool = True
    raw_data_embedded: bool = False
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.goalkeeper_motion_library.v2"

    def __post_init__(self) -> None:
        if self.dataset_id != "MotionDecode" or self.dataset_revision != _MOTIONDECODE_REVISION:
            raise ValueError("goalkeeper motion library requires the pinned MotionDecode revision")
        for value in (
            self.license_terms_hash,
            self.readme_hash,
            self.joint_order_hash,
            self.body_hash,
        ):
            if not value.startswith("sha256:"):
                raise ValueError("goalkeeper motion library provenance requires content hashes")
        if self.commercial_use_allowed or not self.attribution_required:
            raise ValueError("MotionDecode terms require attribution and non-commercial use")
        if (
            not self.training_use_only
            or self.raw_data_embedded
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("goalkeeper motion library must remain train-only SIM_ONLY metadata")
        if len({clip.clip_id for clip in self.clips}) != len(self.clips):
            raise ValueError("goalkeeper motion clip ids must be unique")
        missing = set(GoalkeeperMotionFamily) - {clip.family for clip in self.clips}
        if missing:
            raise ValueError(
                "goalkeeper motion library lacks families: "
                + ",".join(sorted(item.value for item in missing))
            )

    @property
    def library_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    @property
    def contains_only_proxy_motion(self) -> bool:
        return not any(clip.true_goalkeeper_demonstration for clip in self.clips)

    @property
    def human_goalkeeper_claim_allowed(self) -> bool:
        return not self.contains_only_proxy_motion

    def clips_for(self, *, task: str, region: str) -> tuple[GoalkeeperMotionClip, ...]:
        key = (task, region)
        if task in {"ready", "recovery"}:
            key = (task, "center")
        family = _CONDITION_TO_FAMILY.get(key)
        if family is None:
            raise ValueError("goalkeeper motion query has an unsupported task/region")
        selected = tuple(clip for clip in self.clips if clip.family is family)
        if not selected:
            raise ValueError("goalkeeper motion query has no qualified source clip")
        return selected

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "library_id": self.library_id,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "license_terms_hash": self.license_terms_hash,
            "readme_hash": self.readme_hash,
            "joint_order_hash": self.joint_order_hash,
            "body_hash": self.body_hash,
            "clips": [clip.to_dict() for clip in self.clips],
            "attribution_required": self.attribution_required,
            "commercial_use_allowed": self.commercial_use_allowed,
            "training_use_only": self.training_use_only,
            "raw_data_embedded": self.raw_data_embedded,
            "activation_ceiling": self.activation_ceiling,
            "contains_only_proxy_motion": self.contains_only_proxy_motion,
            "human_goalkeeper_claim_allowed": self.human_goalkeeper_claim_allowed,
        }
        if include_hash:
            value["library_hash"] = self.library_hash
        return value


@dataclass(frozen=True)
class _ClipStatistics:
    path: Path
    frame_count: int
    minimum_root_height_frame: int
    recovered_standing_frame: int
    recovery_posture: str
    root_height_min_m: float
    root_height_max_m: float
    velocity_rms_rad_s: float
    arm_excursion_rad: float
    lateral_excursion_m: float


def build_motiondecode_goalkeeper_library(
    *,
    dataset_root: Path,
    output_path: Path,
    source_checkout: Path,
    body_hash: str,
) -> GoalkeeperMotionLibrary:
    """Select bounded proxy families while preserving provenance and claim limits."""

    root = dataset_root.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("goalkeeper motion library must be outside the source checkout")
    if output.exists():
        raise ValueError("goalkeeper motion library output already exists")
    license_path = root / "LICENSE.md"
    readme_path = root / "README.md"
    license_text = _required_text(license_path)
    readme_text = _required_text(readme_path)
    license_lower = license_text.lower()
    if not all(
        phrase in license_lower
        for phrase in ("academic research", "non-commercial", "retain attribution")
    ):
        raise ValueError("MotionDecode non-commercial attribution terms are not verifiable")
    if "120 hz" not in readme_text.lower() or "unitree g1" not in readme_text.lower():
        raise ValueError("MotionDecode README lacks the expected G1/120 Hz declaration")
    license_hash = _file_hash(license_path)
    samples = root / "samples"
    balance = _rank_sources(samples, "Single_Leg_Standing", limit=2)
    lateral = _rank_sources(samples, "Lateral_Walking", limit=2)
    catching = _rank_sources(samples, "Catching_Action", limit=12)
    recovery = (
        _rank_sources(samples, "Lie_Down_to_Stand", limit=24)
        + _rank_sources(samples, "Prone_to_Stand", limit=24)
        + _rank_sources(samples, "Fall_and_Fall_Recovery", limit=24)
    )
    if not all((balance, lateral, catching, recovery)):
        raise ValueError("MotionDecode lacks a required goalkeeper proxy family")

    static_ready = min(balance, key=lambda item: item.velocity_rms_rad_s)
    split = max(balance, key=lambda item: item.velocity_rms_rad_s)
    shuffle = max(lateral, key=lambda item: item.velocity_rms_rad_s)
    low = min(catching, key=lambda item: (item.root_height_min_m, -item.arm_excursion_rad))
    high = max(catching, key=lambda item: (item.root_height_max_m, item.arm_excursion_rad))
    center = min(catching, key=lambda item: (item.lateral_excursion_m, -item.arm_excursion_rad))
    recovery_by_posture = {
        posture: max(
            (item for item in recovery if item.recovery_posture == posture),
            key=lambda item: item.root_height_max_m - item.root_height_min_m,
        )
        for posture in ("LEFT_SIDE", "RIGHT_SIDE", "PRONE", "SUPINE")
        if any(item.recovery_posture == posture for item in recovery)
    }
    if not {"PRONE", "SUPINE"}.issubset(recovery_by_posture):
        raise ValueError("MotionDecode lacks prone and supine recovery teachers")
    definitions = (
        (GoalkeeperMotionFamily.READY, static_ready, False, "balance_proxy", None),
        (GoalkeeperMotionFamily.SPLIT_STEP, split, False, "balance_proxy", None),
        (GoalkeeperMotionFamily.SHUFFLE_LEFT, shuffle, False, "lateral_gait_proxy", None),
        (GoalkeeperMotionFamily.SHUFFLE_RIGHT, shuffle, True, "lateral_gait_proxy", None),
        (GoalkeeperMotionFamily.LOW_SAVE_LEFT, low, False, "catch_proxy", None),
        (GoalkeeperMotionFamily.LOW_SAVE_RIGHT, low, True, "catch_proxy", None),
        (GoalkeeperMotionFamily.HIGH_REACH_LEFT, high, False, "catch_proxy", None),
        (GoalkeeperMotionFamily.HIGH_REACH_RIGHT, high, True, "catch_proxy", None),
        (GoalkeeperMotionFamily.CENTER_BLOCK, center, False, "catch_proxy", None),
        *(
            (
                GoalkeeperMotionFamily.RECOVERY,
                recovery_by_posture[posture],
                False,
                "ground_recovery_proxy",
                posture,
            )
            for posture in sorted(recovery_by_posture)
        ),
    )
    clips = tuple(
        _clip_from_statistics(
            root=root,
            statistics=statistics,
            family=family,
            mirrored=mirrored,
            proxy_kind=proxy_kind,
            posture=posture,
            license_hash=license_hash,
        )
        for family, statistics, mirrored, proxy_kind, posture in definitions
    )
    library = GoalkeeperMotionLibrary(
        library_id="soccer.goalkeeper.motiondecode_proxy.v2",
        dataset_id="MotionDecode",
        dataset_revision=_MOTIONDECODE_REVISION,
        license_terms_hash=license_hash,
        readme_hash=_file_hash(readme_path),
        joint_order_hash=hash_json(G1_DDS_JOINT_NAMES),
        body_hash=body_hash,
        clips=clips,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(library.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return library


def load_goalkeeper_motion_library(
    path: Path,
    *,
    dataset_root: Path | None = None,
) -> GoalkeeperMotionLibrary:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    claimed_hash = str(payload.pop("library_hash", ""))
    payload.pop("contains_only_proxy_motion", None)
    payload.pop("human_goalkeeper_claim_allowed", None)
    try:
        payload["clips"] = tuple(
            GoalkeeperMotionClip(**{**item, "family": GoalkeeperMotionFamily(item["family"])})
            for item in payload["clips"]
        )
        library = GoalkeeperMotionLibrary(**payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("goalkeeper motion library payload is invalid") from exc
    if claimed_hash != library.library_hash:
        raise ValueError("goalkeeper motion library content hash mismatch")
    if dataset_root is not None:
        root = dataset_root.expanduser().resolve()
        if _file_hash(root / "LICENSE.md") != library.license_terms_hash:
            raise ValueError("goalkeeper motion library license terms changed")
        for clip in library.clips:
            if _file_hash(root / clip.source_relative_path) != clip.source_hash:
                raise ValueError("goalkeeper motion source content changed")
    return library


def load_motion_clip_frames(
    *,
    dataset_root: Path,
    clip: GoalkeeperMotionClip,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load one verified segment as canonical q/dq, applying an explicit mirror."""

    path = dataset_root.expanduser().resolve() / clip.source_relative_path
    if _file_hash(path) != clip.source_hash:
        raise ValueError("goalkeeper motion source content changed")
    values = _read_source(path)
    q = values[clip.segment_start_frame : clip.segment_end_frame, 7:].copy()
    if clip.mirrored_left_right:
        q = _mirror_g1_joints(q)
    dq = np.gradient(q, 1.0 / clip.source_fps, axis=0)
    return np.asarray(q, dtype=np.float64), np.asarray(dq, dtype=np.float64)


def _rank_sources(samples: Path, marker: str, *, limit: int) -> tuple[_ClipStatistics, ...]:
    paths = sorted(path for path in samples.rglob("*.csv") if marker in str(path))
    if not paths:
        return ()
    # Reading every 1000-hour category would turn manifest construction into
    # accidental training.  A stable prefix is enough for deterministic data
    # qualification; the selected file hash remains authoritative.
    statistics = tuple(_source_statistics(path) for path in paths[: max(24, limit * 4)])
    ranked = sorted(
        statistics,
        key=lambda item: (
            item.arm_excursion_rad + item.velocity_rms_rad_s,
            str(item.path),
        ),
        reverse=True,
    )
    return tuple(ranked[:limit])


def _source_statistics(path: Path) -> _ClipStatistics:
    values = _read_source(path)
    q = values[:, 7:]
    velocity = np.diff(q, axis=0) * 120.0
    root_xy = values[:, :2]
    minimum_height_frame = int(np.argmin(values[:, 2]))
    quaternion = values[:, 3:7]
    upright = 2.0 * (np.square(quaternion[:, 0]) + np.square(quaternion[:, 3])) - 1.0
    stable = (values[:, 2] >= 0.68) & (upright >= 0.80)
    stable[:minimum_height_frame] = False
    hold_frames = min(30, max(2, len(values) // 10))
    recovered_standing_frame = len(values) - 1
    for frame in range(minimum_height_frame, len(values) - hold_frames + 1):
        if bool(np.all(stable[frame : frame + hold_frames])):
            recovered_standing_frame = frame
            break
    w, x, y, z = quaternion[minimum_height_frame]
    forward_up = 2.0 * (x * z - w * y)
    lateral_up = 2.0 * (y * z + w * x)
    if abs(lateral_up) >= 0.55:
        recovery_posture = "RIGHT_SIDE" if lateral_up > 0.0 else "LEFT_SIDE"
    elif abs(forward_up) >= 0.55:
        recovery_posture = "SUPINE" if forward_up > 0.0 else "PRONE"
    else:
        recovery_posture = "AMBIGUOUS_FALLEN"
    return _ClipStatistics(
        path=path,
        frame_count=len(values),
        minimum_root_height_frame=minimum_height_frame,
        recovered_standing_frame=recovered_standing_frame,
        recovery_posture=recovery_posture,
        root_height_min_m=float(np.min(values[:, 2])),
        root_height_max_m=float(np.max(values[:, 2])),
        velocity_rms_rad_s=float(np.sqrt(np.mean(np.square(velocity)))),
        arm_excursion_rad=float(np.max(np.abs(q[:, 15:] - q[:1, 15:]))),
        lateral_excursion_m=float(np.max(root_xy[:, 1]) - np.min(root_xy[:, 1])),
    )


def _clip_from_statistics(
    *,
    root: Path,
    statistics: _ClipStatistics,
    family: GoalkeeperMotionFamily,
    mirrored: bool,
    proxy_kind: str,
    posture: str | None,
    license_hash: str,
) -> GoalkeeperMotionClip:
    if family is GoalkeeperMotionFamily.RECOVERY:
        # MotionDecode recovery files commonly contain standing context before
        # the fall and after the recovery.  A fixed tail window therefore
        # teaches an already-standing pose instead of the get-up transition.
        # Anchor the segment on causal body events and retain a short terminal
        # hold for downstream handoff learning.
        start = min(statistics.minimum_root_height_frame, statistics.frame_count - 2)
        end = min(
            statistics.frame_count,
            max(start + 2, statistics.recovered_standing_frame + 60),
        )
    else:
        window = min(statistics.frame_count, 240)
        start = max(0, (statistics.frame_count - window) // 2)
        end = start + window
    relative = str(statistics.path.relative_to(root))
    quality = statistics.arm_excursion_rad + 0.25 * statistics.velocity_rms_rad_s
    return GoalkeeperMotionClip(
        clip_id=(
            f"{family.value}.{posture.lower()}.v2"
            if posture is not None
            else f"{family.value}.{'mirror' if mirrored else 'source'}.v1"
        ),
        family=family,
        source_relative_path=relative,
        source_hash=_file_hash(statistics.path),
        license_terms_hash=license_hash,
        source_fps=120.0,
        frame_count=statistics.frame_count,
        segment_start_frame=start,
        segment_end_frame=end,
        mirrored_left_right=mirrored,
        proxy_kind=proxy_kind,
        quality_score=quality,
        root_height_min_m=statistics.root_height_min_m,
        root_height_max_m=statistics.root_height_max_m,
        joint_velocity_rms_rad_s=statistics.velocity_rms_rad_s,
        recovery_posture=posture,
    )


def _mirror_g1_joints(q: NDArray[np.float64]) -> NDArray[np.float64]:
    result = q.copy()
    pairs = tuple(zip(range(0, 6), range(6, 12), strict=True)) + tuple(
        zip(range(15, 22), range(22, 29), strict=True)
    )
    for left, right in pairs:
        result[:, left], result[:, right] = q[:, right], q[:, left]
    # Roll/yaw axes change sign under sagittal reflection.
    sign_flip = (1, 2, 5, 7, 8, 11, 12, 13, 16, 17, 19, 21, 23, 24, 26, 28)
    result[:, sign_flip] *= -1.0
    return result


def _read_source(path: Path) -> NDArray[np.float64]:
    if not path.is_file() or path.stat().st_size > _MAX_SOURCE_BYTES:
        raise ValueError("goalkeeper motion source is missing or oversized")
    header = tuple(path.open(encoding="utf-8").readline().strip().split(","))
    if header != _EXPECTED_COLUMNS:
        raise ValueError("goalkeeper motion source does not match the canonical G1 joint order")
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(_EXPECTED_COLUMNS) or len(values) < 4:
        raise ValueError("goalkeeper motion source shape is invalid")
    if not np.all(np.isfinite(values)):
        raise ValueError("goalkeeper motion source contains non-finite values")
    return np.asarray(values, dtype=np.float64)


def _required_text(path: Path) -> str:
    if not path.is_file() or not (text := path.read_text(encoding="utf-8").strip()):
        raise ValueError(f"required dataset material is missing or empty: {path.name}")
    return text


def _file_hash(path: Path) -> str:
    if not path.is_file() or path.stat().st_size > _MAX_SOURCE_BYTES:
        raise ValueError("bounded source hash requires a readable file under 64 MiB")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "GoalkeeperMotionClip",
    "GoalkeeperMotionFamily",
    "GoalkeeperMotionLibrary",
    "build_motiondecode_goalkeeper_library",
    "load_goalkeeper_motion_library",
    "load_motion_clip_frames",
]
