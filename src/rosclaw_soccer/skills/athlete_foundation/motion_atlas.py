"""Balanced, content-addressed motion suite for athlete-foundation shootouts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.athlete_foundation.full_body_goalkeeper_motion import (
    load_full_body_goalkeeper_motion_library,
)

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")


class MotionAtlasCategory(StrEnum):
    GENERIC_ATHLETE = "generic_athlete"
    FOOTBALL = "football"
    GOALKEEPER = "goalkeeper"


@dataclass(frozen=True)
class MotionAtlasRecord:
    motion_id: str
    source_id: str
    source_path: str
    source_hash: str
    body_id: str
    category: MotionAtlasCategory
    tags: tuple[str, ...]
    frame_count: int
    frame_rate_hz: float
    segment_start_frame: int
    segment_end_frame_exclusive: int
    representation: str
    license_id: str
    attribution_required: bool
    commercial_use_allowed: bool
    schema_version: str = "rosclaw_soccer.motion_atlas_record.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("motion_id", self.motion_id),
            ("source_id", self.source_id),
            ("body_id", self.body_id),
            ("representation", self.representation),
            ("license_id", self.license_id),
        ):
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"{label} is not a normalized identifier")
        path = Path(self.source_path)
        if path.is_absolute() or ".." in path.parts or not self.source_path:
            raise ValueError("motion source path must be normalized and relative")
        if not _HASH.fullmatch(self.source_hash):
            raise ValueError("motion source hash is invalid")
        if (
            not self.tags
            or len(set(self.tags)) != len(self.tags)
            or any(not _IDENTIFIER.fullmatch(tag) for tag in self.tags)
        ):
            raise ValueError("motion tags must be non-empty unique identifiers")
        if self.frame_count < 3 or self.frame_rate_hz <= 0.0:
            raise ValueError("motion timing is invalid")
        if not 0 <= self.segment_start_frame < self.segment_end_frame_exclusive <= self.frame_count:
            raise ValueError("motion segment bounds are invalid")
        if not self.attribution_required or self.commercial_use_allowed:
            raise ValueError("S33 motion atlas is restricted to attributed research use")

    @property
    def segment_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["category"] = self.category.value
        value["tags"] = list(self.tags)
        return value


@dataclass(frozen=True)
class MotionAtlasManifest:
    records: tuple[MotionAtlasRecord, ...]
    target_counts: tuple[tuple[str, int], ...] = (
        ("generic_athlete", 20),
        ("football", 15),
        ("goalkeeper", 15),
    )
    body_id: str = "unitree.g1.29dof"
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.motion_atlas_manifest.v1"

    def __post_init__(self) -> None:
        if len(self.records) != 50 or len({item.motion_id for item in self.records}) != 50:
            raise ValueError("common Motion Atlas must contain 50 unique motions")
        counts = Counter(record.category.value for record in self.records)
        if tuple(sorted(counts.items())) != tuple(sorted(self.target_counts)):
            raise ValueError("Motion Atlas must preserve the 40/30/30 category split")
        if any(record.body_id != self.body_id for record in self.records):
            raise ValueError("Motion Atlas body contract is inconsistent")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_command_sent:
            raise ValueError("Motion Atlas must remain SIM_ONLY")

    @property
    def manifest_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "records": [record.to_dict() for record in self.records],
            "target_counts": [list(item) for item in self.target_counts],
            "body_id": self.body_id,
            "activation_ceiling": self.activation_ceiling,
            "hardware_command_sent": self.hardware_command_sent,
        }
        if include_hash:
            value["manifest_hash"] = self.manifest_hash
        return value


def build_s33_motion_atlas(
    *,
    motiondecode_root: Path,
    full_body_manifest_path: Path,
) -> MotionAtlasManifest:
    """Build a deterministic 20/15/15 suite only from locally available data."""

    root = motiondecode_root.expanduser().resolve()
    generic_root = root / "samples"
    football_root = root / "extracted" / "football" / "3.3.3.Football" / "3.3.3.3.Shooting"
    generic = sorted(generic_root.rglob("*.csv"))[:20]
    football = sorted(football_root.glob("*.csv"))[:15]
    if len(generic) != 20 or len(football) != 15:
        raise ValueError("MotionDecode does not contain the required balanced source sample")
    records: list[MotionAtlasRecord] = []
    for index, path in enumerate(generic):
        records.append(
            _csv_record(
                root=root,
                path=path,
                motion_id=f"generic.{index:02d}",
                category=MotionAtlasCategory.GENERIC_ATHLETE,
                tags=("locomotion", "whole_body", "generic"),
            )
        )
    for index, path in enumerate(football):
        records.append(
            _csv_record(
                root=root,
                path=path,
                motion_id=f"football.shooting.{index:02d}",
                category=MotionAtlasCategory.FOOTBALL,
                tags=("football", "shooting", "whole_body"),
            )
        )

    library = load_full_body_goalkeeper_motion_library(full_body_manifest_path)
    manifest_relative = full_body_manifest_path.expanduser().resolve().name
    source_hash = library.manifest.manifest_hash
    keeper_segments: list[tuple[str, str]] = []
    for family in ("leftjump", "rightjump", "leftstep", "rightstep"):
        keeper_segments.extend((family, phase) for phase in ("prepare", "action", "recovery"))
    keeper_segments.extend(
        (("lefthand", "action"), ("righthand", "action"), ("lefthand", "prepare"))
    )
    frame_counts = dict(library.manifest.family_frame_counts)
    for index, (family, phase) in enumerate(keeper_segments):
        frame_count = frame_counts[family]
        boundaries = (0, frame_count // 3, 2 * frame_count // 3, frame_count)
        phase_index = {"prepare": 0, "action": 1, "recovery": 2}[phase]
        records.append(
            MotionAtlasRecord(
                motion_id=f"goalkeeper.{index:02d}.{family}.{phase}",
                source_id="humanoid-goalkeeper",
                source_path=manifest_relative,
                source_hash=source_hash,
                body_id="unitree.g1.29dof",
                category=MotionAtlasCategory.GOALKEEPER,
                tags=("goalkeeper", family, phase),
                frame_count=frame_count,
                frame_rate_hz=library.manifest.source_frame_rate_hz,
                segment_start_frame=boundaries[phase_index],
                segment_end_frame_exclusive=boundaries[phase_index + 1],
                representation="g1_full_body_npz",
                license_id="CC-BY-NC-SA-4.0",
                attribution_required=True,
                commercial_use_allowed=False,
            )
        )
    return MotionAtlasManifest(records=tuple(records))


def write_s33_motion_atlas(manifest: MotionAtlasManifest, output_path: Path) -> None:
    target = output_path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _csv_record(
    *,
    root: Path,
    path: Path,
    motion_id: str,
    category: MotionAtlasCategory,
    tags: tuple[str, ...],
) -> MotionAtlasRecord:
    payload = path.read_bytes()
    frame_count = max(0, len(payload.splitlines()) - 1)
    return MotionAtlasRecord(
        motion_id=motion_id,
        source_id="chingmu-motiondecode",
        source_path=str(path.relative_to(root)),
        source_hash="sha256:" + hashlib.sha256(payload).hexdigest(),
        body_id="unitree.g1.29dof",
        category=category,
        tags=tags,
        frame_count=frame_count,
        frame_rate_hz=120.0,
        segment_start_frame=0,
        segment_end_frame_exclusive=frame_count,
        representation="g1_29dof_csv",
        license_id="ChingMu-Research-Only",
        attribution_required=True,
        commercial_use_allowed=False,
    )


__all__ = [
    "MotionAtlasCategory",
    "MotionAtlasManifest",
    "MotionAtlasRecord",
    "build_s33_motion_atlas",
    "write_s33_motion_atlas",
]
