from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from rosclaw_soccer.skills.athlete_foundation.motion_atlas import (
    MotionAtlasCategory,
    MotionAtlasManifest,
    MotionAtlasRecord,
)

_HASH = "sha256:" + "a" * 64


def _record(index: int, category: MotionAtlasCategory) -> MotionAtlasRecord:
    return MotionAtlasRecord(
        motion_id=f"motion.{index}",
        source_id="fixture",
        source_path=f"fixture/{index}.csv",
        source_hash=_HASH,
        body_id="unitree.g1.29dof",
        category=category,
        tags=("whole_body",),
        frame_count=120,
        frame_rate_hz=120.0,
        segment_start_frame=0,
        segment_end_frame_exclusive=120,
        representation="g1_29dof_csv",
        license_id="Research-Only",
        attribution_required=True,
        commercial_use_allowed=False,
    )


def _manifest() -> MotionAtlasManifest:
    categories = (
        [MotionAtlasCategory.GENERIC_ATHLETE] * 20
        + [MotionAtlasCategory.FOOTBALL] * 15
        + [MotionAtlasCategory.GOALKEEPER] * 15
    )
    return MotionAtlasManifest(
        records=tuple(_record(index, category) for index, category in enumerate(categories))
    )


def test_motion_atlas_enforces_common_40_30_30_suite() -> None:
    manifest = _manifest()

    assert manifest.manifest_hash.startswith("sha256:")
    assert Counter(record.category for record in manifest.records) == {
        MotionAtlasCategory.GENERIC_ATHLETE: 20,
        MotionAtlasCategory.FOOTBALL: 15,
        MotionAtlasCategory.GOALKEEPER: 15,
    }
    assert manifest.activation_ceiling == "SIM_ONLY"


def test_motion_atlas_rejects_category_skew_or_unsafe_source_path() -> None:
    manifest = _manifest()
    skewed = replace(
        manifest.records[-1],
        category=MotionAtlasCategory.GENERIC_ATHLETE,
    )
    with pytest.raises(ValueError, match="40/30/30"):
        MotionAtlasManifest(records=(*manifest.records[:-1], skewed))
    with pytest.raises(ValueError, match="relative"):
        replace(manifest.records[0], source_path="/absolute/data.csv")


def test_motion_atlas_rejects_commercial_claim_or_invalid_segment() -> None:
    record = _manifest().records[0]
    with pytest.raises(ValueError, match="research use"):
        replace(record, commercial_use_allowed=True)
    with pytest.raises(ValueError, match="segment bounds"):
        replace(record, segment_end_frame_exclusive=121)
