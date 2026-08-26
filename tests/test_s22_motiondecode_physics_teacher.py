from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw_soccer.providers.g1.asset_qualification import g1_body_hash
from rosclaw_soccer.training.goalkeeper_teacher import (
    build_motiondecode_upper_body_teacher,
)


def test_body_hash_matches_full_qualification_when_assets_are_available() -> None:
    asset_root_text = __import__("os").environ.get("ROSCLAW_G1_ASSET_ROOT")
    if not asset_root_text:
        pytest.skip("qualified G1 assets are unavailable")
    from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets

    root = Path(asset_root_text)
    assert g1_body_hash(root) == qualify_g1_assets(root).body_hash


def test_real_motiondecode_teacher_is_proxy_bounded_and_body_bound() -> None:
    dataset_text = __import__("os").environ.get("ROSCLAW_MOTIONDECODE_ROOT")
    library_text = __import__("os").environ.get("ROSCLAW_GOALKEEPER_MOTION_LIBRARY")
    asset_root_text = __import__("os").environ.get("ROSCLAW_G1_ASSET_ROOT")
    if not all((dataset_text, library_text, asset_root_text)):
        pytest.skip("MotionDecode/G1 integration materials are unavailable")

    table, report = build_motiondecode_upper_body_teacher(
        motion_library_path=Path(library_text),
        dataset_root=Path(dataset_text),
        expected_body_hash=g1_body_hash(Path(asset_root_text)),
    )

    assert len(table) == 10
    assert all(len(value) == 17 for value in table.values())
    assert all(abs(value) <= 1.0 for pose in table.values() for value in pose)
    assert report["contains_only_proxy_motion"]
    assert not report["human_goalkeeper_claim_allowed"]
    assert not report["commercial_use_allowed"]
    assert report["lower_body_authority"] == "NONE"
