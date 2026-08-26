from __future__ import annotations

import os
from pathlib import Path

import pytest

from rosclaw_soccer.providers.g1.sonic_audit import audit_g1_sonic_variants


def test_sonic_audit_fails_closed_inside_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        audit_g1_sonic_variants(
            model_root=tmp_path,
            asset_root=tmp_path,
            output_path=tmp_path / "audit.json",
            source_checkout=tmp_path,
        )


@pytest.mark.integration
def test_sonic_v11_closed_loop_audit(tmp_path: Path) -> None:
    model_root = os.environ.get("ROSCLAW_SONIC_MODEL_ROOT")
    asset_root = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if model_root is None or asset_root is None:
        pytest.skip("qualified SONIC/G1 assets are unavailable")
    report = audit_g1_sonic_variants(
        model_root=Path(model_root),
        asset_root=Path(asset_root),
        output_path=tmp_path / "audit.json",
        source_checkout=tmp_path / "source",
        control_frames=50,
    )
    assert report["passed"]
    assert report["trials"][1]["model_variant"] == "sonic_v1_1"
    assert not report["promotion_authorized"]
