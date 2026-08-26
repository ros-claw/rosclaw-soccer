from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw_soccer.skills.athlete_foundation.backend_adapters import (
    export_humanoid_gpt_reference,
    export_opentrack_reference,
)


def test_humanoid_gpt_adapter_rejects_source_tree_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        export_humanoid_gpt_reference(
            motion_manifest_path=tmp_path / "missing.json",
            family="leftjump",
            output_path=tmp_path / "inside.npz",
            source_checkout=tmp_path,
        )


def test_humanoid_gpt_adapter_requires_new_npz(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="new NPZ"):
        export_humanoid_gpt_reference(
            motion_manifest_path=tmp_path / "missing.json",
            family="leftjump",
            output_path=tmp_path.parent / "inside.json",
            source_checkout=tmp_path,
        )


def test_opentrack_adapter_rejects_source_tree_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        export_opentrack_reference(
            canonical_reference_path=tmp_path / "missing.npz",
            output_path=tmp_path / "inside.npz",
            model_xml_path=tmp_path / "missing.xml",
            source_checkout=tmp_path,
            family="leftjump",
        )


def test_opentrack_adapter_requires_new_npz(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="new NPZ"):
        export_opentrack_reference(
            canonical_reference_path=tmp_path / "missing.npz",
            output_path=tmp_path.parent / "inside.json",
            model_xml_path=tmp_path / "missing.xml",
            source_checkout=tmp_path,
            family="leftjump",
        )
