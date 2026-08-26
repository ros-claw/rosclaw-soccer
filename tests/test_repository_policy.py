from __future__ import annotations

import json
import tomllib
from pathlib import Path


def test_committed_age04_paths_are_portable() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "evidence" / "manifests" / "age04.json").read_text())
    for case in manifest["cases"]:
        for key in ("evidence_relpath", "media_relpath"):
            value = Path(case[key])
            assert not value.is_absolute()
            assert ".." not in value.parts


def test_full_videos_are_not_committed() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not list(root.rglob("*.mp4"))
    assert not list(root.rglob("*.webm"))


def test_downstream_extension_groups_are_declared() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    entry_points = project["entry-points"]

    assert entry_points["rosclaw.cli_extensions"] == {
        "soccer": "rosclaw_soccer.plugin:register_cli"
    }
    assert entry_points["rosclaw.growth.adapters"] == {
        "soccer": "rosclaw_soccer.growth.adapter:SOCCER_GROWTH_ADAPTER"
    }
    assert entry_points["rosclaw.simforge.tasks"] == {
        "soccer": "rosclaw_soccer.sim.tasks:SOCCER_TASK_PROVIDER"
    }
    assert entry_points["rosclaw.dataset.sources"] == {
        "g1-retargeted-motions": ("rosclaw_soccer.data.sources:G1_RETARGETED_MOTIONS_SOURCE"),
        "motiondecode": "rosclaw_soccer.data.sources:MOTIONDECODE_SOURCE",
        "omnicontact": "rosclaw_soccer.data.sources:OMNICONTACT_SOURCE",
    }


def test_age04_training_no_longer_imports_football_growth_from_core() -> None:
    root = Path(__file__).resolve().parents[1]
    training = (root / "src/rosclaw_soccer/training/age04_regulation.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "rosclaw.growth.approach_strike_residual",
        "rosclaw.growth.ballistic_contact_impulse_actor",
        "rosclaw.growth.football_motion_prior",
        "rosclaw.growth.phase_conditioned_residual",
    )

    assert all(value not in training for value in forbidden)
