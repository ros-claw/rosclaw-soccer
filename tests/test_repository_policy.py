from __future__ import annotations

import json
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
