import sys

import pytest

from scripts import rsi_screen_omnicontact_foot_geometry as screen


def test_wrong_screen_hash_stops_before_loading_scene(monkeypatch, tmp_path):
    source = tmp_path / "screen.json"
    scene = tmp_path / "scene.xml"
    source.write_text("{}")
    scene.write_text("not an xml model")
    monkeypatch.setattr(screen, "clean", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "screen",
            "--dataset-root",
            str(tmp_path),
            "--screen",
            str(source),
            "--scene",
            str(scene),
            "--output",
            str(tmp_path / "out"),
            "--screen-hash",
            "incorrect",
            "--scene-hash",
            "incorrect",
            "--expected-head",
            "head",
        ],
    )
    with pytest.raises(ValueError, match="pinned screen/scene"):
        screen.main()
    assert not (tmp_path / "out").exists()
