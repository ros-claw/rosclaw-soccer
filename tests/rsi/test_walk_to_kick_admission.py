import hashlib
import json

import pytest

from scripts.rsi_probe_walk_to_kick import authenticated_case_pins


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    asset = tmp_path / "asset"
    asset.write_text("unchanged")
    path = tmp_path / "course" / "complete.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            dict(
                fresh_reexecution_exact=True,
                records=[{}] * 6,
                input_file_hashes={str(asset): digest(asset)},
            )
        )
    )
    report = dict(input_file_hashes={}, results=[dict(course="course", source_hash=digest(path))])
    return report, path, asset


def test_nested_case_assets_are_authenticated_not_guessed(tmp_path):
    report, path, asset = fixture(tmp_path)
    pins = authenticated_case_pins(tmp_path, report)
    assert pins[str(asset)] == digest(asset) and pins[str(path)] == digest(path)


@pytest.mark.parametrize("defect", ["case", "asset", "duplicate", "conflict"])
def test_bad_ancestry_cannot_start_physics(tmp_path, defect):
    report, path, asset = fixture(tmp_path)
    if defect == "case":
        path.write_text("changed")
    elif defect == "asset":
        asset.write_text("changed")
    elif defect == "duplicate":
        report["results"] *= 2
    else:
        report["input_file_hashes"][str(asset)] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        authenticated_case_pins(tmp_path, report)
