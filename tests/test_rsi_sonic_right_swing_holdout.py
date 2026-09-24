import json
import os
import shutil
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.verify_sonic_right_swing_holdout import verify


@pytest.fixture
def external_exam(tmp_path):
    root = os.environ.get("ROSCLAW_SOCCER_RSI_RIGHT_SWING_ROOT")
    stadium = os.environ.get("ROSCLAW_SOCCER_RSI_RIGHT_SWING_STADIUM")
    source = os.environ.get("ROSCLAW_SOCCER_RSI_RIGHT_SWING_SOURCE")
    if not root or not stadium or not source:
        pytest.skip("external right-swing FRESH evidence not provided")
    destination = tmp_path / "exam"
    destination.mkdir()
    for path in Path(root).glob("rsi-sonic-right-swing-fresh-x218-*-20260924"):
        shutil.copytree(path, destination / path.name)
    return destination, Path(stadium), Path(source)


def test_right_swing_exam_reconstructs_eight_physical_episodes(external_exam):
    root, stadium, source = external_exam
    result = verify(root, stadium_assets=stadium, probe_source=source)
    assert result["physical_execution_count"] == 8
    assert result["physical_contact_independently_reconstructed"]
    assert result["candidate_foot_first_goals"] == result["parent_foot_first_goals"] == 4
    assert result["candidate_mean_peak_ball_speed_mps"] > 3.3
    assert result["parent_mean_peak_ball_speed_mps"] < 2.4
    assert not result["promotion_authorized"]


def test_right_swing_exam_rejects_forged_motor_residual(external_exam):
    root, stadium, source = external_exam
    report = root / "rsi-sonic-right-swing-fresh-x218-y95-candidate-20260924" / "report.json"
    body = json.loads(report.read_text(encoding="utf-8"))
    body["right_contact_residual_rad"] = [0.0, 0.0, 0.0]
    report.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="report hash mismatch"):
        verify(root, stadium_assets=stadium, probe_source=source)
