import json
import os
import shutil
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.verify_sonic_ball_goal import verify_sonic_ball_goal
from rosclaw_soccer.sim.contracts import hash_json


@pytest.fixture
def goal_pair(tmp_path):
    primary = os.environ.get("ROSCLAW_SOCCER_RSI_BALL_GOAL_PRIMARY")
    replay = os.environ.get("ROSCLAW_SOCCER_RSI_BALL_GOAL_REPLAY")
    if primary is None or replay is None:
        pytest.skip("external paired SONIC goal evidence was not supplied")
    first, second = Path(primary), Path(replay)
    if not (first / "report.json").is_file() or not (second / "report.json").is_file():
        pytest.skip("external paired SONIC goal evidence is unavailable")
    stadium = os.environ.get("ROSCLAW_SOCCER_RSI_BALL_GOAL_STADIUM")
    return (
        shutil.copytree(first, tmp_path / "primary"),
        shutil.copytree(second, tmp_path / "replay"),
        Path(stadium) if stadium is not None else None,
    )


def test_real_ball_goal_requires_whole_ball_crossing_and_replay(goal_pair):
    primary, replay, stadium = goal_pair
    result = verify_sonic_ball_goal(primary, replay, stadium_assets=stadium)
    assert result["strict_replay"]
    assert result["whole_ball_goal_crossed"]
    assert result["goal_frame"] is not None
    assert result["minimum_pelvis_height_m"] >= 0.55
    assert not result["promotion_authorized"]
    schema = json.loads((primary / "report.json").read_text(encoding="utf-8"))["schema"]
    assert result["foot_ball_contact_independently_reconstructed"] is (
        stadium is not None and schema.endswith(".v2")
    )


def test_ball_goal_rejects_raw_trace_tamper(goal_pair):
    primary, replay, stadium = goal_pair
    path = primary / "trajectory.npz"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="trajectory hash mismatch"):
        verify_sonic_ball_goal(primary, replay, stadium_assets=stadium)


def test_ball_goal_rejects_reforged_goal_claim(goal_pair):
    primary, replay, stadium = goal_pair
    for root in (primary, replay):
        path = root / "report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report.pop("report_hash")
        report["goal_frame"] = 100
        report["report_hash"] = hash_json(report)
        path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="claim differs"):
        verify_sonic_ball_goal(primary, replay, stadium_assets=stadium)


def test_ball_goal_rejects_same_bundle_as_replay(goal_pair):
    primary, _, stadium = goal_pair
    with pytest.raises(ValueError, match="two distinct execution bundles"):
        verify_sonic_ball_goal(primary, primary, stadium_assets=stadium)
