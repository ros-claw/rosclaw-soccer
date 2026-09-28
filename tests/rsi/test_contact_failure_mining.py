"""Consumed first-touch mining cannot count pre-commit collisions as passes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_failure_mining import mine_curriculum, mine_episode
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _fixture(folder: Path) -> None:
    folder.mkdir()
    protocol = {
        "schema": "rosclaw_soccer.rsi.r1_current_parent_replay_protocol.v1",
        "partition": "CONSUMED_DEV",
        "scenario": {"ball_initial_position_m": [1.92, -0.8, 0.115]},
    }
    protocol_path = folder / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    time = np.arange(6) * 0.02
    ball = np.repeat(np.array([[1.92, -0.8, 0.115]]), 6, axis=0)
    foot = np.repeat(np.array([[1.8, -0.8, 0.1]]), 6, axis=0)
    codes = np.zeros(6, dtype=int)
    codes[1] = 6  # Physical foot contact before the pass request at t=0.04.
    feet = np.zeros(6, dtype=int)
    feet[1] = 1
    trace_path = folder / "current-parent.npz"
    np.savez_compressed(
        trace_path,
        time=time,
        ball_pose=ball,
        ball_contact_agent_code=codes,
        ball_contact_foot_code=feet,
        ball_nonfoot_contact_agent_code=np.zeros(6, dtype=int),
        red_playmaker_left_foot_position=foot,
        red_playmaker_right_foot_position=foot,
        red_finisher_left_foot_position=foot,
        red_finisher_right_foot_position=foot,
    )
    report = {
        "schema": "rosclaw_soccer.rsi.r1_current_parent_replay.v1",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "trace_hash": hash_bytes(trace_path.read_bytes()),
        "source_stable_during_run": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "request_time_sec": 0.04,
        "pass_shot_chain_succeeded": False,
        "chain": {"clean_transfer_observed": False},
        "result": {
            "safe": True,
            "player_count": 6,
            "red_player_count": 3,
            "blue_player_count": 3,
            "qualities": [
                {"agent_id": name}
                for name in (
                    "blue.finisher",
                    "blue.goalkeeper",
                    "blue.playmaker",
                    "red.finisher",
                    "red.goalkeeper",
                    "red.playmaker",
                )
            ],
        },
    }
    report["report_hash"] = hash_json(report)
    (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")


def test_precommit_touch_is_not_successful_sender_contact(tmp_path: Path) -> None:
    folder = tmp_path / "episode"
    _fixture(folder)
    episode = mine_episode(folder)
    assert episode["stage"] == "NO_SOURCE_FOOT_CONTACT"
    assert episode["first_source_foot_frame"] is None
    assert episode["incidental_precommit_source_foot_frames"] == 1
    assert mine_curriculum((folder,))["next_training_priority"] == "sender_first_touch"


def test_tamper_and_duplicate_evidence_are_rejected(tmp_path: Path) -> None:
    folder = tmp_path / "episode"
    _fixture(folder)
    with pytest.raises(ValueError, match="unique evidence"):
        mine_curriculum((folder, folder))
    with (folder / "current-parent.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="unauthenticated"):
        mine_episode(folder)


def test_missing_pass_commitment_is_a_failure_stage(tmp_path: Path) -> None:
    folder = tmp_path / "episode"
    _fixture(folder)
    path = folder / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["request_time_sec"] = None
    report.pop("report_hash")
    report["report_hash"] = hash_json(report)
    path.write_text(json.dumps(report), encoding="utf-8")
    episode = mine_episode(folder)
    assert episode["stage"] == "NO_PASS_COMMITMENT"
    assert episode["pass_request_time_sec"] is None
    assert episode["incidental_precommit_source_foot_frames"] == 1
