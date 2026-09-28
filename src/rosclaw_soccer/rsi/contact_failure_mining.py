"""Authenticate consumed 3v3 evidence and mine first-touch failure frontiers.

These labels guide curriculum design; they never certify a learned policy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_SOURCE = "red.playmaker"
_RECEIVER = "red.finisher"


def _load_evidence(folder: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, NDArray[Any]]]:
    report_path = folder / "report.json"
    protocol_path = folder / "protocol.json"
    trace_path = folder / "current-parent.npz"
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    protocol: dict[str, Any] = json.loads(protocol_path.read_text(encoding="utf-8"))
    report_body = {key: value for key, value in report.items() if key != "report_hash"}
    if (
        report.get("report_hash") != hash_json(report_body)
        or report.get("protocol_hash") != hash_bytes(protocol_path.read_bytes())
        or report.get("trace_hash") != hash_bytes(trace_path.read_bytes())
        or report.get("schema") != "rosclaw_soccer.rsi.r1_current_parent_replay.v1"
        or protocol.get("schema") != "rosclaw_soccer.rsi.r1_current_parent_replay_protocol.v1"
        or protocol.get("partition") != "CONSUMED_DEV"
        or report.get("source_stable_during_run") is not True
        or report.get("training_authorized") is not False
        or report.get("promotion_authorized") is not False
        or report.get("result", {}).get("player_count") != 6
        or report.get("result", {}).get("red_player_count") != 3
        or report.get("result", {}).get("blue_player_count") != 3
    ):
        raise ValueError(f"unauthenticated consumed 3v3 evidence: {folder}")
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    return protocol, report, trace


def _distance(foot: NDArray[Any], ball: NDArray[Any]) -> NDArray[np.float64]:
    if foot.ndim != 2 or ball.ndim != 2 or foot.shape[0] != ball.shape[0]:
        raise ValueError("aligned foot and ball trajectories required")
    if foot.shape[1] < 3 or ball.shape[1] < 3:
        raise ValueError("three-dimensional foot and ball positions required")
    result = np.linalg.norm(foot[:, :3] - ball[:, :3], axis=1)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite physical trajectory")
    return cast("NDArray[np.float64]", result)


def mine_episode(folder: Path) -> dict[str, Any]:
    protocol, report, trace = _load_evidence(folder)
    time = trace["time"]
    ball = trace["ball_pose"]
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all():
        raise ValueError("finite physical frame clock required")
    if not np.all(np.diff(time) > 0):
        raise ValueError("strictly ordered physical frames required")
    required = (
        "ball_contact_agent_code",
        "ball_contact_foot_code",
        "ball_nonfoot_contact_agent_code",
        "red_playmaker_left_foot_position",
        "red_playmaker_right_foot_position",
        "red_finisher_left_foot_position",
        "red_finisher_right_foot_position",
    )
    if any(trace[key].shape[0] != len(time) for key in required) or ball.shape[0] != len(time):
        raise ValueError("unaligned physical contact stream")
    ids = tuple(sorted(row["agent_id"] for row in report["result"]["qualities"]))
    if len(ids) != 6 or _SOURCE not in ids or _RECEIVER not in ids:
        raise ValueError("named 3v3 player identities required")
    source_code = ids.index(_SOURCE) + 1
    receiver_code = ids.index(_RECEIVER) + 1
    contacts = trace["ball_contact_agent_code"]
    feet = trace["ball_contact_foot_code"]
    request_sec = report["request_time_sec"]
    if type(request_sec) not in (int, float) or not math.isfinite(request_sec):
        raise ValueError("committed pass request required before contact attribution")
    after_request = time > request_sec + 1e-9
    source_mask = (contacts == source_code) & (feet > 0)
    incidental_precommit_count = int(np.count_nonzero(source_mask & ~after_request))
    source_frames = np.flatnonzero(source_mask & after_request)
    first_source = int(source_frames[0]) if len(source_frames) else None
    after_source = (
        np.arange(len(time)) > first_source
        if first_source is not None
        else np.zeros(len(time), dtype=bool)
    )
    receiver_frames = np.flatnonzero((contacts == receiver_code) & (feet > 0) & after_source)
    nonfoot_frames = np.flatnonzero((trace["ball_nonfoot_contact_agent_code"] > 0) & after_request)
    source_dist = np.minimum(
        _distance(trace["red_playmaker_left_foot_position"], ball),
        _distance(trace["red_playmaker_right_foot_position"], ball),
    )
    receiver_dist = np.minimum(
        _distance(trace["red_finisher_left_foot_position"], ball),
        _distance(trace["red_finisher_right_foot_position"], ball),
    )
    first_receiver = int(receiver_frames[0]) if len(receiver_frames) else None
    if report.get("pass_shot_chain_succeeded", report.get("archived_parent_reproduced")):
        stage = "COMPLETE_CHAIN"
        critical = first_receiver
    elif first_source is None:
        stage = "NO_SOURCE_FOOT_CONTACT"
        search = np.flatnonzero(time <= 2.0)
        critical = int(search[np.argmin(source_dist[search])]) if len(search) else 0
    elif first_receiver is None:
        stage = "NO_RECEIVER_FOOT_CONTACT"
        search = np.flatnonzero((time >= time[first_source]) & (time <= 4.0))
        critical = int(search[np.argmin(receiver_dist[search])]) if len(search) else first_source
    elif not report.get("chain", {}).get("clean_transfer_observed", False):
        stage = "INTERRUPTED_TRANSFER"
        critical = int(nonfoot_frames[0]) if len(nonfoot_frames) else first_receiver
    else:
        stage = "NO_IN_FRAME_SHOT"
        critical = first_receiver
    if critical is None or not 0 <= critical < len(time):
        raise ValueError("physical critical frame missing")
    position = protocol["scenario"]["ball_initial_position_m"]
    if len(position) != 3 or not all(math.isfinite(float(v)) for v in position):
        raise ValueError("finite declared ball initial state required")
    return {
        "schema": "rosclaw_soccer.rsi.contact_failure_frontier_episode.v1",
        "partition": "CONSUMED_DEV",
        "report_hash": report["report_hash"],
        "trace_hash": report["trace_hash"],
        "ball_initial_position_m": position,
        "safe": report["result"]["safe"],
        "stage": stage,
        "first_source_foot_frame": first_source,
        "incidental_precommit_source_foot_frames": incidental_precommit_count,
        "first_receiver_foot_frame": first_receiver,
        "first_nonfoot_frame": int(nonfoot_frames[0]) if len(nonfoot_frames) else None,
        "critical_frame": critical,
        "critical_time_sec": float(time[critical]),
        "minimum_source_foot_distance_first_2s_m": float(np.min(source_dist[time <= 2.0])),
        "minimum_receiver_foot_distance_after_source_m": (
            float(np.min(receiver_dist[first_source:])) if first_source is not None else None
        ),
        "critical_ball_position_m": ball[critical, :3].tolist(),
        "promotion_authorized": False,
    }


def mine_curriculum(folders: tuple[Path, ...]) -> dict[str, Any]:
    if not folders or len(set(folders)) != len(folders):
        raise ValueError("nonempty unique evidence directories required")
    episodes = [mine_episode(folder) for folder in folders]
    if len({row["report_hash"] for row in episodes}) != len(episodes):
        raise ValueError("duplicate physical report must not inflate curriculum")
    counts: dict[str, int] = {}
    for row in episodes:
        counts[row["stage"]] = counts.get(row["stage"], 0) + 1
    result = {
        "schema": "rosclaw_soccer.rsi.contact_failure_curriculum.v1",
        "partition": "CONSUMED_DEV",
        "episodes": episodes,
        "stage_counts": counts,
        "next_training_priority": (
            "sender_first_touch"
            if counts.get("NO_SOURCE_FOOT_CONTACT", 0)
            else "receiver_first_touch"
        ),
        "training_authorized": False,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = mine_curriculum(tuple(args.folders))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(
        json.dumps(
            {key: report[key] for key in ("stage_counts", "next_training_priority", "report_hash")}
        )
    )


if __name__ == "__main__":
    main()
