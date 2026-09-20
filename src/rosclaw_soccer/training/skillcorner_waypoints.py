"""Detected-only human tactical waypoint targets; no invented intent labels.

Coordinates follow SkillCorner's centered metre frame. This adapter does not
normalize attacking direction or claim transfer to robot tactics.
"""

import math
from typing import Any


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def skillcorner_waypoint_pair(
    current: dict[str, Any],
    future: dict[str, Any],
    *,
    pitch_length: float,
    pitch_width: float,
    horizon_frames: int = 5,
) -> tuple[dict[str, Any], ...]:
    """Emit 0.5--1 s targets at the official 10 Hz, dropping undetected endpoints.

    Frame gaps or period transitions are not interpolated. Missing coordinates,
    ball detections and player detections are never filled with zeros.
    """
    if (
        not _finite(pitch_length)
        or not _finite(pitch_width)
        or not 50 <= pitch_length <= 150
        or not 30 <= pitch_width <= 100
        or type(horizon_frames) is not int
        or not 5 <= horizon_frames <= 10
    ):
        raise ValueError("finite pitch dimensions and a 5--10 frame horizon required")
    for frame in (current, future):
        if (
            not isinstance(frame, dict)
            or type(frame.get("frame")) is not int
            or frame["frame"] < 0
            or not isinstance(frame.get("player_data"), list)
        ):
            raise ValueError("typed tracking frame required")
    if (
        current.get("period") not in (1, 2)
        or type(current.get("period")) is not int
        or type(future.get("period")) is not int
        or current["period"] != future["period"]
        or future["frame"] - current["frame"] != horizon_frames
    ):
        return ()
    ball = current.get("ball_data", {})
    if (
        not isinstance(ball, dict)
        or ball.get("is_detected") is not True
        or not all(_finite(ball.get(k)) for k in ("x", "y"))
    ):
        return ()

    def players(frame: dict[str, Any]) -> dict[int, dict[str, Any]]:
        values = {}
        seen = set()
        for row in frame["player_data"]:
            if not isinstance(row, dict) or type(row.get("player_id")) is not int:
                raise ValueError("typed player identity required")
            player = row["player_id"]
            if player in seen:
                raise ValueError("duplicate player in one frame")
            seen.add(player)
            if row.get("is_detected") is True and all(_finite(row.get(k)) for k in ("x", "y")):
                values[player] = row
        return values

    now, later = players(current), players(future)
    rows = []
    for player in sorted(now.keys() & later.keys()):
        a, b = now[player], later[player]
        rows.append(
            {
                "player_id": player,
                "frame": current["frame"],
                "period": current["period"],
                "horizon_sec": horizon_frames / 10,
                "position_normalized": [a["x"] / pitch_length, a["y"] / pitch_width],
                "ball_relative_normalized": [
                    (ball["x"] - a["x"]) / pitch_length,
                    (ball["y"] - a["y"]) / pitch_width,
                ],
                "waypoint_delta_normalized": [
                    (b["x"] - a["x"]) / pitch_length,
                    (b["y"] - a["y"]) / pitch_width,
                ],
                "intent_label": None,
                "source": "detected_tracking_endpoints_not_robot_demonstration",
            }
        )
    return tuple(rows)
