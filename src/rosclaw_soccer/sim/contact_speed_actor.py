"""Small SIM_ONLY outcome-trained contact-speed actor with a sealed identity."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json

SCHEMA = "rosclaw_soccer.sim.contact_speed_actor.v1"
SLOW_M_S = 0.45
FAST_M_S = 0.50


def _joint_success(report: Mapping[str, Any]) -> bool:
    individual = report["individual"]["blue.playmaker"]
    return bool(
        report["clean_foot_only_contact_verified"]
        and report["ball_horizontal_displacement_m"] >= 0.5
        and report["stand_passed"]
        and individual["joint_projection_count"] == 0
    )


def fit_contact_speed_actor(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    """Fit one x-threshold from paired consumed physical episodes only.

    Each pair is (slow, fast).  An exhaustive finite search is sufficient for
    this one-degree-of-freedom actor; it does not imply learned full-body motion.
    """
    if len(pairs) < 4:
        raise ValueError("at least four paired training courses required")
    reference_model_hash = pairs[0][0]["model_hash"]
    reference_asset_hash = pairs[0][0]["asset_hash"]
    reference_source_hash = pairs[0][0]["source_hash"]
    rows: list[tuple[float, bool, bool, str, str]] = []
    for slow, fast in pairs:
        x = float(slow["ball_x_m"])
        y = float(slow["ball_y_m"])
        if (
            not math.isfinite(x)
            or not math.isfinite(y)
            or not 2.0 <= x <= 3.0
            or not -0.4 <= y <= 0.4
            or float(fast["ball_x_m"]) != x
            or float(fast["ball_y_m"]) != y
            or float(slow["forward_command_m_s"]) != SLOW_M_S
            or float(fast["forward_command_m_s"]) != FAST_M_S
            or slow["model_hash"] != fast["model_hash"]
            or slow["asset_hash"] != fast["asset_hash"]
            or slow["source_hash"] != fast["source_hash"]
            or slow["model_hash"] != reference_model_hash
            or slow["asset_hash"] != reference_asset_hash
            or slow["source_hash"] != reference_source_hash
            or not slow["track_ball_contacts"]
            or not fast["track_ball_contacts"]
        ):
            raise ValueError("paired courses must differ only in bounded forward speed")
        rows.append(
            (
                x,
                _joint_success(slow),
                _joint_success(fast),
                str(slow["report_hash"]),
                str(fast["report_hash"]),
            )
        )
    rows.sort(key=lambda row: row[0])
    xs = [row[0] for row in rows]
    if len(set(xs)) != len(xs):
        raise ValueError("duplicate training ball x coordinate")
    thresholds = (
        [xs[0] - 0.001]
        + [0.5 * (a + b) for a, b in zip(xs[:-1], xs[1:], strict=True)]
        + [xs[-1] + 0.001]
    )
    scores = [
        sum(slow if x < threshold else fast for x, slow, fast, _, _ in rows)
        for threshold in thresholds
    ]
    best_index = max(range(len(scores)), key=lambda index: (scores[index], -thresholds[index]))
    actor: dict[str, Any] = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "learner": "paired_outcome_threshold_search",
        "ball_x_threshold_m": thresholds[best_index],
        "near_speed_m_s": SLOW_M_S,
        "far_speed_m_s": FAST_M_S,
        "train_joint_success_count": scores[best_index],
        "train_course_count": len(rows),
        "train_report_hash_pairs": [[row[3], row[4]] for row in rows],
        "model_hash": reference_model_hash,
        "asset_hash": reference_asset_hash,
        "source_hash": reference_source_hash,
    }
    actor["actor_hash"] = hash_json(actor)
    return actor


def choose_contact_speed(actor: Mapping[str, Any], *, observed_ball_x_m: float) -> float:
    identity = {key: value for key, value in actor.items() if key != "actor_hash"}
    threshold = actor.get("ball_x_threshold_m")
    if (
        actor.get("schema") != SCHEMA
        or actor.get("activation_ceiling") != "SIM_ONLY"
        or actor.get("actor_hash") != hash_json(identity)
        or not isinstance(threshold, (int, float))
        or not 2.0 < threshold < 3.0
        or actor.get("near_speed_m_s") != SLOW_M_S
        or actor.get("far_speed_m_s") != FAST_M_S
        or not math.isfinite(observed_ball_x_m)
        or not 2.0 <= observed_ball_x_m <= 3.0
    ):
        raise ValueError("sealed bounded contact-speed actor required")
    return SLOW_M_S if observed_ball_x_m < threshold else FAST_M_S
