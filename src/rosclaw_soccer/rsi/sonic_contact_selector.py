"""Small evidence-trained, SIM-only approach selector above a frozen SONIC body.

This is a discrete course selector, not a neural small-brain or motor authority.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json

SCHEMA = "rosclaw_soccer.rsi.sonic_contact_selector.v1"
SCHEMA_V2 = "rosclaw_soccer.rsi.sonic_contact_selector.v2"
SCHEMA_V3 = "rosclaw_soccer.rsi.sonic_contact_selector.v3"
SCHEMA_V4 = "rosclaw_soccer.rsi.sonic_contact_selector.v4"


def choose_lateral(model: dict[str, Any], *, ball_x_m: float, ball_y_m: float) -> float:
    claimed = model.get("model_hash")
    body = {key: value for key, value in model.items() if key != "model_hash"}
    if claimed != hash_json(body) or model.get("schema") not in (
        SCHEMA,
        SCHEMA_V2,
        SCHEMA_V3,
        SCHEMA_V4,
    ):
        raise ValueError("selector checkpoint integrity or schema failed")
    maximum_x = {SCHEMA: 1.9, SCHEMA_V2: 2.1, SCHEMA_V3: 2.2, SCHEMA_V4: 2.2}[model["schema"]]
    if (
        not math.isfinite(ball_x_m)
        or not math.isfinite(ball_y_m)
        or not 1.3 <= ball_x_m <= maximum_x
        or not 0.0 <= ball_y_m <= 0.2
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("promotion_authorized") is not False
    ):
        raise ValueError("selector input or authority is outside the bounded SIM_ONLY contract")
    threshold = model.get("threshold_y_m")
    if not isinstance(threshold, float) or not 0.025 < threshold < 0.175:
        raise ValueError("selector threshold outside trained ball-position domain")
    if model["schema"] in (SCHEMA_V2, SCHEMA_V3, SCHEMA_V4):
        x_switch = model.get("far_distance_switch_x_m")
        far_y_switch = model.get("far_right_foot_switch_y_m")
        if (
            not isinstance(x_switch, float)
            or not isinstance(far_y_switch, float)
            or not 1.8 < x_switch < 1.9
            or not 0.08 < far_y_switch < 0.12
        ):
            raise ValueError("phase-aware selector thresholds outside trained range")
        if model["schema"] in (SCHEMA_V3, SCHEMA_V4):
            continuous_x = model.get("continuous_right_foot_start_x_m")
            knots = model.get("continuous_right_foot_knots")
            if (
                not isinstance(continuous_x, float)
                or not 1.9 < continuous_x < 2.0
                or not isinstance(knots, list)
                or len(knots) != (5 if model["schema"] == SCHEMA_V4 else 4)
                or any(not isinstance(knot, list) or len(knot) != 2 for knot in knots)
                or [knot[0] for knot in knots]
                != (
                    [0.04, 0.08, 0.09, 0.12, 0.16]
                    if model["schema"] == SCHEMA_V4
                    else [0.04, 0.08, 0.12, 0.16]
                )
                or any(
                    not isinstance(knot[1], float) or not 0.0 <= knot[1] <= 0.1 for knot in knots
                )
            ):
                raise ValueError("continuous right-foot teacher knots outside contract")
            if ball_x_m >= continuous_x:
                if ball_y_m <= knots[0][0]:
                    return float(knots[0][1])
                for left, right in zip(knots, knots[1:], strict=False):
                    if ball_y_m <= right[0]:
                        weight = (ball_y_m - left[0]) / (right[0] - left[0])
                        return float((1.0 - weight) * left[1] + weight * right[1])
                return float(knots[-1][1])
        if ball_x_m >= x_switch:
            return -0.1 if ball_y_m < far_y_switch else 0.1
    return -0.1 if ball_y_m < threshold else 0.0


def load_selector(path: Path) -> dict[str, Any]:
    model: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    choose_lateral(model, ball_x_m=1.5, ball_y_m=0.1)
    return model
