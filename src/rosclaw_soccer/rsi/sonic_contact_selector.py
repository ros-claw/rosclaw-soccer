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


def choose_lateral(model: dict[str, Any], *, ball_x_m: float, ball_y_m: float) -> float:
    claimed = model.get("model_hash")
    body = {key: value for key, value in model.items() if key != "model_hash"}
    if claimed != hash_json(body) or model.get("schema") != SCHEMA:
        raise ValueError("selector checkpoint integrity or schema failed")
    if (
        not math.isfinite(ball_x_m)
        or not math.isfinite(ball_y_m)
        or not 1.3 <= ball_x_m <= 1.9
        or not 0.0 <= ball_y_m <= 0.2
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("promotion_authorized") is not False
    ):
        raise ValueError("selector input or authority is outside the bounded SIM_ONLY contract")
    threshold = model.get("threshold_y_m")
    if not isinstance(threshold, float) or not 0.025 < threshold < 0.175:
        raise ValueError("selector threshold outside trained ball-position domain")
    return -0.1 if ball_y_m < threshold else 0.0


def load_selector(path: Path) -> dict[str, Any]:
    model: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    choose_lateral(model, ball_x_m=1.5, ball_y_m=0.1)
    return model
