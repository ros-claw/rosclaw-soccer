"""Explicit motor policy profiles; historical contracts are never inferred."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.rsi import contact_motor_phase as phase
from rosclaw_soccer.rsi import contact_motor_primitive as preparation
from rosclaw_soccer.rsi import contact_motor_strike as strike


def module(policy: dict[str, Any]) -> Any:
    if policy.get("schema") == preparation.SCHEMA:
        return preparation
    if policy.get("schema") == strike.SCHEMA:
        return strike
    if policy.get("schema") == phase.SCHEMA:
        return phase
    raise ValueError("unsupported explicit motor policy schema")


def validate_policy(policy: dict[str, Any]) -> tuple[np.ndarray[Any, Any], str]:
    return cast(tuple[np.ndarray[Any, Any], str], module(policy).validate_policy(policy))


def load_policy(path: Path) -> tuple[dict[str, Any], np.ndarray[Any, Any]]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    knots, _ = validate_policy(policy)
    return policy, knots


def motor_delta(
    policy: dict[str, Any],
    knots: np.ndarray[Any, Any],
    gap_m: float,
    baseline: np.ndarray[Any, Any],
    limits: np.ndarray[Any, Any],
    previous: np.ndarray[Any, Any],
    contact_delta: np.ndarray[Any, Any],
    frames_since_contact: int | None,
) -> np.ndarray[Any, Any]:
    if policy.get("schema") == phase.SCHEMA:
        return phase.motor_delta(
            knots,
            gap_m,
            baseline,
            limits,
            previous,
            contact_delta,
            frames_since_contact,
            end_gap_m=policy["phase_gap_end_m"],
        )
    return cast(
        np.ndarray[Any, Any],
        module(policy).motor_delta(
            knots, gap_m, baseline, limits, previous, contact_delta, frames_since_contact
        ),
    )
