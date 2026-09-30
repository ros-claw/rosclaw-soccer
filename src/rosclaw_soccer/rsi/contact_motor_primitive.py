"""SIM_ONLY learnable bilateral motor primitive with causal phase and slew limits.

This is a low-dimensional policy-search substrate, not an end-to-end neural actor.
Its phase depends on measured ball/root separation, never a future contact label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.rsi.taskspace_swing_evidence import LEG_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

JOINT_NAMES = tuple(name for leg in LEG_NAMES for name in leg)
CAP_RAD = 0.16
SLEW_RAD = 0.012
RELEASE_FRAMES = 20
SCHEMA = "rsi_bilateral_contact_motor_primitive_v303"


def validate_policy(policy: dict[str, Any]) -> tuple[np.ndarray[Any, Any], str]:
    content = {k: v for k, v in policy.items() if k != "policy_hash"}
    knots = np.asarray(policy.get("knots_rad"), dtype=np.float64)
    if (
        policy.get("schema") != SCHEMA
        or policy.get("activation_ceiling") != "SIM_ONLY"
        or policy.get("promotion_authorized") is not False
        or policy.get("joint_names") != list(JOINT_NAMES)
        or policy.get("cap_rad") != CAP_RAD
        or policy.get("slew_rad_per_frame") != SLEW_RAD
        or policy.get("release_frames") != RELEASE_FRAMES
        or policy.get("policy_hash") != hash_json(content)
        or knots.shape != (3, 12)
        or not np.isfinite(knots).all()
        or np.max(np.abs(knots)) > CAP_RAD
        or not isinstance(policy.get("training_commitment"), str)
        or not policy["training_commitment"].startswith("sha256:")
        or policy.get("policy_source_hash") != hash_bytes(Path(__file__).read_bytes())
    ):
        raise ValueError("unsealed SIM_ONLY bilateral motor policy")
    return knots, str(policy["policy_hash"])


def make_policy(knots: np.ndarray[Any, Any], commitment: str) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "joint_names": list(JOINT_NAMES),
        "cap_rad": CAP_RAD,
        "slew_rad_per_frame": SLEW_RAD,
        "release_frames": RELEASE_FRAMES,
        "knots_rad": np.asarray(knots, dtype=np.float64).tolist(),
        "training_commitment": commitment,
        "policy_source_hash": hash_bytes(Path(__file__).read_bytes()),
    }
    policy["policy_hash"] = hash_json(policy)
    validate_policy(policy)
    return policy


def load_policy(path: Path) -> tuple[dict[str, Any], np.ndarray[Any, Any]]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    knots, _ = validate_policy(policy)
    return policy, knots


def motor_delta(
    knots: np.ndarray[Any, Any],
    gap_m: float,
    baseline: np.ndarray[Any, Any],
    limits: np.ndarray[Any, Any],
    previous: np.ndarray[Any, Any],
    contact_delta: np.ndarray[Any, Any],
    frames_since_contact: int | None,
) -> np.ndarray[Any, Any]:
    if (
        knots.shape != (3, 12)
        or baseline.shape != (12,)
        or limits.shape != (12, 2)
        or previous.shape != (12,)
        or contact_delta.shape != (12,)
        or not all(np.isfinite(x).all() for x in (knots, baseline, limits, previous, contact_delta))
        or not np.isfinite(gap_m)
        or np.max(np.abs(knots)) > CAP_RAD
        or np.max(np.abs(previous)) > CAP_RAD + 1e-5
        or np.max(np.abs(contact_delta)) > CAP_RAD + 1e-5
        or np.any(limits[:, 0] >= limits[:, 1])
        or (
            frames_since_contact is not None
            and (type(frames_since_contact) is not int or frames_since_contact < 1)
        )
    ):
        raise ValueError("invalid measured motor policy state")
    if frames_since_contact is None:
        phase = float(np.clip((1.4 - gap_m) / 1.15, 0.0, 1.0))
        segment = min(int(phase * 2), 1)
        t = phase * 2 - segment
        smooth = t * t * (3 - 2 * t)
        desired = ((1 - smooth) * knots[segment] + smooth * knots[segment + 1]) * np.sin(
            np.pi * phase
        )
    else:
        desired = contact_delta * max(0.0, 1 - frames_since_contact / RELEASE_FRAMES)
    proposed = previous + np.clip(desired - previous, -SLEW_RAD, SLEW_RAD)
    # Do not enlarge an existing foundation limit violation; never force a bad
    # baseline into range with a discontinuous target jump.
    lower = np.minimum(0.0, limits[:, 0] - baseline)
    upper = np.maximum(0.0, limits[:, 1] - baseline)
    return cast(
        np.ndarray[Any, Any],
        np.clip(proposed, np.maximum(lower, -CAP_RAD), np.minimum(upper, CAP_RAD)),
    )
