"""Learnable contact-preparation duration, retaining the unchanged motor basis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.rsi import contact_motor_primitive as basis
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "rsi_bilateral_learned_phase_motor_v306"
END_GAP_LIMITS_M = (-0.35, 0.25)


def validate_policy(policy: dict[str, Any]) -> tuple[np.ndarray[Any, Any], str]:
    content = {k: v for k, v in policy.items() if k != "policy_hash"}
    end = policy.get("phase_gap_end_m")
    if type(end) not in (float, int):
        raise ValueError("numeric learned preparation duration required")
    end_number = float(cast(float, end))
    basis_hash = hash_bytes(Path(basis.__file__).read_bytes())
    if (
        policy.get("schema") != SCHEMA
        or policy.get("phase_gap_start_m") != 1.4
        or not np.isfinite(end_number)
        or not END_GAP_LIMITS_M[0] <= end_number <= END_GAP_LIMITS_M[1]
        or policy.get("basis_source_hash") != basis_hash
        or policy.get("contract_source_hash")
        != hash_bytes(Path(__file__).with_name("contact_motor_contract.py").read_bytes())
        or policy.get("policy_source_hash") != hash_bytes(Path(__file__).read_bytes())
        or policy.get("policy_hash") != hash_json(content)
    ):
        raise ValueError("unsealed learned-phase motor policy")
    common = {
        k: v
        for k, v in content.items()
        if k
        not in ("phase_gap_start_m", "phase_gap_end_m", "basis_source_hash", "contract_source_hash")
    }
    common["schema"] = basis.SCHEMA
    common["policy_source_hash"] = basis_hash
    common["policy_hash"] = hash_json(common)
    knots, _ = basis.validate_policy(common)
    return knots, str(policy["policy_hash"])


def make_policy(knots: np.ndarray[Any, Any], end_gap_m: float, commitment: str) -> dict[str, Any]:
    if type(end_gap_m) not in (float, int) or not np.isfinite(end_gap_m):
        raise ValueError("finite numeric learned phase required")
    policy = basis.make_policy(knots, commitment)
    policy.pop("policy_hash")
    policy.update(
        schema=SCHEMA,
        phase_gap_start_m=1.4,
        phase_gap_end_m=float(end_gap_m),
        basis_source_hash=policy["policy_source_hash"],
        policy_source_hash=hash_bytes(Path(__file__).read_bytes()),
        contract_source_hash=hash_bytes(
            Path(__file__).with_name("contact_motor_contract.py").read_bytes()
        ),
    )
    policy["policy_hash"] = hash_json(policy)
    validate_policy(policy)
    return policy


def motor_delta(
    knots: np.ndarray[Any, Any],
    gap_m: float,
    baseline: np.ndarray[Any, Any],
    limits: np.ndarray[Any, Any],
    previous: np.ndarray[Any, Any],
    contact_delta: np.ndarray[Any, Any],
    frames_since_contact: int | None,
    *,
    end_gap_m: float,
) -> np.ndarray[Any, Any]:
    if (
        type(end_gap_m) not in (float, int)
        or not np.isfinite(end_gap_m)
        or not END_GAP_LIMITS_M[0] <= end_gap_m <= END_GAP_LIMITS_M[1]
    ):
        raise ValueError("invalid learned preparation duration")
    # Preserve the exact old arithmetic for the champion/rollback anchor.
    equivalent = gap_m if end_gap_m == 0.25 else 1.4 - 1.15 * (1.4 - gap_m) / (1.4 - end_gap_m)
    return basis.motor_delta(
        knots, equivalent, baseline, limits, previous, contact_delta, frames_since_contact
    )
