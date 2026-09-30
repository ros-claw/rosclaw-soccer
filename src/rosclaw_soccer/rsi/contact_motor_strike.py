"""Strike-phase successor: retain bilateral control until real ball contact.

An explicit new policy contract, not reinterpretation of preparation-only v303.
The same audited bounds, projection, interpolation and release engine is reused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi import contact_motor_primitive as basis
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "rsi_bilateral_strike_motor_primitive_v304"
START_GAP_M = 1.4
END_GAP_M = -0.35


def validate_policy(policy: dict[str, Any]) -> tuple[np.ndarray[Any, Any], str]:
    content = {k: v for k, v in policy.items() if k != "policy_hash"}
    basis_hash = hash_bytes(Path(basis.__file__).read_bytes())
    if (
        policy.get("schema") != SCHEMA
        or policy.get("phase_gap_start_m") != START_GAP_M
        or policy.get("phase_gap_end_m") != END_GAP_M
        or policy.get("policy_source_hash") != hash_bytes(Path(__file__).read_bytes())
        or policy.get("basis_source_hash") != basis_hash
        or policy.get("contract_source_hash")
        != hash_bytes(Path(__file__).with_name("contact_motor_contract.py").read_bytes())
        or policy.get("policy_hash") != hash_json(content)
    ):
        raise ValueError("unsealed strike-phase motor policy")
    # Validate all shared numeric and safety contracts through the unchanged
    # preparation engine, with an explicit schema/source conversion for checking.
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


def make_policy(knots: np.ndarray[Any, Any], commitment: str) -> dict[str, Any]:
    policy = basis.make_policy(knots, commitment)
    policy.pop("policy_hash")
    policy.update(
        schema=SCHEMA,
        phase_gap_start_m=START_GAP_M,
        phase_gap_end_m=END_GAP_M,
        basis_source_hash=policy["policy_source_hash"],
        policy_source_hash=hash_bytes(Path(__file__).read_bytes()),
        contract_source_hash=hash_bytes(
            Path(__file__).with_name("contact_motor_contract.py").read_bytes()
        ),
    )
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
    equivalent_gap = START_GAP_M - 1.15 * (START_GAP_M - gap_m) / (START_GAP_M - END_GAP_M)
    return basis.motor_delta(
        knots, equivalent_gap, baseline, limits, previous, contact_delta, frames_since_contact
    )
