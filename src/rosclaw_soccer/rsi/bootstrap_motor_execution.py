"""Evidence-bearing SIM_ONLY counterfactual of an offline neural proposal.

No Runtime, driver or hardware executor. Neural artifact remains unqualified;
the explicit simulator experiment, not the artifact, authorizes this preview.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.motor_bootstrap_network import actor_parameters
from rosclaw_soccer.rsi.precontact_proprio_policy import proprio_vector
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def context_at30(body: Any) -> tuple[float, ...]:
    return proprio_vector(
        np.asarray(body["root_pose_xyzw_m"])[30, 0],
        np.asarray(body["root_velocity_world"])[30, 0],
        np.asarray(body["ball_position_before_step_m"])[20, 0],
        np.asarray(body["ball_position_before_step_m"])[30, 0],
        np.asarray(body["ball_linear_velocity_before_step_m_s"])[30, 0],
        np.asarray(body["foot_geometry_position_before_step_m"])[20, 0],
        np.asarray(body["foot_geometry_position_before_step_m"])[30, 0],
    )


def configure_preview(
    model: dict[str, Any], body: Any, first_contact: int | None
) -> dict[str, Any]:
    if first_contact is not None and first_contact < 30:
        raise ValueError("neural preview context must precede contact")
    features = context_at30(body)
    parameters = actor_parameters(model, features)
    policy = make_policy(parameters[:36].reshape(3, 12), float(parameters[36]), model["model_hash"])
    policy["bootstrap_proof"] = {
        "schema": "soccer.rsi.neural_motor_sim_preview_proof.v1",
        "model": model,
        "context": list(features),
        "decision_frame": 30,
        "residual_zero_before_decision": True,
        "preview_source_hash": hash_bytes(Path(__file__).read_bytes()),
        "qualification": "UNQUALIFIED_SIM_COUNTERFACTUAL",
        "promotion_authorized": False,
    }
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


def audit_preview(policy: dict[str, Any], body: Any, force: np.ndarray[Any, Any]) -> None:
    proof = policy["bootstrap_proof"]
    if not isinstance(proof, dict):
        raise ValueError("neural simulator preview proof required")
    if (
        proof.get("schema") != "soccer.rsi.neural_motor_sim_preview_proof.v1"
        or proof.get("decision_frame") != 30
        or proof.get("residual_zero_before_decision") is not True
        or proof.get("promotion_authorized") is not False
        or proof.get("qualification") != "UNQUALIFIED_SIM_COUNTERFACTUAL"
        or proof.get("preview_source_hash") != hash_bytes(Path(__file__).read_bytes())
        or force.shape != (300, 1, 6)
        or np.any(force[:30] > 1)
    ):
        raise ValueError("unsealed or noncausal neural simulator preview")
    rebuilt = configure_preview(proof["model"], body, None)
    if rebuilt != policy:
        raise ValueError("neural motor proposal differs from measured causal context")
