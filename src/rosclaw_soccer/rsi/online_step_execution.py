"""Explicit SIM preview of the updated PPO actor, preserving learning provenance."""

from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi import step_motor_ppo as network
from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.step_motor_network import predict_delta
from rosclaw_soccer.rsi.stochastic_step_execution import features_at_frame
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def make_preview(model: dict[str, Any]) -> dict[str, Any]:
    network.validate_model(model)
    policy = make_policy(np.zeros((3, 12)), 0.25, model["model_hash"])
    policy["execution_profile"] = "causal_per_frame_neural_residual"
    policy["step_motor_proof"] = dict(
        schema="soccer.rsi.online_step_motor_sim_preview.v1",
        model=model,
        decision_start_frame=30,
        nominal_target_is_pre_motor=True,
        force_input="previous_completed_frame",
        execution_source_hash=hash_bytes(Path(__file__).read_bytes()),
        qualification="UNQUALIFIED_SIM_COUNTERFACTUAL",
        promotion_authorized=False,
    )
    policy["online_step_motor_proof"] = dict(
        physical_batch_hash=model["physical_batch_hash"],
        learning_receipt=model["learning_receipt"],
        promotion_authorized=False,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


def delta_at_frame(
    policy: dict[str, Any],
    body: Any,
    *,
    frame: int,
    nominal_target: Any,
    baseline: Any,
    limits: Any,
    previous: Any,
    previous_contact_forces: Any,
) -> np.ndarray[Any, Any]:
    model = policy["step_motor_proof"]["model"]
    if make_preview(model) != policy:
        raise ValueError("unsealed online per-frame policy preview")
    features = features_at_frame(
        body,
        frame=frame,
        nominal_target=nominal_target,
        previous=previous,
        previous_contact_forces=previous_contact_forces,
    )
    return predict_delta(
        network.numeric_view(model),
        features,
        frame=frame,
        baseline=baseline,
        limits=limits,
        previous=previous,
    )
