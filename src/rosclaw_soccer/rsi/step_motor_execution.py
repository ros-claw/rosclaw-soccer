"""Causal explicit SIM counterfactual of a per-frame neural residual proposal."""

from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi import step_motor_network as network
from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.step_motor_features import feature_vector
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def make_preview(model: dict[str, Any]) -> dict[str, Any]:
    network.validate_model(model)
    # Zero phase knots are an envelope carrier only, NOT the executing actor.
    policy = make_policy(np.zeros((3, 12)), 0.25, model["model_hash"])
    policy["execution_profile"] = "causal_per_frame_neural_residual"
    policy["step_motor_proof"] = dict(
        schema="soccer.rsi.step_motor_sim_preview.v1",
        model=model,
        decision_start_frame=30,
        nominal_target_is_pre_motor=True,
        force_input="previous_completed_frame",
        execution_source_hash=hash_bytes(Path(__file__).read_bytes()),
        qualification="UNQUALIFIED_SIM_COUNTERFACTUAL",
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
    proof = policy["step_motor_proof"]
    if make_preview(proof["model"]) != policy:
        raise ValueError("per-frame preview contract drift")
    features = feature_vector(
        joint_position=np.asarray(body["joint_position_rad"])[frame, 0],
        joint_velocity=np.asarray(body["joint_velocity_rad_s"])[frame, 0],
        root_pose_xyzw=np.asarray(body["root_pose_xyzw_m"])[frame, 0],
        root_velocity_world=np.asarray(body["root_velocity_world"])[frame, 0],
        ball_position_world=np.asarray(body["ball_position_before_step_m"])[frame, 0],
        ball_velocity_world=np.asarray(body["ball_linear_velocity_before_step_m_s"])[frame, 0],
        geometry_position_world=np.asarray(body["foot_geometry_position_before_step_m"])[frame, 0],
        nominal_target=nominal_target,
        previous_motor_delta=previous,
        previous_contact_forces=previous_contact_forces,
        frame=frame,
    )
    return network.predict_delta(
        proof["model"], features, baseline=baseline, limits=limits, previous=previous, frame=frame
    )
