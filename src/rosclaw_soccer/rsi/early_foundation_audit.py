"""Recorded early-impact neural arithmetic/history, not planner or policy proof.

Use only after authenticating source/model/trace bytes; independently verified
physics is still required to trust the measured body history. No producer
tracker or causal controller is called. Planner references and encoder feature
construction remain deliberately unqualified here.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.foundation_observation_capture import (
    capture_contract,
    validate_capture,
    validate_measured_history,
)
from rosclaw_soccer.rsi.foundation_torch_capture_audit import TorchCallAuditor
from rosclaw_soccer.sim.contracts import hash_json


def audit_early_foundation(
    trace: dict[str, np.ndarray], auditor: TorchCallAuditor
) -> dict[str, Any]:
    if type(auditor) is not TorchCallAuditor:
        raise ValueError("separately constructed independent Torch arithmetic auditor required")
    validate_capture(capture_contract(), trace, frames=300, lanes=1)
    validate_measured_history(trace)
    navigation = trace.get("foundation_navigation_joint_target_rad")
    if (
        not isinstance(navigation, np.ndarray)
        or navigation.shape != (300, 1, 29)
        or navigation.dtype != np.float64
        or not np.isfinite(navigation).all()
    ):
        raise ValueError("whole finite actual navigation joint targets required")
    error = float(np.max(abs(navigation - trace["foundation_neural_joint_target_mujoco_rad"])))
    if error > 1e-6:
        raise ValueError("actual navigation targets differ from reconstructed Foundation output")
    arithmetic = auditor.recompute(trace)
    if arithmetic["recomputed_executed_calls"] != 300:
        raise ValueError("all300 actual neural calls must be independently reconstructed")
    result = dict(
        schema="soccer.rsi.early_h300_foundation_component_audit.v1",
        arithmetic=arithmetic,
        actual_neural_calls_reconstructed=300,
        measured_decoder_history_reconstructed=True,
        maximum_navigation_capture_rounding_error_rad=error,
        producer_tracker_called=False,
        planner_reference_reconstructed=False,
        encoder_feature_construction_reconstructed=False,
        source_and_weights_authentication_required_from_caller=True,
        physical_replay_required_from_caller=True,
        full_causal_policy_qualified=False,
        teacher_qualified=False,
        new_physical_steps=0,
        new_optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    return result
