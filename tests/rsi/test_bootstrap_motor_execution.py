import copy
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi import motor_bootstrap_network as neural
from rosclaw_soccer.rsi.bootstrap_motor_execution import audit_preview, configure_preview
from rosclaw_soccer.rsi.contact_motor_phase import validate_policy
from rosclaw_soccer.rsi.precontact_proprio_policy import FEATURE_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def fixture():
    actor = {
        "dimensions": [13, 32, 37],
        "mean": [0] * 13,
        "scale": [1] * 13,
        "layers": [
            {"weight": np.zeros((32, 13)).tolist(), "bias": [0] * 32},
            {"weight": np.zeros((37, 32)).tolist(), "bias": [0.1] * 36 + [0.9]},
        ],
    }
    critic = {
        "dimensions": [50, 32, 3],
        "mean": [0] * 50,
        "scale": [1] * 50,
        "layers": [
            {"weight": np.zeros((32, 50)).tolist(), "bias": [0] * 32},
            {"weight": np.zeros((3, 32)).tolist(), "bias": [0] * 3},
        ],
    }
    model = {
        "schema": "soccer.rsi.offline_motor_actor_critic_bootstrap.v1",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "runtime_execution_authorized": False,
        "fresh_holdout_open_authorized": False,
        "learning_kind": "offline_imitation_and_outcome_regression",
        "observation_names": list(FEATURE_NAMES),
        "source_hash": hash_bytes(Path(neural.__file__).read_bytes()),
        "bank_hash": hash_json({"test_bank": True}),
        "actor": actor,
        "critic": critic,
    }
    model["model_hash"] = hash_json(model)
    body = {
        "root_pose_xyzw_m": np.zeros((300, 1, 7)),
        "root_velocity_world": np.zeros((300, 1, 6)),
        "ball_position_before_step_m": np.ones((300, 1, 3)),
        "ball_linear_velocity_before_step_m_s": np.zeros((300, 1, 3)),
        "foot_geometry_position_before_step_m": np.zeros((300, 1, 4, 3)),
    }
    return model, body


def test_causal_neural_proposal_seals_valid_phase_policy_without_runtime_authority():
    model, body = fixture()
    policy = configure_preview(model, body, None)
    validate_policy(policy)
    audit_preview(policy, body, np.zeros((300, 1, 6)))
    assert policy["bootstrap_proof"]["qualification"] == "UNQUALIFIED_SIM_COUNTERFACTUAL"
    assert policy["promotion_authorized"] is False
    for value in body.values():
        value[31:] = 9999
    audit_preview(policy, body, np.zeros((300, 1, 6)))


def test_forged_motor_knots_or_context_rejected_even_when_policy_is_resealed():
    model, body = fixture()
    policy = configure_preview(model, body, None)
    altered = copy.deepcopy(policy)
    altered["knots_rad"][0][0] += 0.01
    altered["policy_hash"] = hash_json({k: v for k, v in altered.items() if k != "policy_hash"})
    validate_policy(altered)
    with pytest.raises(ValueError, match="measured causal context"):
        audit_preview(altered, body, np.zeros((300, 1, 6)))
    body["root_velocity_world"][30, 0, 0] += 0.01
    with pytest.raises(ValueError):
        audit_preview(policy, body, np.zeros((300, 1, 6)))


def test_contact_before_decision_and_incomplete_proof_rejected():
    model, body = fixture()
    with pytest.raises(ValueError):
        configure_preview(model, body, 29)
    policy = configure_preview(model, body, 30)
    force = np.zeros((300, 1, 6))
    force[29, 0, 0] = 2
    with pytest.raises(ValueError):
        audit_preview(policy, body, force)
    policy["bootstrap_proof"] = None
    with pytest.raises(ValueError):
        audit_preview(policy, body, np.zeros((300, 1, 6)))
