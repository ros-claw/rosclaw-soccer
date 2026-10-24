"""First-generation on-policy refinement contracts, reward and actual-time returns.

Task thresholds and physics are unchanged. This training reward is not an exam,
promotion, retention proof, or a claim that online adaptation is continuous.
"""

import hashlib
import math
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.joint_delivery_teacher import joint_delivery_rank
from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label

KIND = "refinement-actor-critic-v1"
PEER_KIND = "refinement-actor-critic-peer-loo-v2"
STATUS = "REFINEMENT_OWN_AR_ACTOR_CRITIC_UPDATED_PENDING_PHYSICAL_VALIDATION"
SCHEMA = "soccer.refinement.own_actor_critic_checkpoint.v1"
COMMITMENTS = frozenset(
    {
        "behavior_training_hash",
        "behavior_checkpoint_hash",
        "parent_lineage_hash",
        "experience_hash",
        "experience_own_hash",
        "frozen_parent_hash",
    }
)


def learning_config(state: int, *, variant: str = "zero-v1") -> dict[str, Any]:
    if type(state) is not int or state not in (25, 26):
        raise ValueError("two declared research pilot models only")
    if variant not in ("zero-v1", "peer-loo-v2"):
        raise ValueError("explicit declared generation0 advantage variant required")
    cfg = dict(
        protocol="refinement.own_AR_PPO_generation0.v1",
        state_id=state,
        seed=206023 + 101 * state,
        generation_index=0,
        epochs=4,
        actor_lr=3e-5,
        critic_lr=0.001,
        max_conditional_kl=0.015,
        max_backtracks=8,
        actor_gradient_norm=1.0,
        critic_gradient_norm=1.0,
        sampling_law="box-truncated-ar1",
        sampling_profile="contact-v1",
        rho=0.9,
        sigma_policy_units=0.02,
        gamma=0.995,
        return_rule="MONTE_CARLO_WITH_ACTUAL_OBSERVED_NEURAL_RECOVERY_DELAY",
        advantage_rule="WHOLE_BATCH_STANDARDIZED_FIXED_ROLLOUT_ZERO_CRITIC_RETURNS",
        reward="JOINT_DELIVERY_RESEARCH_SHAPING_v1_NOT_EXAM",
        source_batch_worlds=24,
        source_batch_generations=1,
        device="cpu",
        final_only=True,
        whole_parent_frozen=True,
        whole_parent_anchor_weight=0.02,
        physical_retention_proven=False,
        continuous_learning_proven=False,
    )
    if variant == "peer-loo-v2":
        cfg.update(
            protocol="refinement.own_AR_PPO_generation0_peer_LOO.v2",
            advantage_rule="IDENTICAL_INITIAL_LEAVE_OWN_TRAJECTORY_OUT_RETURNS_GLOBAL_STD",
        )
    return cfg


def kind_for_config(config: dict[str, Any]) -> str:
    state = config.get("state_id")
    if type(state) is not int:
        raise ValueError("exact integer model state required")
    for variant, kind in (("zero-v1", KIND), ("peer-loo-v2", PEER_KIND)):
        if config == learning_config(state, variant=variant):
            return kind
    raise ValueError("exact declared learning config required")


def config_for_kind(kind: str, state: int) -> dict[str, Any]:
    if kind == KIND:
        return learning_config(state)
    if kind == PEER_KIND:
        return learning_config(state, variant="peer-loo-v2")
    raise ValueError("explicit old-zero or new-peer on-policy kind required")


def joint_delivery_reward(outcome: dict[str, Any]) -> float:
    rank = joint_delivery_rank(dict(outcome=outcome))
    if not rank[0]:
        return -3.0
    error = outcome["incoming"]["incoming_target_error_3d_before_B_contact_m"]
    speed = outcome["incoming_arrival_speed_mps"]
    if error is None or speed is None:
        return -2.0
    # Joint rational shaping retains signal on distant failures. Neither
    # precision nor speed nor a receive bonus can hide the OTHER violation.
    penalty = min(float(error) / 0.1, 1e6) ** 2 + min(abs((float(speed) - 2.7) / 0.3), 1e6) ** 2
    return float(2.0 / (1.0 + penalty) + 0.25 * controlled_pass_label(outcome) + rank[1])


def observed_terminal_returns(
    stochastic_controls: int,
    observed_recovery_controls: int,
    reward: float,
) -> np.ndarray:
    if (
        type(stochastic_controls) is not int
        or not 1 <= stochastic_controls <= 300
        or type(observed_recovery_controls) is not int
        or not 0 <= observed_recovery_controls <= 100
        or type(reward) not in (int, float)
        or not math.isfinite(reward)
        or not -3 <= reward <= 3.25
    ):
        raise ValueError(
            "actual bounded stochastic/recovery clock and finite terminal reward required"
        )
    exponents = np.arange(stochastic_controls - 1, -1, -1) + observed_recovery_controls
    return np.asarray(reward * 0.995**exponents, dtype=np.float32)


def tensor_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError("finite tensor state required")
        digest.update(f"{key}:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
        digest.update(tensor.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def validate_tensor_map(actual: Any, expected: dict[str, torch.Tensor]) -> None:
    if type(actual) is not dict or actual.keys() != expected.keys():
        raise ValueError("exact declared tensor-map keys required")
    if any(
        not isinstance(v, torch.Tensor)
        or v.device.type != "cpu"
        or v.dtype != expected[k].dtype
        or v.shape != expected[k].shape
        or not torch.isfinite(v).all()
        for k, v in actual.items()
    ):
        raise ValueError("finite exact CPU checkpoint tensors required")
