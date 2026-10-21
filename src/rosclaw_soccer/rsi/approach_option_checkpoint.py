"""Fixed40 long-approach SIM inference; no optimizer or physics construction.

New128-step critic warmup and parent132 baseline cannot use old20/132 readers.
The parent is independently admitted under its original strict intent first.
This module grants no policy promotion or hardware authorization.
"""

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.providers.g1.velocity_policy import (
    file_hash,
    observation98,
    phase_delta,
    target29,
)
from rosclaw_soccer.rsi.native_contact_checkpoint import NativeContinuedContactSession
from rosclaw_soccer.rsi.velocity_prior_actor import LearnedPriorNativeSession
from rosclaw_soccer.sim.contracts import hash_json

IDENTITY_FIELDS = (
    "initialization_checkpoint_hash",
    "original_onnx_hash",
    "parameter_hash",
    "native_model_hash",
    "reset_bank_hash",
    "course_proof_hash",
    "fault_proof_hash",
    "course_audit_hash",
    "fault_audit_hash",
    "source_head",
)


def validate_approach40_state(
    state: dict[str, Any], *, identities: dict[str, str], parent_actor: dict[str, Any]
) -> None:
    if set(identities) != set(IDENTITY_FIELDS) or any(
        state.get(k) != value for k, value in identities.items()
    ):
        raise ValueError(
            "exact body, parent132, source and independently audited course identities required"
        )
    required = dict(
        schema="soccer.rsi.native_approach_contact_recovery_ppo.v1",
        protocol="native-approach-measured-contact-recovery60-ppo-v1",
        episode_count=40,
        mode="train40",
        training_seed=7608,
        actor_observation_dim=177,
        critic_observation_dim=177,
        native_pd_position_targets=29,
        activation_ceiling="SIM_ONLY",
        gamma=0.995,
        GAE_lambda=0.95,
        maximum_normalized_residual=0.2,
    )
    if (
        any(state.get(k) != value for k, value in required.items())
        or state.get("fresh_critic") is not True
        or state.get("fresh_optimizer") is not True
        or state.get("optimizer_resumed") is not False
        or state.get("hardware_authorized") is not False
        or state.get("promotion_authorized") is not False
    ):
        raise ValueError("complete fixed40 new SIM-only learning intent required")
    c = state.get("counts", {})
    needed = (
        "value_only_steps",
        "PPO_steps",
        "optimizer_completed",
        "optimizer_attempted",
        "actor_update_steps",
        "head_policy_rows",
        "head_critic_rows",
        "recovery_foundation_rows",
    )
    if any(type(c.get(k)) is not int for k in needed):
        raise ValueError("explicit actual integer learning counters required")
    if (
        c["value_only_steps"] != 128
        or not 0 < c["PPO_steps"] <= 512
        or c["optimizer_completed"] != 128 + c["PPO_steps"]
        or c["optimizer_attempted"] != c["optimizer_completed"]
        or not 0 < c["actor_update_steps"] <= c["PPO_steps"]
        or not 1280 <= c["head_policy_rows"] <= 40 * 32 * 300
        or c["head_critic_rows"] != 2 * c["head_policy_rows"] - 1280
        or not 0 < c["recovery_foundation_rows"] <= 40 * 32 * 60
        or state.get("native_steps") != 10 * (c["head_policy_rows"] + c["recovery_foundation_rows"])
    ):
        raise ValueError("actual fixed40 critic/PPO/native/inference history required")
    actor, policy = state["actor_state_dict"], state["policy_state_dict"]
    keys = set(parent_actor)
    if set(actor) != keys:
        raise ValueError("complete exact177 residual actor keys required")
    for k, parent in parent_actor.items():
        value, baseline, embedded = actor[k], policy.get("baseline." + k), policy.get("actor." + k)
        for x in (value, baseline, embedded):
            if (
                not isinstance(x, torch.Tensor)
                or x.shape != parent.shape
                or x.dtype != parent.dtype
                or not torch.isfinite(x).all()
            ):
                raise ValueError("finite shape/dtype-bound actor and baseline tensors required")
        if not torch.equal(baseline, parent) or not torch.equal(value, embedded):
            raise ValueError(
                "initial baseline must be actual parent132 and actor copies must agree"
            )
        if k.startswith("base.") and not torch.equal(value, parent):
            raise ValueError("immutable original177 foundation/normalization changed")
    std = policy.get("log_std")
    if not isinstance(std, torch.Tensor) or std.shape != (29,) or not torch.isfinite(std).all():
        raise ValueError("finite29 exploration tensor required, though inference uses only mean")


class NativeApproachContactSession(LearnedPriorNativeSession):
    """Read-only neural session, bounded360 actual controls including recovery."""

    def __init__(
        self,
        model: Path,
        config: Path,
        *,
        expected_model_hash: str,
        expected_config_hash: str,
        checkpoint: Path,
        expected_checkpoint_hash: str,
        parent_checkpoint: Path,
        expected_parent_hash: str,
        initialization_checkpoint: Path,
        expected_initialization_hash: str,
        expected_native_model_hash: str,
        expected_reset_bank_hash: str,
        expected_course_proof_hash: str,
        expected_fault_proof_hash: str,
        expected_course_audit_hash: str,
        expected_fault_audit_hash: str,
        expected_source_head: str,
        arm: str,
    ) -> None:
        if (
            arm not in ("initial", "final")
            or file_hash(checkpoint) != expected_checkpoint_hash
            or file_hash(initialization_checkpoint) != expected_initialization_hash
        ):
            raise ValueError("fixed40 arm and exact candidate/initialization checkpoints required")
        initial = torch.load(initialization_checkpoint, map_location="cpu", weights_only=True)
        parent = NativeContinuedContactSession(
            model,
            config,
            expected_model_hash=expected_model_hash,
            expected_config_hash=expected_config_hash,
            checkpoint=initialization_checkpoint,
            expected_checkpoint_hash=expected_initialization_hash,
            parent_checkpoint=parent_checkpoint,
            expected_parent_hash=expected_parent_hash,
            expected_native_model_hash=expected_native_model_hash,
            arm="final",
            expected_resume_checkpoint_hash=initial["resume_checkpoint_hash"],
            expected_resume_source_hash=initial["resume_source_hash"],
            expected_resume_proof_hash=initial["resume_proof_hash"],
            expected_recovery_proof_hash=initial["recovery_proof_hash"],
            expected_recovery_fault_hash=initial["recovery_fault_hash"],
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        identities = dict(
            initialization_checkpoint_hash=expected_initialization_hash,
            original_onnx_hash=expected_model_hash,
            parameter_hash=expected_config_hash,
            native_model_hash=expected_native_model_hash,
            reset_bank_hash=expected_reset_bank_hash,
            course_proof_hash=expected_course_proof_hash,
            fault_proof_hash=expected_fault_proof_hash,
            course_audit_hash=expected_course_audit_hash,
            fault_audit_hash=expected_fault_audit_hash,
            source_head=expected_source_head,
        )
        validate_approach40_state(
            state, identities=identities, parent_actor=parent.model.state_dict()
        )
        self.model = copy.deepcopy(parent.model).eval()
        self.model.load_state_dict(
            parent.model.state_dict() if arm == "initial" else state["actor_state_dict"],
            strict=True,
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.params = parent.params
        self.reference_features, self.reference_mode = 79, "declared79"
        self.contract_hash = hash_json(
            dict(
                schema="soccer.rsi.native_approach40_inference.v1",
                checkpoint_hash=expected_checkpoint_hash,
                identities=identities,
                arm=arm,
                maximum_controls=360,
                hardware_authorized=False,
            )
        )
        self.phase = phase_delta()
        self.last_action = np.zeros(29, dtype=np.float32)
        self.last_observation = np.zeros(98, dtype=np.float32)
        self.last_reference = np.zeros(79, dtype=np.float32)
        self.next_frame = self.neural_calls = self.attempted_neural_calls = 0
        self.faulted = False

    def _propose(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        angular_velocity_body: np.ndarray,
        command_body: np.ndarray,
        reference: np.ndarray,
        *,
        frame: int,
    ) -> np.ndarray:
        if self.faulted:
            raise ValueError("new approach SIM inference fault latched")
        try:
            if type(frame) is not int or frame != self.next_frame or not 0 <= frame < 360:
                raise ValueError("consecutive actual head<=300 plus recovery<=60 controls required")
            if (
                reference.shape != (79,)
                or reference.dtype != np.float32
                or not np.isfinite(reference).all()
            ):
                raise ValueError("actual finite float32 declared79 observations required")
            obs, phase = observation98(
                qpos,
                qvel,
                angular_velocity_body,
                command_body,
                self.last_action,
                self.phase,
                self.params,
            )
            self.attempted_neural_calls += 1
            with torch.no_grad():
                action = self.model(torch.from_numpy(np.r_[obs, reference][None])).numpy()[0].copy()
            self.neural_calls += 1
            target = target29(action, self.params)
            self.last_observation, self.last_reference = obs.copy(), reference.copy()
            self.last_action, self.phase = action, phase
            self.next_frame += 1
            return target
        except Exception:
            self.faulted = True
            raise
