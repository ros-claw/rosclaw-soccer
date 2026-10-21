"""Explicit native-contact24 SIM checkpoint reader; never a legacy certificate.

Reuse the established measured-state inference/history implementation, but admit
only the new177-critic protocol with its own immutable identities. No optimizer,
critic or runtime is constructed for inference. Fixed final20, no best-of picking.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.providers.g1.velocity_policy import (
    VelocityPolicyParameters,
    file_hash,
    phase_delta,
)
from rosclaw_soccer.rsi.velocity_prior_actor import LearnedPriorNativeSession, VelocityPriorActor
from rosclaw_soccer.sim.contracts import hash_json


class NativeContactSession(LearnedPriorNativeSession):
    checkpoint_schema = "soccer.rsi.native_contact24_ppo.v1"
    inference_schema = "soccer.rsi.native_contact24_inference.v1"
    protocol = "native-contact24-v1"
    final_episode = 20
    ppo_episodes = 16
    max_ppo_steps = 256
    fresh_optimizer_required = True
    model: Any

    def _make_model(self, model: Path, expected_hash: str) -> Any:
        return VelocityPriorActor.from_onnx(
            model, expected_hash=expected_hash, reference_features=79
        )

    def _validate_intent(self, state: dict[str, Any]) -> None:
        """Legacy native shared actor has no adapter intent to inherit."""
        if state.get("actor_kind", "shared") != "shared":
            raise ValueError("shared native checkpoint cannot contain an adapter intent")

    def _validate_actor(
        self, actor: dict[str, Any], baseline: dict[str, Any], parent: dict[str, Any]
    ) -> None:
        if any(
            not torch.equal(v, parent["algorithm"]["actor_state_dict"]["mlp." + k])
            for k, v in baseline.items()
        ):
            raise ValueError("frozen checkpoint baseline differs from actual declared parent")
        for k in ("prior_mean", "prior_denominator"):
            if not torch.equal(actor[k], self.model.state_dict()[k]):
                raise ValueError("original fixed deployment normalization required")

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
        expected_native_model_hash: str,
        arm: str,
    ) -> None:
        # Deliberately do not call the legacy admission constructor: this is a
        # separate intent/schema. Inherited propose methods retain exact177 ABI.
        if (
            arm not in ("initial", "final")
            or file_hash(checkpoint) != expected_checkpoint_hash
            or file_hash(parent_checkpoint) != expected_parent_hash
        ):
            raise ValueError("explicit new native-contact arm and pinned checkpoints required")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            state.get("schema") != self.checkpoint_schema
            or state.get("protocol") != self.protocol
            or state.get("actor_observation_dim") != 177
            or state.get("critic_observation_dim") != 177
            or state.get("native_pd_position_targets") != 29
            or state.get("episode_count") != self.final_episode
            or state.get("value_warmup_episodes") != 4
            or state.get("ppo_episodes") != self.ppo_episodes
            or state.get("parent_checkpoint_hash") != expected_parent_hash
            or state.get("original_onnx_hash") != expected_model_hash
            or state.get("parameter_hash") != expected_config_hash
            or state.get("native_model_hash") != expected_native_model_hash
            or state.get("activation_ceiling") != "SIM_ONLY"
            or state.get("hardware_authorized") is not False
            or state.get("promotion_authorized") is not False
            or state.get("fresh_optimizer") is not self.fresh_optimizer_required
        ):
            raise ValueError(
                f"complete fixed final{self.final_episode} native-contact schema/identity required"
            )
        self._validate_intent(state)
        counts = state["counts"]
        if (
            not 0 < counts["ppo_steps"] <= self.max_ppo_steps
            or counts["value_only_steps"] != 64
            or counts["optimizer_completed"] != counts["optimizer_attempted"]
            or counts["optimizer_completed"] != counts["ppo_steps"] + 64
        ):
            raise ValueError("actual completed native actor/critic optimization required")
        self.params = VelocityPolicyParameters.load(config, expected_hash=expected_config_hash)
        self.model = self._make_model(model, expected_model_hash)
        keys = set(self.model.state_dict())
        parent = torch.load(parent_checkpoint, map_location="cpu", weights_only=True)
        baseline = {k: state["policy_state_dict"]["baseline." + k] for k in keys}
        actor = baseline if arm == "initial" else state["actor_state_dict"]
        if set(actor) != keys or any(not torch.isfinite(v).all() for v in actor.values()):
            raise ValueError("complete finite native177 actor required")
        self._validate_actor(actor, baseline, parent)
        self.model.load_state_dict(actor, strict=True)
        self.model.eval()
        self.reference_features = 79
        self.reference_mode = "declared79"
        self.contract_hash = hash_json(
            dict(
                schema=self.inference_schema,
                checkpoint_hash=expected_checkpoint_hash,
                parent_checkpoint_hash=expected_parent_hash,
                native_model_hash=expected_native_model_hash,
                parameter_hash=self.params.contract_hash,
                arm=arm,
                observation=177,
                targets=29,
                hardware_authorized=False,
            )
        )
        self.phase = phase_delta()
        self.last_action = np.zeros(29, dtype=np.float32)
        self.last_observation = np.zeros(98, dtype=np.float32)
        self.last_reference = np.zeros(79, dtype=np.float32)
        self.next_frame = self.neural_calls = self.attempted_neural_calls = 0
        self.faulted = False


class NativeResidualContactSession(NativeContactSession):
    """Distinct intent: immutable177 foundation and learned bounded contact head."""

    checkpoint_schema = "soccer.rsi.native_contact24_residual_ppo.v1"
    inference_schema = "soccer.rsi.native_contact24_residual_inference.v1"

    def _make_model(self, model: Path, expected_hash: str) -> Any:
        from rosclaw_soccer.rsi.native_contact_adapter import NativeResidualContactActor

        return NativeResidualContactActor(super()._make_model(model, expected_hash))

    def _validate_intent(self, state: dict[str, Any]) -> None:
        if (
            state.get("actor_kind") != "bounded-residual"
            or state.get("adapter_config")
            != dict(
                near_radius_m=0.35,
                far_radius_m=0.7,
                maximum_normalized_residual=0.2,
                missing_ball9_disables_adapter=True,
                hidden_dims=[128, 64],
                input_dim=177,
                output_dim=29,
            )
            or not isinstance(state.get("adapter_proof_hash"), str)
        ):
            raise ValueError("exact bounded residual intent and configuration required")

    def _validate_actor(
        self, actor: dict[str, Any], baseline: dict[str, Any], parent: dict[str, Any]
    ) -> None:
        if any(not torch.isfinite(v).all() for v in baseline.values()):
            raise ValueError("complete finite residual baseline required")
        for k, value in baseline.items():
            if k.startswith("base.") and (
                not torch.equal(
                    value, parent["algorithm"]["actor_state_dict"]["mlp." + k.removeprefix("base.")]
                )
                or not torch.equal(actor[k], value)
            ):
                raise ValueError("residual checkpoint changed immutable foundation")
        for k in ("adapter.4.weight", "adapter.4.bias"):
            if torch.count_nonzero(baseline[k]):
                raise ValueError("baseline adapter must retain zero-influence head")
        for k in ("base.prior_mean", "base.prior_denominator"):
            if not torch.equal(actor[k], self.model.state_dict()[k]):
                raise ValueError("original residual deployment normalization required")


class NativeRecoveryContactSession(NativeResidualContactSession):
    """New finite-option intent; never admitted under the old contact24 schema."""

    checkpoint_schema = "soccer.rsi.native_contact24_recovery60_residual_ppo.v1"
    inference_schema = "soccer.rsi.native_contact24_recovery60_inference.v1"
    protocol = "native-contact24-recovery60-v1"
    max_recovery_rows = 38400

    def __init__(
        self,
        *args: Any,
        expected_recovery_proof_hash: str,
        expected_recovery_fault_hash: str,
        **kwargs: Any,
    ) -> None:
        for h in (expected_recovery_proof_hash, expected_recovery_fault_hash):
            if (
                not h.startswith("sha256:")
                or len(h) != 71
                or any(c not in "0123456789abcdef" for c in h[7:])
            ):
                raise ValueError("explicit actual recovery and fault proof hashes required")
        self._recovery_proof_hash = expected_recovery_proof_hash
        self._recovery_fault_hash = expected_recovery_fault_hash
        super().__init__(*args, **kwargs)

    def _validate_intent(self, state: dict[str, Any]) -> None:
        super()._validate_intent(state)
        if (
            state.get("recovery_proof_hash") != self._recovery_proof_hash
            or state.get("recovery_fault_hash") != self._recovery_fault_hash
            or state.get("training_seed") != 7607
            or not 0 < state["counts"].get("recovery_foundation_rows", 0) <= self.max_recovery_rows
        ):
            raise ValueError("actual finite-option recovery intent/counts/qualification required")


class NativeContinuedContactSession(NativeRecoveryContactSession):
    """Only the fixed132 resumed experiment; legacy readers retain their ceiling."""

    checkpoint_schema = "soccer.rsi.native_contact24_recovery60_continuation_ppo.v1"
    inference_schema = "soccer.rsi.native_contact24_recovery60_continuation_inference.v1"
    protocol = "native-contact24-recovery60-continuation-v1"
    final_episode = 132
    ppo_episodes = 128
    max_ppo_steps = 2048
    max_recovery_rows = 253440
    fresh_optimizer_required = False

    def __init__(
        self,
        *args: Any,
        expected_resume_checkpoint_hash: str,
        expected_resume_proof_hash: str,
        expected_resume_source_hash: str,
        **kwargs: Any,
    ):
        self._resume_expected = dict(
            resume_checkpoint_hash=expected_resume_checkpoint_hash,
            resume_proof_hash=expected_resume_proof_hash,
            resume_source_hash=expected_resume_source_hash,
        )
        for h in self._resume_expected.values():
            if (
                not h.startswith("sha256:")
                or len(h) != 71
                or any(c not in "0123456789abcdef" for c in h[7:])
            ):
                raise ValueError("explicit actual resume/source/paired-proof hashes required")
        super().__init__(*args, **kwargs)

    def _validate_intent(self, state: dict[str, Any]) -> None:
        super()._validate_intent(state)
        before = state.get("counts_before_resume", {})
        if (
            state.get("resume_mode") != "train132"
            or state.get("optimizer_resumed") is not True
            or any(state.get(k) != v for k, v in self._resume_expected.items())
            or before.get("optimizer_completed") != 320
            or before.get("optimizer_attempted") != 320
            or before.get("value_only_steps") != 64
            or before.get("ppo_steps") != 256
        ):
            raise ValueError("qualified resumed132 training intent and prior history required")
