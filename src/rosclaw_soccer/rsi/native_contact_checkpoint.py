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
            or state.get("protocol") != "native-contact24-v1"
            or state.get("actor_observation_dim") != 177
            or state.get("critic_observation_dim") != 177
            or state.get("native_pd_position_targets") != 29
            or state.get("episode_count") != 20
            or state.get("value_warmup_episodes") != 4
            or state.get("ppo_episodes") != 16
            or state.get("parent_checkpoint_hash") != expected_parent_hash
            or state.get("original_onnx_hash") != expected_model_hash
            or state.get("parameter_hash") != expected_config_hash
            or state.get("native_model_hash") != expected_native_model_hash
            or state.get("activation_ceiling") != "SIM_ONLY"
            or state.get("hardware_authorized") is not False
            or state.get("promotion_authorized") is not False
            or state.get("fresh_optimizer") is not True
        ):
            raise ValueError("complete fixed final20 native-contact24 schema/identity required")
        self._validate_intent(state)
        counts = state["counts"]
        if (
            not 0 < counts["ppo_steps"] <= 256
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
