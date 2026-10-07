"""Private SIM numeric episodes; independently qualified physics still needed.

Original complete model/preview/decoder validation runs at allocation. Only
fixed read-only numeric parameters are reused; all causal episode state is
fresh. This factory is NOT selected by any existing collector or native CLI.
"""

import copy
from typing import Any

import numpy as np
from rosclaw.growth.correlated_exploration import stationary_noise

from rosclaw_soccer.rsi.owned_recurrent_sampling_preview import OwnedRecurrentSamplingPreview
from rosclaw_soccer.rsi.recurrent_sampling_motor import CompiledRecurrentSamplingMotor
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory
from rosclaw_soccer.sim.contracts import hash_json


class RecurrentSamplingEpisodeFactory:
    """Fixed full student plus separately owned body history and AR draws."""

    def __init__(self, mean_model: dict[str, Any]) -> None:
        self._preview = OwnedRecurrentSamplingPreview(mean_model)
        view = self._preview.sampling_view(seed=0)
        policy = self._preview.preview(view)
        # The original decoder checks the entire original preview and builds
        # the original allowlisted numeric objects, not an external callback.
        self._prototype = CompiledRecurrentSamplingMotor(policy)
        self._mean_hash = hash_json(view["mean_model"])
        self._initial_policy_hash = str(policy["policy_hash"])
        self._preview._stable()

    def preview(self, view: dict[str, Any]) -> dict[str, Any]:
        return self._preview.preview(view)

    def sampling_view(self, *, seed: int) -> dict[str, Any]:
        return self._preview.sampling_view(seed=seed)

    def bind(self, policy: dict[str, Any]) -> CompiledRecurrentSamplingMotor:
        view = self._preview.validate_preview(policy)
        if (
            self._prototype._policy_hash != self._initial_policy_hash
            or self._mean_hash != self._preview._mean_hash
        ):
            raise ValueError("fixed original recurrent prototype commitment changed")
        decoder = copy.copy(self._prototype)
        behavior = copy.copy(self._prototype._behavior)
        behavior._layers = list(self._prototype._behavior._layers)
        behavior._parent = copy.copy(self._prototype._behavior._parent)
        behavior._parent._layers = list(self._prototype._behavior._parent._layers)
        behavior._parent._residual_layers = list(self._prototype._behavior._parent._residual_layers)
        behavior._parent._warm = copy.copy(self._prototype._behavior._parent._warm)
        behavior._parent._warm.layers = list(self._prototype._behavior._parent._warm.layers)
        behavior._parent._warm.sampling = copy.deepcopy(
            self._prototype._behavior._parent._warm.sampling
        )
        behavior._parent._memory = ContactPhaseMemory()
        behavior._memory = ContactPhaseMemory()
        behavior._sampling = None
        decoder._behavior, decoder._parent = behavior, behavior._parent
        decoder._recurrent = self._prototype._recurrent.new_episode()
        decoder._memory = ContactPhaseMemory()
        decoder._active_frame = None
        decoder._sampling = {k: view[k] for k in ("seed", "std_raw", "rho")}
        decoder._noise = stationary_noise(
            seed=view["seed"], rho=view["rho"], count=270, dimension=12, first_frame=30
        )
        decoder._noise.flags.writeable = False
        decoder._policy_hash = policy["policy_hash"]
        decoder._last_mean = np.zeros(12)
        decoder._last_draw = np.zeros(12)
        decoder._last_log_probability = 0.0
        decoder._last_sampled = False
        self._preview._stable()
        return decoder

    def contract(self) -> dict[str, Any]:
        self._preview._stable()
        if (
            self._prototype._policy_hash != self._initial_policy_hash
            or self._mean_hash != self._preview._mean_hash
        ):
            raise ValueError("fixed original recurrent prototype commitment changed")
        return dict(
            schema="soccer.rsi.private_recurrent_sampling_episode_factory.v1",
            complete_canonical_mean_hash=self._mean_hash,
            source_pins=dict(self._preview._pins),
            complete_original_model_and_decoder_validation_at_allocation=True,
            complete_original_preview_verification_at_each_bind=True,
            independent_contact_history=True,
            independent_recurrent_hidden_state=True,
            independent_stationary_ar_draws=True,
            actor_or_critic_weights_changed=False,
            physical_action_bounds_changed=False,
            physics_parity_requires_external_evidence=True,
            native_transport_qualification_performed=False,
            activation_ceiling="SIM_ONLY",
            runtime_execution_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
