"""Explicit owned-preview compilation, never policy updates or activation.

Mean semantics are validated with the original code once. Canonical mean
bytes, source and complete original preview integrity are checked each bind.
Original reference construction remains the independent audit option.
"""

import copy
from typing import Any

from rosclaw.growth.correlated_exploration import stationary_noise

from rosclaw_soccer.rsi.owned_smooth_preview import OwnedSmoothPreview
from rosclaw_soccer.rsi.smooth_memory_motor import CompiledSmoothMemoryMotor
from rosclaw_soccer.rsi.smooth_sampling_decoder_factory import SmoothSamplingDecoderFactory
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory


class OwnedSmoothSamplingFactory:
    def __init__(self, mean_model: dict[str, Any]) -> None:
        owned_mean = copy.deepcopy(mean_model)
        self._preview = OwnedSmoothPreview(owned_mean)
        # ORIGINAL complete constructor: no learned weights or guards replaced.
        self._prototype = SmoothSamplingDecoderFactory(owned_mean)._prototype

    def preview(self, model: dict[str, Any]) -> dict[str, Any]:
        return self._preview.preview(model)

    def bind(self, policy: dict[str, Any]) -> CompiledSmoothMemoryMotor:
        wrapped = self._preview.validate_preview(policy)
        decoder = copy.copy(self._prototype)
        decoder._parent = copy.copy(self._prototype._parent)
        decoder._layers = list(self._prototype._layers)
        decoder._parent._layers = list(self._prototype._parent._layers)
        decoder._parent._residual_layers = list(self._prototype._parent._residual_layers)
        decoder._parent._warm = copy.copy(self._prototype._parent._warm)
        decoder._parent._warm.layers = list(self._prototype._parent._warm.layers)
        decoder._parent._memory = ContactPhaseMemory()
        decoder._memory = ContactPhaseMemory()
        decoder._sampling = {k: wrapped[k] for k in ("seed", "std_raw", "rho")}
        decoder._noise = stationary_noise(
            seed=wrapped["seed"], rho=wrapped["rho"], count=270, dimension=12, first_frame=30
        )
        decoder._noise.flags.writeable = False
        decoder._policy_hash = policy["policy_hash"]
        return decoder
