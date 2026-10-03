"""Reuse verified numeric mean parameters; never reuse episode phase/noise state.

Each sampling view still passes the ORIGINAL complete preview validation.
This removes repeated nested decoder construction, not integrity checking,
weights, likelihoods, RNG law, action bounds or physical auditing.
"""

import copy
from typing import Any

from rosclaw.growth.correlated_exploration import stationary_noise

from rosclaw_soccer.rsi.smooth_memory_motor import (
    SAMPLING_SCHEMA,
    SCHEMA,
    CompiledSmoothMemoryMotor,
    make_preview,
)
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory


class SmoothSamplingDecoderFactory:
    """One private verified mean, with independent causal state for each bind."""

    def __init__(self, mean_model: dict[str, Any]) -> None:
        if mean_model.get("schema") != SCHEMA:
            raise ValueError("an ordinary sealed smooth mean model is required")
        self._mean_model = copy.deepcopy(mean_model)
        self._prototype = CompiledSmoothMemoryMotor(make_preview(self._mean_model))

    def bind(self, policy: dict[str, Any]) -> CompiledSmoothMemoryMotor:
        wrapped = policy.get("step_motor_proof", {}).get("model", {})
        if (
            wrapped.get("schema") != SAMPLING_SCHEMA
            or wrapped.get("mean_model") != self._mean_model
            or make_preview(wrapped) != policy
        ):
            raise ValueError("complete same-mean sealed sampling preview required")
        decoder = copy.copy(self._prototype)
        # Numeric arrays and search trees are private immutable copies from the
        # original constructor. All episode-owned mutable state is fresh.
        decoder._parent = copy.copy(self._prototype._parent)
        decoder._parent._memory = ContactPhaseMemory()
        decoder._memory = ContactPhaseMemory()
        decoder._sampling = {k: wrapped[k] for k in ("seed", "std_raw", "rho")}
        decoder._noise = stationary_noise(
            seed=wrapped["seed"],
            rho=wrapped["rho"],
            count=270,
            dimension=12,
            first_frame=30,
        )
        decoder._noise.flags.writeable = False
        decoder._policy_hash = policy["policy_hash"]
        return decoder
