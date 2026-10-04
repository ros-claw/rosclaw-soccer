"""Explicit original-reference episodes from one private validated mean.

No simulator, callbacks, weights updates or motion interface is exposed. Each
bind performs complete original preview validation; only immutable numeric
allocation is shared. Separate native parity evidence is still required.
"""

import copy
from pathlib import Path
from typing import Any

from rosclaw.growth.correlated_exploration import stationary_noise

from rosclaw_soccer.rsi.proposal_sampling_motor import (
    CompiledProposalSamplingMotor,
    make_preview,
    make_sampling_view,
)
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def compilation_contract(policy: dict[str, Any]) -> dict[str, Any]:
    from rosclaw_soccer.rsi import proposal_sampling_motor

    view = policy["step_motor_proof"]["model"]
    return dict(
        schema="soccer.rsi.offline_proposal_sampling_compilation.v1",
        implementation="private_original_reference_mean_fresh_episode",
        factory_source_hash=hash_bytes(Path(__file__).read_bytes()),
        sampling_source_hash=hash_bytes(Path(proposal_sampling_motor.__file__).read_bytes()),
        mean_model_hash=view["mean_model"]["model_hash"],
        sampling_model_hash=view["model_hash"],
        full_original_preview_validation=True,
        independent_contact_history=True,
        independent_noise_state=True,
        actor_weights_changed=False,
        physical_dynamics_changed=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )


def validate_compilation_contract(value: Any, policy: dict[str, Any]) -> None:
    if type(value) is not dict or hash_json(value) != hash_json(compilation_contract(policy)):
        raise ValueError("complete original-reference sampling compilation contract required")


class ProposalSamplingEpisodeFactory:
    """One private complete reference constructor, fresh state at every bind."""

    def __init__(self, mean_model: dict[str, Any]) -> None:
        self._mean_model = copy.deepcopy(mean_model)
        self._mean_hash = hash_json(self._mean_model)
        self._prototype = CompiledProposalSamplingMotor(
            make_preview(make_sampling_view(self._mean_model, seed=0))
        )

    def bind(self, policy: dict[str, Any]) -> CompiledProposalSamplingMotor:
        view = policy.get("step_motor_proof", {}).get("model", {})
        if hash_json(view.get("mean_model")) != self._mean_hash or hash_json(
            make_preview(view)
        ) != hash_json(policy):
            raise ValueError("complete same-mean original proposal sampling preview required")
        # Never copy a used episode. The prototype remains untouched; arrays
        # from original constructors are private/read-only, containers fresh.
        decoder = copy.copy(self._prototype)
        decoder._layers = list(self._prototype._layers)
        decoder._parent = copy.copy(self._prototype._parent)
        decoder._parent._layers = list(self._prototype._parent._layers)
        decoder._parent._residual_layers = list(self._prototype._parent._residual_layers)
        decoder._parent._warm = copy.copy(self._prototype._parent._warm)
        decoder._parent._warm.layers = list(self._prototype._parent._warm.layers)
        decoder._parent._warm.sampling = copy.deepcopy(self._prototype._parent._warm.sampling)
        decoder._parent._memory = ContactPhaseMemory()
        decoder._memory = ContactPhaseMemory()
        decoder._sampling = {k: view[k] for k in ("seed", "std_raw", "rho")}
        decoder._noise = stationary_noise(
            seed=view["seed"], rho=view["rho"], count=270, dimension=12, first_frame=30
        )
        decoder._noise.flags.writeable = False
        decoder._policy_hash = policy["policy_hash"]
        return decoder
