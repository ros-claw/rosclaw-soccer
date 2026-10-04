"""Explicit source-pinned preview and original numeric episode allocation.

Offline SIM compilation only. Independent native equivalence is required
before use in a collection; no default constructor is replaced.
"""

import copy
from pathlib import Path
from typing import Any

from rosclaw.growth.correlated_exploration import stationary_noise

from rosclaw_soccer.rsi import owned_proposal_sampling_preview, proposal_sampling_episode_factory
from rosclaw_soccer.rsi.owned_proposal_sampling_preview import OwnedProposalSamplingPreview
from rosclaw_soccer.rsi.proposal_sampling_episode_factory import ProposalSamplingEpisodeFactory
from rosclaw_soccer.rsi.proposal_sampling_motor import CompiledProposalSamplingMotor
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def compilation_contract(policy: dict[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = proposal_sampling_episode_factory.compilation_contract(policy)
    value.update(
        schema="soccer.rsi.owned_proposal_sampling_compilation.v1",
        implementation="private_source_pinned_preview_original_numeric_fresh_episode",
        factory_source_hash=hash_bytes(Path(__file__).read_bytes()),
        preview_source_hash=hash_bytes(Path(owned_proposal_sampling_preview.__file__).read_bytes()),
        reference_factory_source_hash=hash_bytes(
            Path(proposal_sampling_episode_factory.__file__).read_bytes()
        ),
        full_original_preview_validation=False,
        original_mean_semantics_validated_at_allocation=True,
        complete_canonical_mean_and_preview_checked_each_bind=True,
        dependency_source_pins_checked_each_bind=True,
    )
    return value


def validate_compilation_contract(value: Any, policy: dict[str, Any]) -> None:
    if type(value) is not dict or hash_json(value) != hash_json(compilation_contract(policy)):
        raise ValueError("complete source-pinned original-numeric sampling contract required")


class OwnedProposalSamplingEpisodeFactory:
    def __init__(self, mean_model: dict[str, Any]) -> None:
        self._reference = ProposalSamplingEpisodeFactory(mean_model)
        self._preview = OwnedProposalSamplingPreview(mean_model)

    def preview(self, wrapped: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = self._preview.preview(wrapped)
        return result

    def restore_envelope(self, envelope: Any) -> dict[str, Any]:
        """Compact input only; all complete preview and bind checks still apply."""
        result: dict[str, Any] = self._preview.restore_envelope(envelope)
        return result

    def bind(self, policy: dict[str, Any]) -> CompiledProposalSamplingMotor:
        view = self._preview.validate_preview(policy)
        prototype = self._reference._prototype
        decoder = copy.copy(prototype)
        decoder._layers = list(prototype._layers)
        decoder._parent = copy.copy(prototype._parent)
        decoder._parent._layers = list(prototype._parent._layers)
        decoder._parent._residual_layers = list(prototype._parent._residual_layers)
        decoder._parent._warm = copy.copy(prototype._parent._warm)
        decoder._parent._warm.layers = list(prototype._parent._warm.layers)
        decoder._parent._warm.sampling = copy.deepcopy(prototype._parent._warm.sampling)
        decoder._parent._memory = ContactPhaseMemory()
        decoder._memory = ContactPhaseMemory()
        decoder._sampling = {k: view[k] for k in ("seed", "std_raw", "rho")}
        decoder._noise = stationary_noise(
            seed=view["seed"], rho=view["rho"], count=270, dimension=12, first_frame=30
        )
        decoder._noise.flags.writeable = False
        decoder._policy_hash = policy["policy_hash"]
        return decoder
