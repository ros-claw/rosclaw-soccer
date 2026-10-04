"""Independent offline episodes sharing a private validated numeric snapshot.

This factory accepts one policy at construction, not replacements at episode
boundaries. It never starts a simulator or changes a runtime default. Shared
numeric storage is read-only; contact histories and sampling state are private.
Native batch adoption requires a separate pinned experiment and replay audit.
"""

import copy
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.proposal_memory_motor import CompiledProposalMemoryMotor
from rosclaw_soccer.rsi.proposal_snapshot_compilation import compile_proposal_snapshot
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory
from rosclaw_soccer.sim.contracts import hash_bytes


class ProposalEpisodeDecoderFactory:
    """One fully validated, owned mean; one fresh causal state per episode."""

    def __init__(self, policy: dict[str, Any]) -> None:
        self._prototype = compile_proposal_snapshot(policy)
        from rosclaw.growth import indexed_anchor_output_memory

        paths = list(Path(__file__).parent.glob("*.py"))
        paths += list(Path(indexed_anchor_output_memory.__file__).parent.glob("*.py"))
        self._pins = {str(p): hash_bytes(p.read_bytes()) for p in paths}

    def _stable(self) -> None:
        if any(hash_bytes(Path(p).read_bytes()) != h for p, h in self._pins.items()):
            raise ValueError("private proposal episode factory sources changed")

    @property
    def policy_hash(self) -> str:
        self._stable()
        return str(self._prototype._policy_hash)

    def contract(self) -> dict[str, Any]:
        self._stable()
        return dict(
            schema="soccer.rsi.private_proposal_episode_factory.v1",
            policy_hash=self.policy_hash,
            source_pins=dict(self._pins),
            complete_original_preview_validation_at_allocation=True,
            source_verified_at_each_episode=True,
            independent_contact_history=True,
            actor_weights_changed=False,
            physics_parity_requires_external_evidence=True,
            activation_ceiling="SIM_ONLY",
            promotion_authorized=False,
            hardware_authorized=False,
        )

    def new_episode(self) -> CompiledProposalMemoryMotor:
        """Return a fresh decoder without accepting mutable external weights.

        Only the fixed private prototype is reused. Layer containers, warm
        decoder and both contact histories are independent. Numeric arrays and
        immutable search indices retain the original law and logical banks.
        No history is reset in an already-running or faulted episode.
        """
        self._stable()
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
        decoder._sampling = None
        decoder._noise = None
        return decoder
