"""Independent offline episodes sharing a private validated numeric snapshot.

This factory accepts one policy at construction, not replacements at episode
boundaries. It never starts a simulator or changes a runtime default. Shared
numeric storage is read-only; contact histories and sampling state are private.
Native batch adoption requires a separate pinned experiment and replay audit.
"""

import copy
from typing import Any

from rosclaw_soccer.rsi.proposal_memory_motor import CompiledProposalMemoryMotor
from rosclaw_soccer.rsi.proposal_snapshot_compilation import compile_proposal_snapshot
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory


class ProposalEpisodeDecoderFactory:
    """One fully validated, owned mean; one fresh causal state per episode."""

    def __init__(self, policy: dict[str, Any]) -> None:
        self._prototype = compile_proposal_snapshot(policy)

    @property
    def policy_hash(self) -> str:
        return str(self._prototype._policy_hash)

    def new_episode(self) -> CompiledProposalMemoryMotor:
        """Return a fresh decoder without accepting mutable external weights.

        Only the fixed private prototype is reused. Layer containers, warm
        decoder and both contact histories are independent. Numeric arrays and
        immutable search indices retain the original law and logical banks.
        No history is reset in an already-running or faulted episode.
        """
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
