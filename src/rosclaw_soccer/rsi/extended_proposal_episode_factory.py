"""Explicit private fixed extended candidate; fresh numeric episode state.

No default is replaced. Native adoption requires independent parity and full
physical review. A source/canonical binding failure is not a reset operation.
"""

import copy
from pathlib import Path
from typing import Any

import rosclaw.growth.extended_proposal_regression as optimizer

from rosclaw_soccer.rsi.extended_proposal_motor import CompiledExtendedProposalMotor, make_preview
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


class ExtendedProposalEpisodeFactory:
    def __init__(self, model: dict[str, Any]) -> None:
        self._policy = make_preview(copy.deepcopy(model))
        self._canonical_model_hash = hash_json(self._policy["step_motor_proof"]["model"])
        self._prototype = CompiledExtendedProposalMotor(self._policy)
        paths = list(Path(__file__).parent.glob("*.py"))
        paths += list(Path(optimizer.__file__).parent.glob("*.py"))
        paths += [Path(__file__).parents[1] / "sim/contracts.py"]
        self._pins = {str(p): hash_bytes(p.read_bytes()) for p in paths}

    def _stable(self) -> None:
        if any(hash_bytes(Path(p).read_bytes()) != h for p, h in self._pins.items()):
            raise ValueError("fixed extended episode factory sources changed")

    @property
    def policy_hash(self) -> str:
        self._stable()
        return str(self._policy["policy_hash"])

    def preview(self, model: dict[str, Any]) -> dict[str, Any]:
        self._stable()
        if hash_json(model) != self._canonical_model_hash:
            raise ValueError("complete canonical fixed extended model changed")
        result: dict[str, Any] = copy.deepcopy(self._policy)
        return result

    def contract(self) -> dict[str, Any]:
        self._stable()
        return dict(
            schema="soccer.rsi.private_extended_proposal_factory.v1",
            policy_hash=self.policy_hash,
            complete_canonical_model_hash=self._canonical_model_hash,
            source_pins=dict(self._pins),
            complete_model_validation_at_allocation=True,
            independent_contact_history=True,
            private_fixed_plastic_head=True,
            physics_parity_requires_external_evidence=True,
            activation_ceiling="SIM_ONLY",
            promotion_authorized=False,
            hardware_authorized=False,
        )

    def new_episode(self) -> CompiledExtendedProposalMotor:
        self._stable()
        result = copy.copy(self._prototype)
        decoder = copy.copy(self._prototype._decoder)
        decoder._parent = copy.copy(self._prototype._decoder._parent)
        decoder._parent._layers = list(self._prototype._decoder._parent._layers)
        decoder._parent._residual_layers = list(self._prototype._decoder._parent._residual_layers)
        decoder._parent._warm = copy.copy(self._prototype._decoder._parent._warm)
        decoder._parent._warm.layers = list(self._prototype._decoder._parent._warm.layers)
        decoder._parent._warm.sampling = copy.deepcopy(
            self._prototype._decoder._parent._warm.sampling
        )
        decoder._parent._memory = ContactPhaseMemory()
        decoder._layers = list(self._prototype._decoder._layers)
        decoder._memory = ContactPhaseMemory()
        decoder._sampling, decoder._noise = None, None
        result._decoder, result._parent = decoder, decoder._parent
        result._layers, result._memory = decoder._layers, decoder._memory
        return result
