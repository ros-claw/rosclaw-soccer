"""Private fixed imitation candidate with independently owned episode state."""

import copy
from pathlib import Path
from typing import Any

import rosclaw.growth.bounded_residual_imitation as learner

from rosclaw_soccer.rsi.imitation_proposal_motor import CompiledImitationProposalMotor, make_preview
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def compilation_contract(policy: dict[str, Any]) -> dict[str, Any]:
    if (
        type(policy) is not dict
        or "verified_success_imitation_motor_proof" not in policy
        or type(policy.get("step_motor_proof")) is not dict
        or type(policy["step_motor_proof"].get("model")) is not dict
        or policy["step_motor_proof"]["model"].get("schema")
        != "soccer.rsi.verified_success_imitation_motor.v1"
        or policy.get("policy_hash")
        != hash_json({k: v for k, v in policy.items() if k != "policy_hash"})
    ):
        raise ValueError("complete sealed imitation policy required")
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += list(Path(learner.__file__).parent.glob("*.py"))
    paths += [Path(__file__).parents[1] / "sim/contracts.py"]
    return dict(
        schema="soccer.rsi.private_imitation_proposal_factory.v1",
        policy_hash=policy["policy_hash"],
        complete_canonical_model_hash=hash_json(policy["step_motor_proof"]["model"]),
        source_pins={str(p): hash_bytes(p.read_bytes()) for p in paths},
        complete_model_validation_at_allocation=True,
        independent_contact_history=True,
        private_fixed_plastic_head=True,
        physics_parity_requires_external_evidence=True,
        activation_ceiling="SIM_ONLY",
        promotion_authorized=False,
        hardware_authorized=False,
    )


def validate_compilation_contract(value: Any, policy: dict[str, Any]) -> None:
    if type(value) is not dict or hash_json(value) != hash_json(compilation_contract(policy)):
        raise ValueError("complete imitation factory provenance required")


class ImitationProposalEpisodeFactory:
    def __init__(self, model: dict[str, Any]) -> None:
        self._policy = make_preview(copy.deepcopy(model))
        self._canonical_model_hash = hash_json(self._policy["step_motor_proof"]["model"])
        self._prototype = CompiledImitationProposalMotor(self._policy)
        paths = list(Path(__file__).parent.glob("*.py"))
        paths += list(Path(learner.__file__).parent.glob("*.py"))
        paths += [Path(__file__).parents[1] / "sim/contracts.py"]
        self._pins = {str(p): hash_bytes(p.read_bytes()) for p in paths}

    def _stable(self) -> None:
        if any(hash_bytes(Path(p).read_bytes()) != h for p, h in self._pins.items()):
            raise ValueError("fixed imitation factory sources changed")

    @property
    def policy_hash(self) -> str:
        self._stable()
        return str(self._policy["policy_hash"])

    def preview(self, model: dict[str, Any]) -> dict[str, Any]:
        self._stable()
        if hash_json(model) != self._canonical_model_hash:
            raise ValueError("complete canonical imitation model changed")
        result: dict[str, Any] = copy.deepcopy(self._policy)
        return result

    def contract(self) -> dict[str, Any]:
        self._stable()
        result = compilation_contract(self._policy)
        if (
            result["source_pins"] != self._pins
            or result["policy_hash"] != self._prototype._policy_hash
            or result["complete_canonical_model_hash"] != self._canonical_model_hash
        ):
            raise ValueError("fixed imitation factory private commitment changed")
        return result

    def new_episode(self) -> CompiledImitationProposalMotor:
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
