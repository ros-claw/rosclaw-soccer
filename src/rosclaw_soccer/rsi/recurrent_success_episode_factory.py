"""Fixed sequence weights, fresh causal/contact state; never authorization."""

import copy
from pathlib import Path
from typing import Any

import rosclaw.growth.causal_residual_memory as memory_module
from rosclaw.growth.canonical_json_snapshot import CanonicalJSONSnapshot

from rosclaw_soccer.rsi.recurrent_success_motor import CompiledRecurrentSuccessMotor, make_preview
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def compilation_contract(policy: dict[str, Any]) -> dict[str, Any]:
    if (
        type(policy) is not dict
        or "recurrent_success_motor_proof" not in policy
        or type(policy.get("step_motor_proof")) is not dict
        or type(policy["step_motor_proof"].get("model")) is not dict
        or policy["step_motor_proof"]["model"].get("schema")
        != "soccer.rsi.recurrent_success_imitation_motor.v1"
        or policy.get("policy_hash")
        != hash_json({k: v for k, v in policy.items() if k != "policy_hash"})
    ):
        raise ValueError("complete canonical sequence policy required")
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += list(Path(memory_module.__file__).parent.glob("*.py"))
    paths += [Path(__file__).parents[1] / "sim/contracts.py"]
    return dict(
        schema="soccer.rsi.private_recurrent_success_factory.v1",
        policy_hash=policy["policy_hash"],
        complete_canonical_model_hash=hash_json(policy["step_motor_proof"]["model"]),
        source_pins={str(p): hash_bytes(p.read_bytes()) for p in paths},
        complete_original_model_validation_at_allocation=True,
        independent_contact_history=True,
        independent_recurrent_hidden_state=True,
        physical_action_bounds_changed=False,
        physics_parity_requires_external_evidence=True,
        activation_ceiling="SIM_ONLY",
        promotion_authorized=False,
        hardware_authorized=False,
    )


def validate_compilation_contract(value: Any, policy: dict[str, Any]) -> None:
    if type(value) is not dict or hash_json(value) != hash_json(compilation_contract(policy)):
        raise ValueError("complete source-bound causal sequence factory contract required")


class RecurrentSuccessEpisodeFactory:
    def __init__(self, model: dict[str, Any]) -> None:
        policy = make_preview(copy.deepcopy(model))
        self._prototype = CompiledRecurrentSuccessMotor(policy)
        self._canonical_model_hash = hash_json(policy["step_motor_proof"]["model"])
        contract = compilation_contract(policy)
        self._pins = contract["source_pins"]
        self._policy_snapshot = CanonicalJSONSnapshot(policy)
        self._contract_snapshot = CanonicalJSONSnapshot(contract)
        self._policy_hash = str(policy["policy_hash"])

    def _stable(self) -> None:
        if any(hash_bytes(Path(p).read_bytes()) != h for p, h in self._pins.items()):
            raise ValueError("fixed sequence factory sources changed")

    @property
    def policy_hash(self) -> str:
        self._stable()
        self._policy_snapshot.verify()
        return self._policy_hash

    def preview(self, model: dict[str, Any]) -> dict[str, Any]:
        self._stable()
        if hash_json(model) != self._canonical_model_hash:
            raise ValueError("complete canonical sequence model changed")
        result: dict[str, Any] = self._policy_snapshot.restore()
        return result

    def contract(self) -> dict[str, Any]:
        self._stable()
        self._policy_snapshot.verify()
        result: dict[str, Any] = self._contract_snapshot.restore()
        if (
            result["source_pins"] != self._pins
            or result["policy_hash"] != self._prototype._policy_hash
            or result["complete_canonical_model_hash"] != self._canonical_model_hash
        ):
            raise ValueError("fixed sequence factory private commitment changed")
        return result

    def new_episode(self) -> CompiledRecurrentSuccessMotor:
        self._stable()
        self._policy_snapshot.verify()
        result = copy.copy(self._prototype)
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
        result._behavior, result._parent = behavior, behavior._parent
        result._recurrent = self._prototype._recurrent.new_episode()
        result._memory = ContactPhaseMemory()
        result._active_frame = None
        return result
