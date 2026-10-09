"""Fixed deterministic SIM episodes; native parity is an external prerequisite.

Validate the complete actor, critic, ancestry and original decoder once. Own
the canonical policy privately and reset every causal state per episode. This
is never an independent auditor and never authorizes policy activation.
"""

import copy
from pathlib import Path
from typing import Any

import rosclaw.growth.canonical_json_snapshot as snapshot_module
from rosclaw.growth.canonical_json_snapshot import CanonicalJSONSnapshot

from rosclaw_soccer.rsi.fixed_recurrent_reference_audit import _numeric_graph_hash
from rosclaw_soccer.rsi.recurrent_clipped_motor import CompiledRecurrentClippedMotor, make_preview
from rosclaw_soccer.rsi.recurrent_success_episode_factory import RecurrentSuccessEpisodeFactory
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def compilation_contract(policy: dict[str, Any]) -> dict[str, Any]:
    if (
        type(policy) is not dict
        or "recurrent_clipped_motor_proof" not in policy
        or type(policy.get("step_motor_proof")) is not dict
        or type(policy["step_motor_proof"].get("model")) is not dict
        or policy["step_motor_proof"]["model"].get("schema")
        != "soccer.rsi.recurrent_clipped_mc_motor.v1"
        or policy.get("policy_hash")
        != hash_json({k: v for k, v in policy.items() if k != "policy_hash"})
    ):
        raise ValueError("complete original deterministic clipped policy required")
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += list(Path(snapshot_module.__file__).parent.glob("*.py"))
    paths += [Path(__file__).parents[1] / "sim/contracts.py"]
    return dict(
        schema="soccer.rsi.private_recurrent_clipped_episode_factory.v1",
        policy_hash=policy["policy_hash"],
        complete_canonical_model_hash=hash_json(policy["step_motor_proof"]["model"]),
        source_pins={str(path): hash_bytes(path.read_bytes()) for path in paths},
        original_complete_decoder_validation_at_allocation=True,
        complete_actor_critic_and_ancestry_retained=True,
        independent_contact_history=True,
        independent_recurrent_hidden_state=True,
        fixed_initial_numeric_graph_verified_before_each_episode=True,
        physical_action_bounds_changed=False,
        physics_parity_requires_external_evidence=True,
        native_transport_qualified_here=False,
        independent_auditor=False,
        activation_ceiling="SIM_ONLY",
        runtime_execution_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )


def validate_compilation_contract(value: Any, policy: dict[str, Any]) -> None:
    if type(value) is not dict or hash_json(value) != hash_json(compilation_contract(policy)):
        raise ValueError("complete non-authorizing deterministic factory contract required")


class RecurrentClippedEpisodeFactory(RecurrentSuccessEpisodeFactory):
    """Reuse only verified constants; never reuse progressed motor memory.

    The original clipped decoder inherits the same causal episode interface from
    the original success decoder. Reuse that existing fresh-state implementation,
    not a second imitation of its reset law. The auditor must own another decoder.
    """

    def __init__(self, model: dict[str, Any]) -> None:
        policy = make_preview(copy.deepcopy(model))
        self._prototype = CompiledRecurrentClippedMotor(policy)
        self._prototype_hash = _numeric_graph_hash(self._prototype)
        self._canonical_model_hash = hash_json(policy["step_motor_proof"]["model"])
        contract = compilation_contract(policy)
        self._pins = contract["source_pins"]
        self._policy_snapshot = CanonicalJSONSnapshot(policy)
        self._contract_snapshot = CanonicalJSONSnapshot(contract)
        self._policy_hash = str(policy["policy_hash"])
        self._stable()

    def _stable(self) -> None:
        super()._stable()
        if _numeric_graph_hash(self._prototype) != self._prototype_hash:
            raise ValueError("fixed deterministic factory prototype changed")

    def new_episode(self) -> CompiledRecurrentClippedMotor:
        result = super().new_episode()
        if type(result) is not CompiledRecurrentClippedMotor:
            raise ValueError("original deterministic clipped episode required")
        return result
