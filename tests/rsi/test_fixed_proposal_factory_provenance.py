"""Complete numeric provenance; synthetic contracts are not physical proof."""

import copy

import pytest

from rosclaw_soccer.rsi.proposal_episode_decoder_factory import (
    compilation_contract,
    validate_compilation_contract,
)
from rosclaw_soccer.sim.contracts import hash_json


def policy():
    value = {"proposal_memory_motor_proof": {}, "payload": {"joint": 0.0}}
    return {**value, "policy_hash": hash_json(value)}


def test_complete_contract_and_policy_seal_checked():
    proposed = policy()
    contract = compilation_contract(proposed["policy_hash"])
    validate_compilation_contract(contract, proposed)
    assert contract["physics_parity_requires_external_evidence"] is True
    assert contract["hardware_authorized"] is False
    proposed["payload"]["joint"] = -0.0
    with pytest.raises(ValueError):
        validate_compilation_contract(contract, proposed)


@pytest.mark.parametrize(
    "fault", ["source", "identity", "authority", "extra", "missing", "false_int"]
)
def test_forged_or_partial_factory_claim_rejected(fault):
    proposed = policy()
    claim = copy.deepcopy(compilation_contract(proposed["policy_hash"]))
    if fault == "source":
        claim["source_pins"][next(iter(claim["source_pins"]))] = "sha256:" + "a" * 64
    elif fault == "identity":
        claim["policy_hash"] = "sha256:" + "b" * 64
    elif fault == "authority":
        claim["hardware_authorized"] = True
    elif fault == "extra":
        claim["bypass"] = True
    elif fault == "missing":
        claim.pop("source_pins")
    else:
        claim["actor_weights_changed"] = 0
    with pytest.raises(ValueError):
        validate_compilation_contract(claim, proposed)


@pytest.mark.parametrize("identity", [None, True, "sha256:x", "a" * 64])
def test_invalid_fixed_identity_rejected(identity):
    with pytest.raises(ValueError):
        compilation_contract(identity)
