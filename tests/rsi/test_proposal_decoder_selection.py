import copy

import pytest

from rosclaw_soccer.rsi import proposal_decoder_selection as selection


def test_reference_is_default_metadata_free_and_selection_is_explicit(monkeypatch):
    assert selection.compilation_contract("reference") is None
    policy = {"sealed_fixture": True}
    calls = []
    monkeypatch.setattr(
        selection, "CompiledProposalMemoryMotor", lambda p: calls.append(("reference", p)) or "old"
    )
    monkeypatch.setattr(
        selection.snapshot_module,
        "compile_proposal_snapshot",
        lambda p: calls.append(("owned", p)) or "new",
    )
    assert selection.select_proposal_decoder(policy, implementation="reference") == "old"
    assert selection.select_proposal_decoder(policy, implementation="owned_snapshot") == "new"
    assert calls == [("reference", policy), ("owned", policy)]


@pytest.mark.parametrize("implementation", [None, True, "", "fast", "hardware"])
def test_unknown_selection_rejected_before_decoder_allocation(monkeypatch, implementation):
    monkeypatch.setattr(
        selection, "CompiledProposalMemoryMotor", lambda _: pytest.fail("must reject")
    )
    monkeypatch.setattr(
        selection.snapshot_module, "compile_proposal_snapshot", lambda _: pytest.fail("must reject")
    )
    with pytest.raises(ValueError, match="known proposal"):
        selection.select_proposal_decoder({}, implementation=implementation)


@pytest.mark.parametrize(
    "fault", ["compiler", "selection", "authority", "false-as-zero", "extra", "null"]
)
def test_full_compiler_contract_rejects_resealed_drift(fault):
    contract = selection.compilation_contract("owned_snapshot")
    selection.validate_compilation_contract(contract)
    bad = copy.deepcopy(contract)
    if fault == "null":
        bad = None
    elif fault == "compiler":
        bad["compiler_source_hash"] = "sha256:" + "f" * 64
    elif fault == "selection":
        bad["selection_source_hash"] = "sha256:" + "f" * 64
    elif fault == "authority":
        bad["hardware_authorized"] = True
    elif fault == "false-as-zero":
        bad["actor_weights_changed"] = 0
    else:
        bad["unbound"] = True
    with pytest.raises(ValueError, match="compilation contract"):
        selection.validate_compilation_contract(bad)
