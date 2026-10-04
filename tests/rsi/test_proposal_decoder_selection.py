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


def test_bounded_selection_pins_core_and_preserves_default_contract(monkeypatch):
    from rosclaw_soccer.rsi import bounded_query_proposal_compilation as bounded

    policy = {"sealed_fixture": True}
    monkeypatch.setattr(bounded, "compile_bounded_query_proposal", lambda p: p)
    assert selection.select_proposal_decoder(policy, implementation="bounded_snapshot") is policy
    value = selection.compilation_contract("bounded_snapshot")
    assert value["complete_logical_anchor_banks_retained"] is True
    assert value["physical_parity_requires_separate_evidence"] is True
    assert value["actor_weights_changed"] is value["hardware_authorized"] is False
    selection.validate_compilation_contract(value)
    assert "core_query_source_hash" not in selection.compilation_contract("owned_snapshot")
    for key in ("core_query_source_hash", "original_snapshot_compiler_source_hash"):
        forged = copy.deepcopy(value)
        forged[key] = "sha256:" + "f" * 64
        with pytest.raises(ValueError, match="compilation contract"):
            selection.validate_compilation_contract(forged)
