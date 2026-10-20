"""Metadata and cause classification fixtures, not physical review evidence."""

from types import SimpleNamespace

import pytest

from scripts import rsi_native_context_recovery_common as common


def test_exception_classification_requires_exact_leaf_type_and_message():
    reason = "proposed measured-joint context outside native joint limits; no clipping"
    inner = ValueError(reason)
    outer = ValueError("navigation proposal failed; option latched off")
    outer.__cause__ = inner
    assert common.policy_leaf_reason(outer) == reason
    assert common.cause_chain(outer) == [
        dict(type="ValueError", message=str(outer)),
        dict(type="ValueError", message=reason),
    ]
    for bad in (RuntimeError(reason), ValueError(reason + " altered"), FloatingPointError("NaN")):
        outer.__cause__ = bad
        assert common.policy_leaf_reason(outer) is None


def test_cause_cycles_are_fatal_not_whitelisted():
    error = ValueError("proposed measured-joint context outside native joint limits; no clipping")
    error.__cause__ = error
    with pytest.raises(ValueError, match="cyclic"):
        common.policy_leaf_reason(error)


def test_changed_input_hash_rejected(tmp_path):
    p = tmp_path / "input"
    p.write_bytes(b"original")
    pins = {str(p): common.digest(p)}
    common.check_pins(pins)
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="pinned source"):
        common.check_pins(pins)


@pytest.mark.parametrize("kind", ["source", "output"])
def test_preimport_source_output_admission(tmp_path, monkeypatch, kind):
    args = SimpleNamespace(output=tmp_path / "output", expected_head="expected")
    if kind == "output":
        args.output.mkdir()
    monkeypatch.setattr(
        common.subprocess,
        "check_output",
        lambda *a, **k: "wrong\n" if kind == "source" else "expected\n",
    )
    with pytest.raises(ValueError, match="sealed new source"):
        common.admit_inventory(args)
