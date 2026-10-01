import pytest

from scripts.rsi_transport_equivalence import compare_transport


def fixture():
    old = dict(
        source_hash="old", report_hash="old-report", parent_report_hash="old-parent", frames=300
    )
    new = {
        **old,
        "source_hash": "new",
        "report_hash": "new-report",
        "parent_report_hash": "new-parent",
    }
    outcome = dict(command_audit_hash="sha256:" + "a" * 64, high_quality=True, forward_m=1.5)
    current = {**outcome, "command_audit_hash": "sha256:" + "b" * 64}
    return new, old, current, outcome


def test_distinct_provenance_is_explicit_not_falsely_equal():
    result = compare_transport(*fixture())
    assert result["measured_physical_fields_equal"] is True
    assert result["provenance_hashes_equal"] is False
    assert result["reference_command_audit_hash"] != result["current_command_audit_hash"]


@pytest.mark.parametrize("key,value", [("frames", 299), ("extra_field", 1)])
def test_any_nonprovenance_raw_change_is_rejected(key, value):
    new, old, current, outcome = fixture()
    new[key] = value
    with pytest.raises(ValueError, match="raw physical"):
        compare_transport(new, old, current, outcome)


@pytest.mark.parametrize(
    "key,value", [("forward_m", 1.5000000001), ("high_quality", False), ("extra", 1)]
)
def test_any_measured_change_is_rejected_without_tolerance(key, value):
    new, old, current, outcome = fixture()
    current[key] = value
    with pytest.raises(ValueError, match="physical outcome"):
        compare_transport(new, old, current, outcome)


def test_unbound_audit_identity_is_rejected():
    new, old, current, outcome = fixture()
    current["command_audit_hash"] = "invalid"
    with pytest.raises(ValueError, match="audit identities"):
        compare_transport(new, old, current, outcome)
