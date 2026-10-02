from copy import deepcopy

import pytest

from scripts.rsi_validate_physical_report_transport import compare_complete_reports


def test_same_source_complete_report_and_outcome_required() -> None:
    report = {"source_hash": "runner", "report_hash": "seal", "actions": [0.0, 1e-12]}
    outcome = {"command_audit_hash": "audit", "reward": 3.5}
    compare_complete_reports(report, deepcopy(report), outcome, deepcopy(outcome))


@pytest.mark.parametrize("field", ["source_hash", "report_hash", "actions"])
def test_no_provenance_or_numerical_exception(field: str) -> None:
    report = {"source_hash": "runner", "report_hash": "seal", "actions": [0.0, 1e-12]}
    changed = deepcopy(report)
    changed[field] = "different"
    with pytest.raises(ValueError, match="complete numerical"):
        compare_complete_reports(report, changed, {}, {})


def test_independent_outcome_change_is_rejected() -> None:
    with pytest.raises(ValueError, match="reconstructed"):
        compare_complete_reports({}, {}, {"reward": 3}, {"reward": 4})
