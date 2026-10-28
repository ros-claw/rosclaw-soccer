import json

import pytest

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_watch_return_smoothing_compare import check_receipt


def test_receipt_requires_expected_completed_development_trial(tmp_path):
    path = tmp_path / "complete.json"
    data = dict(
        status="READY",
        source_head="head",
        source_hash="source",
        rows=[{}] * 24,
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
        independent_fresh=False,
    )
    path.write_text(json.dumps(data))
    assert check_receipt(path, "READY", "head", "source") == digest(path)
    data["promotion_authorized"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unpromoted"):
        check_receipt(path, "READY", "head", "source")


def test_missing_or_wrong_source_fails_closed(tmp_path):
    path = tmp_path / "complete.json"
    with pytest.raises(ValueError, match="completed receipt"):
        check_receipt(path, "READY", "head", "source")
    path.write_text(json.dumps({"status": "READY", "source_head": "head", "source_hash": "other"}))
    with pytest.raises(ValueError, match="unpromoted"):
        check_receipt(path, "READY", "head", "source")
