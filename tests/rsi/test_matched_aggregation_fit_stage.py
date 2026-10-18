"""Guarded stage with physical audits, support authentication and fit mocked.

Pure matched-support authentication has separate whole-source synthetic tests.
No real MuJoCo/128-step optimizer evidence is claimed by this orchestration.
"""

import json
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.matched_aggregation_support import PROFILE
from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_fit_reference_state_aggregation as stage
from tests.rsi.test_reference_aggregation_fit_stage import fixture as original_fixture


@pytest.mark.parametrize(
    "mode",
    [
        "fit",
        "no_support",
        "wrong_generation",
        "missing_generation",
        "missing_support",
        "support_failure",
        "preflight",
        "failure",
    ],
)
def test_current_generation_requires_whole_matched_expert_benefit_before_any_optimizer(
    tmp_path, monkeypatch, mode
):
    root, calls = original_fixture(tmp_path, monkeypatch, "failure" if mode == "failure" else "fit")
    cp = tmp_path / "config.json"
    cfg = json.loads(cp.read_text())
    cfg.update(
        profile=PROFILE, matched_comparison=dict(mock="authenticator is mocked in this test")
    )
    prepared_path = Path(cfg["prepared_aggregation"])
    prepared = json.loads(prepared_path.read_text())
    prepared.update(
        actual_behavior_model_hash="current-model",
        actual_behavior_fit_report_hash="current-fit",
        bounded_current_own_state_window_not_cumulative_dagger=True,
    )
    if mode == "wrong_generation":
        prepared["actual_behavior_model_hash"] = "old-generation"
    elif mode == "missing_generation":
        del prepared["actual_behavior_model_hash"]
    elif mode == "missing_support":
        del cfg["matched_comparison"]
    prepared["report_hash"] = hash_json({k: v for k, v in prepared.items() if k != "report_hash"})
    prepared_path.write_text(json.dumps(prepared))
    cfg["prepared_aggregation_hash"] = prepared["report_hash"]
    cp.write_text(json.dumps(cfg))
    declaration_path = tmp_path / "closed/declaration.json"
    d = json.loads(declaration_path.read_text())
    d.update(models=dict(student=dict(model_hash="current-model")), fit_report_hash="current-fit")
    d["report_hash"] = hash_json({k: v for k, v in d.items() if k != "report_hash"})
    declaration_path.write_text(json.dumps(d))
    checked = []

    def authenticated(spec, **kw):
        checked.append(kw)
        assert kw["current_contact_configuration"] == cfg["current_contact_configuration"]
        # Real whole-cohort loading reads receipt_hash as well as report_hash.
        # Keep this callback contract even when authentication is mocked here.
        receipt_path = tmp_path / "typed-seal-probe.json"
        receipt = dict(value="synthetic receipt, not physical evidence")
        receipt["receipt_hash"] = hash_json(receipt)
        receipt_path.write_text(json.dumps(receipt))
        assert kw["sealed"](receipt_path, "receipt_hash") == receipt
        if mode == "support_failure":
            raise ValueError("mock authentication failed")
        return dict(
            considers_one_numeric_research_fit=mode != "no_support",
            matched_comparison_hash="mock-proof",
        )

    monkeypatch.setattr(stage, "require_matched_support", authenticated)
    if mode == "preflight":
        import sys

        monkeypatch.setattr(sys, "argv", ["fit", "--config", str(cp), "--preflight"])
    if mode in ("wrong_generation", "missing_generation", "missing_support", "support_failure"):
        with pytest.raises(ValueError):
            stage.main()
        assert not root.exists() and not calls
        assert len(checked) == (1 if mode == "support_failure" else 0)
        return
    if mode == "failure":
        with pytest.raises(RuntimeError, match="mock optimizer"):
            stage.main()
        assert (root / "failure.json").exists() and not (root / "complete.json").exists()
    else:
        stage.main()
        if mode == "preflight":
            assert not root.exists() and not calls
            return
        report = json.loads((root / "complete.json").read_text())
        assert report["optimizer_updates"] == (0 if mode == "no_support" else 128)
        assert len(calls) == (0 if mode == "no_support" else 1)
        assert report["decision"]["matched_support"]["considers_one_numeric_research_fit"] == (
            mode != "no_support"
        )
        assert (
            not report["dataset_admission_authorized"] and not report["own_body_student_evaluated"]
        )
    with pytest.raises(ValueError, match="one matched"):
        stage.main()
    assert len(calls) <= 1
