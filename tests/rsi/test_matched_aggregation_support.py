"""Synthetic full chains; no real physical evidence or successful G1 claims."""

import json
import sys
from pathlib import Path

import pytest

from rosclaw_soccer.rsi import matched_aggregation_support as support
from rosclaw_soccer.rsi.reference_exam_profiles import ZERO, assess_reference
from rosclaw_soccer.rsi.reference_student_exam_diagnostics import measured_readout
from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_compare_matched_references as comparison_stage
from tests.rsi.test_matched_reference_comparison_stage import fixture as comparison_fixture
from tests.rsi.test_reference_aggregation_decision import results


def save(path, value, key="report_hash"):
    value[key] = hash_json({k: v for k, v in value.items() if k != key})
    Path(path).write_text(json.dumps(value))


def make_zero_failures(cfg):
    """Reseal an entirely synthetic fixture, never alter real source evidence."""
    spec = cfg["cohorts"]["zero"]
    cc = json.loads(Path(spec["contact_configuration"]).read_text())
    pc = json.loads(Path(cc["producer_config"]).read_text())
    root = Path(pc["output_root"])
    d, c, a, r, m = [
        json.loads(p.read_text())
        for p in (
            root / "declaration.json",
            root / "complete.json",
            Path(cc["readback"]),
            Path(cc["replay_output_root"]) / "complete.json",
            Path(cc["metrics_output"]),
        )
    ]
    for i, receipt in enumerate(c["results"]):
        if receipt["role"] == "zero":
            receipt["outcome"] = dict(task_pass=False, safe=True)
        save(root / f"receipt-{i:04d}.json", receipt, "receipt_hash")
        a["rows"][i].update(receipt_hash=receipt["receipt_hash"], outcome=receipt["outcome"])
        for rows in (r["rows"], m["rows"]):
            rows[i].update(
                source_receipt_hash=receipt["receipt_hash"], original_outcome=receipt["outcome"]
            )
    pairs = assess_reference(
        ZERO,
        d["scope"],
        c["results"],
        teachers={int(k): v["outcome"] for k, v in d["reused_teachers"].items()},
        original_retention={},
    )
    c["paired_results"] = a["paired_results"] = pairs
    save(root / "complete.json", c)
    a["complete_hash"] = c["report_hash"]
    save(cc["readback"], a)
    save(Path(cc["replay_output_root"]) / "complete.json", r)
    save(cc["metrics_output"], m)
    dense = json.loads(Path(cc["complete_path"]).read_text())
    dense.update(
        upstream_readback_hash=a["report_hash"],
        replay_complete_hash=r["report_hash"],
        metrics_hash=m["report_hash"],
        paired_results=pairs,
        measured_contact_readout=measured_readout(d, c, r, m),
    )
    save(cc["complete_path"], dense)


def fixture(tmp_path, monkeypatch, supported=False):
    cfg, cp, _ = comparison_fixture(tmp_path, monkeypatch)
    if supported:
        make_zero_failures(cfg)
    monkeypatch.setattr(sys, "argv", ["comparison", "--config", str(cp)])
    comparison_stage.main()
    checked = []
    monkeypatch.setattr(support, "require_terminal", checked.append)
    pins = {}

    def sealed(path, key="report_hash"):
        path = Path(path)
        v = json.loads(path.read_text())
        if v[key] != hash_json({k: x for k, x in v.items() if k != key}):
            raise ValueError("test original seal changed")
        pins[str(path)] = support.digest(path)
        return v

    report_path = Path(cfg["output_root"]) / "complete.json"
    report = sealed(report_path)
    spec = dict(
        configuration=str(cp), configuration_file_hash=support.digest(cp), unit="mock-comparison"
    )
    kw = dict(
        current_contact_configuration=cfg["cohorts"]["student"]["contact_configuration"],
        current_complete_hash=report["cohort_complete_hashes"]["student"],
        sealed=sealed,
        pins=pins,
    )
    return spec, kw, report_path, checked


@pytest.mark.parametrize("supported", [False, True])
def test_reconstructs_whole_matching_physical_chain_not_a_boolean_approval(
    tmp_path, monkeypatch, supported
):
    spec, kw, _, checked = fixture(tmp_path, monkeypatch, supported)
    result = support.require_matched_support(spec, **kw)
    assert result["considers_one_numeric_research_fit"] is supported
    assert len(result["expert_advantage_training_sources"]) == (3 if supported else 0)
    assert checked == ["mock-comparison"]
    assert not result["validation_used_to_select_fit"] and not result["neural_training_authorized"]


@pytest.mark.parametrize(
    "tamper",
    [
        "config",
        "current_config",
        "current_complete",
        "report",
        "approval",
        "input",
        "spec",
        "source",
        "terminal",
    ],
)
def test_bad_or_borrowed_matched_report_cannot_motivate_a_fit(tmp_path, monkeypatch, tamper):
    spec, kw, path, _ = fixture(tmp_path, monkeypatch)
    report = json.loads(path.read_text())
    if tamper == "config":
        Path(spec["configuration"]).write_text("{}")
    elif tamper == "current_config":
        kw["current_contact_configuration"] = str(tmp_path / "other-generation")
    elif tamper == "current_complete":
        kw["current_complete_hash"] = "other-generation"
    elif tamper == "report":
        report["new_physical_steps"] = 1
        save(path, report)
    elif tamper == "approval":
        report["comparison"]["next_research"] = (
            "BOUNDED_SAME_STATE_EXPERT_SUPERVISION_RESEARCH_HAS_TRAIN_SOURCE_SUPPORT"
        )
        save(path, report)
    elif tamper == "input":
        report["input_file_hashes"][spec["configuration"]] = "sha256:" + "0" * 64
        save(path, report)
    elif tamper == "spec":
        spec["approved"] = True
    elif tamper == "source":
        monkeypatch.setattr(
            support.subprocess,
            "check_output",
            lambda cmd, **kw: "borrowed" if "rev-parse" in cmd else "",
        )
    elif tamper == "terminal":

        def fail(_):
            raise RuntimeError("still active")

        monkeypatch.setattr(support, "require_terminal", fail)
    with pytest.raises((ValueError, RuntimeError)):
        support.require_matched_support(spec, **kw)


@pytest.mark.parametrize("eligible", [False, True])
def test_matched_support_can_only_reduce_existing_finite_fit_recommendation(eligible):
    from rosclaw_soccer.rsi.reference_aggregation_decision import recommend_aggregation

    original = recommend_aggregation(results(), positive_dense_nonfoot=0)
    decision = support.constrain_recommendation(
        original, dict(considers_one_numeric_research_fit=eligible)
    )
    assert not decision["finite_numeric_fit_recommended"]
    assert decision["maximum_optimizer_steps_per_candidate"] == 128
