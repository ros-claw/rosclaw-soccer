"""Software contract tests, not native physical success evidence."""

import copy
import json
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.receding_teacher_scope import receding_jobs
from scripts import rsi_read_mesh_contact_native_pilot as audit
from scripts.rsi_replay_first_touch_contacts import fixed_replay_scope
from scripts.rsi_stage_receding_contact_replay import (
    require_dense_union,
    require_receding_endpoints,
)
from tests.rsi.test_mesh_teacher_terminal_readback import expanded_fixture


@pytest.mark.parametrize("tamper", [None, "partial", "order", "budget", "bool", "duplicate"])
def test_ten_receding_diagnostics_are_not_an_unbounded_or_repeated_policy_study(tamper):
    cfg = dict(
        replay_profile="receding-catalog2-ten",
        new_replay_budget=10,
        new_substep_budget=30000,
        cases=[
            dict(case_id=j["job"].replace("-", "_"), receipt_hash=str(i))
            for i, j in enumerate(receding_jobs())
        ],
    )
    if tamper == "partial":
        cfg["cases"].pop()
    elif tamper == "order":
        cfg["cases"].reverse()
    elif tamper == "budget":
        cfg["new_substep_budget"] += 3000
    elif tamper == "bool":
        cfg["new_replay_budget"] = True
    elif tamper == "duplicate":
        cfg["cases"][1]["receipt_hash"] = cfg["cases"][0]["receipt_hash"]
    if tamper:
        with pytest.raises(ValueError):
            fixed_replay_scope(cfg)
    else:
        assert fixed_replay_scope(cfg) == (10, 30000)


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "partial",
        "order",
        "outcome",
        "reuse",
        "pairing",
        "fresh",
        "teacher",
        "world",
        "reproduction",
        "truthy",
    ],
)
def test_post_audit_requires_whole_original_failure_bank(tmp_path, monkeypatch, tamper):
    output = expanded_fixture(tmp_path, monkeypatch, receding=True)
    audit.main()
    report = json.loads(output.read_text())
    config = json.loads((tmp_path / "teacher-config.json").read_text())
    config.update(mode="SIM_ONLY", hardware_authorized=False, promotion_authorized=False)
    # The fixture's producer root is deliberately read from its own manifest.
    root = Path(config["output_root"])
    declaration = json.loads((root / "declaration.json").read_text())
    complete = json.loads((root / "complete.json").read_text())
    expected = copy.deepcopy(complete["paired_development"])
    if tamper == "partial":
        complete["results"].pop()
    elif tamper == "order":
        report["rows"].reverse()
    elif tamper == "outcome":
        report["rows"][0]["outcome"]["task_pass"] = not report["rows"][0]["outcome"]["task_pass"]
    elif tamper == "reuse":
        complete["reused_endpoints"] = []
    elif tamper == "pairing":
        report["paired_development"]["evaluated_passes"] = 999
    elif tamper == "fresh":
        report["fresh_exam"] = True
    elif tamper == "teacher":
        report["teacher_qualified"] = True
    elif tamper == "world":
        report["rows"][0]["world_hash"] = "other"
    elif tamper == "reproduction":
        report["archived_c7_full_numeric_reproduction"] = False
    elif tamper == "truthy":
        config["max_attempts"] = True
    if tamper:
        with pytest.raises(ValueError):
            require_receding_endpoints(config, declaration, complete, report)
    else:
        assert require_receding_endpoints(config, declaration, complete, report) == expected


@pytest.mark.parametrize(
    "tamper", [None, "missing", "duplicate", "outcome", "replay", "budget", "reuse"]
)
def test_dense_union_selects_twelve_scoped_endpoints_not_twenty_two_courses(tamper):
    def row(identity):
        return dict(source_receipt_hash=identity, original_outcome=dict(task_pass=False, safe=True))

    prior = dict(rows=[row(f"old{i}") for i in range(12)])
    fresh = [row(f"new{i}") for i in range(10)]
    complete = dict(
        results=[
            dict(receipt_hash=r["source_receipt_hash"], outcome=r["original_outcome"])
            for r in fresh
        ],
        reused_endpoints=[dict(receipt=dict(receipt_hash=f"old{i}")) for i in (8, 10)],
    )
    replay = dict(
        schema="soccer.rsi.fixed_control_contact_replay_complete.v1",
        rows=fresh,
        integrated_substeps=30000,
        attempted_step_calls=30000,
    )
    metrics = dict(
        schema="soccer.rsi.contact_exit_metrics_readback.v1",
        rows=copy.deepcopy(fresh + prior["rows"]),
    )
    if tamper == "missing":
        metrics["rows"].pop()
    elif tamper == "duplicate":
        metrics["rows"][-1] = metrics["rows"][0]
    elif tamper == "outcome":
        metrics["rows"][0]["original_outcome"]["task_pass"] = True
    elif tamper == "replay":
        replay["rows"] = list(reversed(fresh))
    elif tamper == "budget":
        replay["integrated_substeps"] = 33000
    elif tamper == "reuse":
        complete["reused_endpoints"][1] = complete["reused_endpoints"][0]
    if tamper:
        with pytest.raises(ValueError):
            require_dense_union(complete, prior, replay, metrics)
    else:
        selected = require_dense_union(complete, prior, replay, metrics)
        assert len(selected) == 12
        assert [r["source_receipt_hash"] for r in selected[:2]] == ["old8", "old10"]
