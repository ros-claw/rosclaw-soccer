"""Software contract tests, not native physical success evidence."""

import copy
import json
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.receding_teacher_scope import receding_jobs
from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_read_mesh_contact_native_pilot as audit
from scripts import rsi_stage_receding_contact_replay as successor
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


def test_child_preflight_failure_is_persisted_and_same_stage_cannot_retry(tmp_path, monkeypatch):
    output = expanded_fixture(tmp_path, monkeypatch, receding=True)
    audit.main()
    producer_path = tmp_path / "teacher-config.json"
    producer = json.loads(producer_path.read_text())
    producer.update(mode="SIM_ONLY", hardware_authorized=False, promotion_authorized=False)
    producer_path.write_text(json.dumps(producer))
    root = Path(producer["output_root"])
    complete = json.loads((root / "complete.json").read_text())
    # The whole-reader fixture omits execution-only safety fields. Bind the
    # added synthetic fields consistently rather than bypassing byte checks.
    declaration_path = root / "declaration.json"
    declaration = json.loads(declaration_path.read_text())
    declaration["input_file_hashes"][str(producer_path)] = successor.digest(producer_path)

    def reseal(value, path):
        value["report_hash"] = hash_json({k: v for k, v in value.items() if k != "report_hash"})
        path.write_text(json.dumps(value))

    reseal(declaration, declaration_path)
    complete["declaration_hash"] = declaration["report_hash"]
    reseal(complete, root / "complete.json")
    readback = json.loads(output.read_text())
    readback["complete_hash"] = complete["report_hash"]
    reseal(readback, output)
    old_rows = [
        dict(case_id=f"{f}_context_{i}", source_receipt_hash=f"old-{i}-{f}")
        for i in range(2, 8)
        for f in ("none", "option4")
    ]
    for index, reused in zip((8, 10), complete["reused_endpoints"], strict=True):
        old_rows[index]["source_receipt_hash"] = reused["receipt"]["receipt_hash"]
    prior = dict(
        schema="soccer.rsi.fixed_control_contact_replay_complete.v1",
        integrated_substeps=36000,
        rows=old_rows,
    )
    prior["report_hash"] = hash_json(prior)
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps(prior))
    config = dict(
        source_head="frozen",
        deadline_unix=1e20,
        new_substep_budget=30000,
        producer_unit="native.service",
        audit_unit="audit.service",
        producer_config=str(producer_path),
        readback=str(output),
        audit_source=str(Path(audit.__file__)),
        prior_replay_complete=str(prior_path),
        prior_replay_complete_hash=prior["report_hash"],
    )
    for key in (
        "started_path",
        "summary_output",
        "replay_config",
        "replay_output_root",
        "metrics_output",
        "complete_path",
    ):
        config[key] = str(tmp_path / f"stage-{key}.json")
    config_path = tmp_path / "stage-config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(successor, "require_terminal", lambda unit: None)
    monkeypatch.setattr(
        successor.subprocess,
        "check_output",
        lambda args, **kw: "frozen\n" if "rev-parse" in args else "",
    )
    calls = []

    def fail_child(args, **kwargs):
        calls.append(args)
        raise RuntimeError("deliberate zero-allocation child preflight failure")

    monkeypatch.setattr(successor.subprocess, "run", fail_child)
    monkeypatch.setattr("sys.argv", ["stage", "--config", str(config_path)])
    with pytest.raises(RuntimeError, match="zero-allocation"):
        successor.main()
    failure = Path(config["started_path"]).with_suffix(".failure.json")
    assert json.loads(failure.read_text())["automatic_retry_authorized"] is False
    assert not Path(config["complete_path"]).exists()
    assert len(calls) == 1 and calls[0][-1] == "--preflight"
    with pytest.raises(ValueError, match="unused bounded"):
        successor.main()
    assert len(calls) == 1
