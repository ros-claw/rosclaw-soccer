"""Synthetic fail-closed diagnostic tests, never physical-success evidence."""

import copy
import json
import subprocess
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.receding_teacher_scope import receding_jobs
from rosclaw_soccer.rsi.surface_receding_diagnostics import (
    require_endpoints,
    research_decision,
    scoped_metrics,
)
from rosclaw_soccer.rsi.surface_receding_evidence import surface_receding_jobs
from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_read_mesh_contact_native_pilot as audit
from scripts import rsi_stage_surface_receding_contacts as stage
from scripts.rsi_replay_first_touch_contacts import fixed_replay_scope
from tests.rsi.test_release_four_full_terminal_readback import fixture


@pytest.mark.parametrize("tamper", [None, "partial", "order", "budget", "bool", "duplicate"])
def test_exact_six_replay_scope(tamper):
    cfg = dict(
        replay_profile="surface-receding-six",
        new_replay_budget=6,
        new_substep_budget=18000,
        cases=[
            dict(case_id=j["job"].replace("-", "_"), receipt_hash=str(i))
            for i, j in enumerate(surface_receding_jobs())
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
        assert fixed_replay_scope(cfg) == (6, 18000)


@pytest.mark.parametrize(
    "tamper", [None, "partial", "order", "outcome", "teacher", "fresh", "prefix", "clusters"]
)
def test_whole_audited_endpoints_required(tmp_path, monkeypatch, tamper):
    output = fixture(tmp_path, monkeypatch, surface_receding=True)
    audit.main()
    report = json.loads(output.read_text())
    root = tmp_path / "release"
    # Locate the synthetic producer, rather than assuming a real evidence path.
    paths = list(tmp_path.rglob("config.json"))
    producer_path = next(
        p for p in paths if json.loads(p.read_text()).get("surface_profile") == "receding-six"
    )
    root = producer_path.parent
    cfg = json.loads(producer_path.read_text())
    cfg.update(mode="SIM_ONLY", hardware_authorized=False, promotion_authorized=False)
    declaration = json.loads((root / "declaration.json").read_text())
    complete = json.loads((root / "complete.json").read_text())
    # Historical synthetic None-C6/C7 alias raw group4. This test of the pure
    # structural contract needs explicit group identities, not physical seals.
    for doc in (declaration, complete):
        for job, row in zip(
            surface_receding_jobs(), doc["reused_original_none_endpoints"], strict=True
        ):
            row["receipt"]["group"] = job["group"]
    if tamper == "partial":
        complete["results"].pop()
    elif tamper == "order":
        report["rows"].reverse()
    elif tamper == "outcome":
        report["rows"][0]["outcome"]["task_pass"] = not report["rows"][0]["outcome"]["task_pass"]
    elif tamper == "teacher":
        report["teacher_qualified"] = True
    elif tamper == "fresh":
        report["fresh_exam"] = True
    elif tamper == "prefix":
        report["all_original_pre_activation_prefixes_bit_equal"] = False
    elif tamper == "clusters":
        declaration["source_scene_clusters"] = 6
    if tamper:
        with pytest.raises(ValueError):
            require_endpoints(cfg, declaration, complete, report)
    else:
        paired, retention = require_endpoints(cfg, declaration, complete, report)
        assert paired == complete["paired_development"]
        assert retention == complete["original_none_context_retention"]


def dense_fixture():
    def row(name):
        return dict(
            source_receipt_hash=name,
            original_outcome=dict(task_pass=False, safe=True),
            metrics=dict(
                total_robot_downward_impulse_n_s=0.2,
                maximum_linear_newton_residual_n=1e-12,
                first_nonfoot_robot_contact_substep=None,
            ),
        )

    old = [row(f"old{i}") for i in range(22)]
    fresh = [
        dict(row(f"new{i}"), case_id=j["job"].replace("-", "_"))
        for i, j in enumerate(surface_receding_jobs())
    ]
    complete = dict(
        results=[
            dict(receipt_hash=r["source_receipt_hash"], outcome=r["original_outcome"])
            for r in fresh
        ],
        reused_endpoints=[dict(receipt=dict(receipt_hash=f"old{i}")) for i in range(6)],
        reused_original_none_endpoints=[
            dict(receipt=dict(receipt_hash=f"old{i}")) for i in range(6, 12)
        ],
    )
    old_replays = [
        dict(rows=old[:10], integrated_substeps=30000),
        dict(rows=old[10:], integrated_substeps=36000),
    ]
    replay = dict(
        schema="soccer.rsi.fixed_control_contact_replay_complete.v1",
        rows=fresh,
        integrated_substeps=18000,
        attempted_step_calls=18000,
    )
    metrics = dict(
        schema="soccer.rsi.contact_exit_metrics_readback.v1", rows=copy.deepcopy(old + fresh)
    )
    return complete, old_replays, replay, metrics


@pytest.mark.parametrize(
    "tamper", [None, "missing", "duplicate", "outcome", "order", "nan", "newton"]
)
def test_dense_union_preserves_failures_and_selects_six_times_three(tamper):
    complete, old, replay, metrics = dense_fixture()
    if tamper == "missing":
        metrics["rows"].pop()
    elif tamper == "duplicate":
        metrics["rows"][-1] = metrics["rows"][0]
    elif tamper == "outcome":
        metrics["rows"][0]["original_outcome"]["task_pass"] = True
    elif tamper == "order":
        replay["rows"].reverse()
    elif tamper == "nan":
        metrics["rows"][0]["metrics"]["total_robot_downward_impulse_n_s"] = float("nan")
    elif tamper == "newton":
        metrics["rows"][0]["metrics"]["maximum_linear_newton_residual_n"] = 0.01
    if tamper:
        with pytest.raises(ValueError):
            scoped_metrics(complete, old, replay, metrics)
    else:
        selected = scoped_metrics(complete, old, replay, metrics)
        assert len(selected) == 18
        assert [r["family"] for r in selected[:3]] == [
            "none",
            "body_origin_catalog2",
            "material_point_catalog2",
        ]


@pytest.mark.parametrize("condition", ["gain", "plateau", "regression", "unsafe", "nonfoot"])
def test_decision_never_auto_admits_even_when_gain(condition):
    selected = scoped_metrics(*dense_fixture())
    paired = dict(regressed_pairs=0, newly_unsafe_pairs=0, net_success_gain=1)
    if condition == "plateau":
        paired["net_success_gain"] = 0
    elif condition == "regression":
        paired["regressed_pairs"] = 1
    elif condition == "unsafe":
        selected[2]["dense_row"]["original_outcome"]["safe"] = False
    elif condition == "nonfoot":
        selected[2]["dense_row"]["metrics"]["first_nonfoot_robot_contact_substep"] = 1000
    result = research_decision(paired, selected)
    assert not any(
        result[k]
        for k in (
            "teacher_qualified",
            "dataset_admission_authorized",
            "neural_training_authorized",
            "promotion_authorized",
            "hardware_authorized",
        )
    )
    expected = (
        "MECHANISM_GAIN"
        if condition == "gain"
        else "NO_NET_GAIN"
        if condition == "plateau"
        else "REJECT_DECLARED_TEACHER"
    )
    assert result["decision"].startswith(expected)


def test_failed_child_is_journaled_and_cannot_automatically_retry(tmp_path, monkeypatch):
    # This exercises orchestration only. Whole-stage and upstream numerical
    # qualification are explicitly mocked; no NN or physics is called.
    def save(name, value, key="report_hash"):
        path = tmp_path / name
        if key:
            value[key] = hash_json({k: v for k, v in value.items() if k != key})
        path.write_text(json.dumps(value))
        return path

    complete, old, _, _ = dense_fixture()
    producer = dict(
        output_root=str(tmp_path),
        source_root=str(tmp_path),
        mesh_task_contact_world_hash="world",
        core_root=str(tmp_path),
    )
    declaration = dict(prepared_pins={}, input_file_hashes={})
    save("declaration.json", declaration)
    receipts = []
    for i, receipt in enumerate(complete["results"]):
        save(f"receipt-{i:04d}.json", receipt, "receipt_hash")
        receipts.append(receipt)
    complete["results"] = receipts
    save("complete.json", complete)
    reader = save("reader.py", {}, key=None)
    report = dict(reader_source_hash=stage.digest(reader), rows=[])
    readback = save("audit.json", report)
    old_metrics = save("old-metrics.json", dict(rows=[r for p in old for r in p["rows"]]))
    producer["receding_predecessor"] = dict(metrics=str(old_metrics))
    producer_path = save("producer.json", producer, key=None)
    prior_replays = []
    for index, (prior, profile, count) in enumerate(
        zip(old, ("receding-catalog2-ten", "context-baseline-twelve"), (10, 12), strict=True)
    ):
        folder = tmp_path / f"old{index}"
        folder.mkdir()
        names = (
            [j["job"].replace("-", "_") for j in receding_jobs()]
            if index == 0
            else [f"{family}_context_{i}" for i in range(2, 8) for family in ("none", "option4")]
        )
        decl = dict(
            replay_profile=profile,
            new_replay_budget=count,
            new_substep_budget=count * 3000,
            world_hash="world",
            cases=[dict(case_id=n, receipt_hash=str(i)) for i, n in enumerate(names)],
        )
        save(f"old{index}/declaration.json", decl)
        prior.update(
            schema="soccer.rsi.fixed_control_contact_replay_complete.v1",
            declaration_hash=decl["report_hash"],
            attempted_step_calls=count * 3000,
        )
        path = save(f"old{index}/complete.json", prior)
        prior_replays.append(
            dict(path=str(path), file_hash=stage.digest(path), report_hash=prior["report_hash"])
        )
    cfg = dict(
        diagnostic_profile="surface-receding-six",
        source_head="frozen-test-head",
        deadline_unix=10**20,
        new_substep_budget=18000,
        producer_unit="producer",
        audit_unit="audit",
        producer_config=str(producer_path),
        readback=str(readback),
        audit_source=str(reader),
        prior_replays=prior_replays,
    )
    for key in (
        "started_path",
        "summary_output",
        "replay_config",
        "metrics_output",
        "complete_path",
    ):
        cfg[key] = str(tmp_path / f"new-{key}.json")
    cfg["replay_output_root"] = str(tmp_path / "new-replay")
    config_path = save("stage-config.json", cfg, key=None)
    monkeypatch.setattr(stage.sys, "argv", ["stage", "--config", str(config_path)])
    monkeypatch.setattr(
        stage.subprocess,
        "check_output",
        lambda args, **kw: "frozen-test-head" if "rev-parse" in args else "",
    )
    monkeypatch.setattr(stage, "require_terminal", lambda unit: None)
    monkeypatch.setattr(stage, "require_endpoints", lambda *args: ({}, {}))

    def upstream(producer, source, sealed, digest, pins):
        for item in prior_replays:
            pins[item["path"]] = item["file_hash"]

    monkeypatch.setattr(stage, "load_baselines", upstream)
    calls = []

    def fail_child(args, **kwargs):
        calls.append(args)
        raise subprocess.CalledProcessError(9, args)

    monkeypatch.setattr(stage.subprocess, "run", fail_child)
    with pytest.raises(subprocess.CalledProcessError):
        stage.main()
    assert len(calls) == 1 and calls[0][-1] == "--preflight"
    started = Path(cfg["started_path"])
    failure = json.loads(started.with_suffix(".failure.json").read_text())
    assert not failure["automatic_retry_authorized"]
    assert not Path(cfg["complete_path"]).exists()
    with pytest.raises(ValueError, match="once-only"):
        stage.main()
    assert len(calls) == 1
