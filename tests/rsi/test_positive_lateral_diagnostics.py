"""Synthetic pre-replay contracts, never new physics or teacher evidence."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.positive_lateral_diagnostics import measured_contact_readout, require_cohort
from rosclaw_soccer.rsi.positive_lateral_episode_inputs import prospective_split
from rosclaw_soccer.rsi.positive_lateral_receding_scope import build_scope
from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_read_mesh_contact_native_pilot as audit
from scripts import rsi_stage_positive_lateral_contacts as stage
from scripts.rsi_replay_first_touch_contacts import fixed_replay_scope
from tests.rsi.test_mesh_teacher_terminal_readback import expanded_fixture


@pytest.mark.parametrize(
    "tamper", [None, "partial", "order", "duplicate", "budget", "bool", "retag"]
)
def test_fixed_sixteen_dense_replays_are_not_extra_policy_trials(tamper):
    cfg = dict(
        replay_profile="positive-lateral-sixteen",
        new_replay_budget=16,
        new_substep_budget=48000,
        cases=[
            dict(case_id=f"{role}_positive_local_c{i}", receipt_hash=f"{role}-{i}")
            for i in range(8)
            for role in ("control", "teacher")
        ],
    )
    if tamper == "partial":
        cfg["cases"].pop()
    elif tamper == "order":
        cfg["cases"].reverse()
    elif tamper == "duplicate":
        cfg["cases"][1]["receipt_hash"] = cfg["cases"][0]["receipt_hash"]
    elif tamper == "budget":
        cfg["new_substep_budget"] = 96000
    elif tamper == "bool":
        cfg["new_replay_budget"] = True
    elif tamper == "retag":
        cfg["cases"][0]["case_id"] = "surface_receding_g76"
    if tamper:
        with pytest.raises(ValueError):
            fixed_replay_scope(cfg)
    else:
        assert fixed_replay_scope(cfg) == (16, 48000)


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "partial",
        "order",
        "failure",
        "world",
        "teacher",
        "training",
        "fresh",
        "bool",
        "split",
        "cluster",
        "pairing",
    ],
)
def test_cohort_requires_all_sixteen_original_failures_and_scoped_sources(
    tmp_path, monkeypatch, tamper
):
    output = expanded_fixture(tmp_path, monkeypatch, positive_lateral=True)
    audit.main()
    cfg = json.loads((tmp_path / "teacher-config.json").read_text())
    declaration = json.loads((tmp_path / "teacher/declaration.json").read_text())
    complete = json.loads((tmp_path / "teacher/complete.json").read_text())
    report = json.loads(output.read_text())
    if tamper == "partial":
        complete["results"].pop()
    elif tamper == "order":
        report["rows"].reverse()
    elif tamper == "failure":
        report["rows"][0]["outcome"]["task_pass"] = True
    elif tamper == "world":
        report["rows"][0]["world_hash"] = "other"
    elif tamper == "teacher":
        report["teacher_qualified"] = True
    elif tamper == "training":
        complete["neural_training_authorized"] = True
    elif tamper == "fresh":
        report["fresh_exam"] = True
    elif tamper == "bool":
        complete["new_optimizer_updates"] = False
    elif tamper == "split":
        report["prospective_split"]["train_source_seeds"].reverse()
    elif tamper == "cluster":
        report["source_scene_clusters"] = 8
    elif tamper == "pairing":
        report["paired_development"]["rescued_pairs"] = 99
    if tamper:
        with pytest.raises(ValueError):
            require_cohort(cfg, declaration, complete, report)
    else:
        assert require_cohort(cfg, declaration, complete, report) == complete["paired_development"]


def dense_fixture():
    scope = build_scope("sha256:" + "a" * 64)
    declaration = dict(jobs=scope["proposed_jobs"], prospective_split=prospective_split(scope))
    receipts, rows = [], []
    for i, job in enumerate(declaration["jobs"]):
        outcome = dict(task_pass=job["role"] == "teacher", safe=True)
        receipts.append(dict(receipt_hash=f"r{i}", outcome=outcome))
        rows.append(
            dict(
                case_id=job["job"].replace("-", "_"),
                source_receipt_hash=f"r{i}",
                original_outcome=outcome,
                metrics=dict(
                    total_robot_downward_impulse_n_s=0.5,
                    maximum_linear_newton_residual_n=1e-12,
                    first_nonfoot_robot_contact_substep=None,
                ),
            )
        )
    complete = dict(
        results=receipts,
        paired_development=dict(regressed_pairs=0, newly_unsafe_pairs=0, net_success_gain=8),
    )
    replay = dict(
        schema="soccer.rsi.fixed_control_contact_replay_complete.v1",
        rows=rows,
        integrated_substeps=48000,
        attempted_step_calls=48000,
    )
    metrics = dict(schema="soccer.rsi.contact_exit_metrics_readback.v1", rows=copy.deepcopy(rows))
    return declaration, complete, replay, metrics


@pytest.mark.parametrize(
    "tamper", [None, "partial", "duplicate", "outcome", "case", "budget", "nan", "newton"]
)
def test_contact_readout_keeps_whole_original_membership_and_finite_physics(tamper):
    declaration, complete, replay, metrics = dense_fixture()
    if tamper == "partial":
        metrics["rows"].pop()
    elif tamper == "duplicate":
        metrics["rows"][1] = metrics["rows"][0]
    elif tamper == "outcome":
        metrics["rows"][1]["original_outcome"]["task_pass"] = False
    elif tamper == "case":
        metrics["rows"][1]["case_id"] = "wrong"
    elif tamper == "budget":
        replay["attempted_step_calls"] = 48001
    elif tamper == "nan":
        metrics["rows"][0]["metrics"]["total_robot_downward_impulse_n_s"] = float("nan")
    elif tamper == "newton":
        metrics["rows"][0]["metrics"]["maximum_linear_newton_residual_n"] = 1e-3
    if tamper:
        with pytest.raises(ValueError):
            measured_contact_readout(declaration, complete, replay, metrics)
    else:
        result = measured_contact_readout(declaration, complete, replay, metrics)
        assert len(result["rows"]) == 16
        assert len(result["successful_train_source_seeds"]) == 3
        assert len(result["successful_validation_source_seeds"]) == 1
        assert result["research_decision"].startswith("LOCAL_REFERENCE_EVIDENCE")
        assert not result["neural_training_authorized"] and not result["teacher_qualified"]


@pytest.mark.parametrize(
    "condition",
    ["unsafe", "nonfoot", "regression", "no_gain", "no_validation", "one_train_cluster"],
)
def test_gate_does_not_launder_failures_or_conflate_endpoints_and_sources(condition):
    declaration, complete, replay, metrics = dense_fixture()
    if condition == "unsafe":
        complete["results"][1]["outcome"].update(safe=False, task_pass=False)
    elif condition == "nonfoot":
        metrics["rows"][1]["metrics"]["first_nonfoot_robot_contact_substep"] = 1000
    elif condition == "regression":
        complete["paired_development"]["regressed_pairs"] = 1
    elif condition == "no_gain":
        complete["paired_development"]["net_success_gain"] = 0
    else:
        indices = (13, 15) if condition == "no_validation" else (5, 7, 9, 11)
        for index in indices:
            complete["results"][index]["outcome"]["task_pass"] = False
    for index, receipt in enumerate(complete["results"]):
        metrics["rows"][index]["original_outcome"] = copy.deepcopy(receipt["outcome"])
    result = measured_contact_readout(declaration, complete, replay, metrics)
    expected = "REJECT" if condition in ("unsafe", "nonfoot", "regression") else "INSUFFICIENT"
    assert result["research_decision"].startswith(expected)
    assert not result["teacher_qualified"] and not result["dataset_admission_authorized"]


@pytest.mark.parametrize("fail_preflight", [False, True])
def test_stage_success_or_failure_is_persisted_and_cannot_implicitly_retry(
    tmp_path, monkeypatch, fail_preflight
):
    output = expanded_fixture(tmp_path, monkeypatch, positive_lateral=True)
    audit.main()
    cfg = dict(
        diagnostic_profile="positive-lateral-sixteen",
        source_head="test-frozen-head",
        new_substep_budget=48000,
        deadline_unix=10**20,
        producer_unit="producer",
        audit_unit="audit",
        producer_config=str(tmp_path / "teacher-config.json"),
        readback=str(output),
        audit_source=str(Path(audit.__file__).resolve()),
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
    path = tmp_path / "diagnostic-config.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, "argv", ["stage", "--config", str(path)])
    monkeypatch.setattr(
        stage.subprocess,
        "check_output",
        lambda args, **kw: "test-frozen-head" if "rev-parse" in args else "",
    )
    monkeypatch.setattr(stage, "require_terminal", lambda unit: None)
    monkeypatch.setattr(stage, "whole_predecessor", lambda *a: None)
    calls = []

    def child(args, **kwargs):
        calls.append(args)
        assert kwargs["check"] is True
        if fail_preflight:
            raise subprocess.CalledProcessError(8, args)
        if "--preflight" in args:
            return
        complete = json.loads((tmp_path / "teacher/complete.json").read_text())
        replay_cfg = json.loads(Path(cfg["replay_config"]).read_text())
        rows = [
            dict(
                case_id=c["case_id"],
                source_receipt_hash=c["receipt_hash"],
                original_outcome=r["outcome"],
            )
            for c, r in zip(replay_cfg["cases"], complete["results"], strict=True)
        ]
        if Path(args[1]).name == "rsi_replay_first_touch_contacts.py":
            root = Path(cfg["replay_output_root"])
            root.mkdir()
            value = dict(
                schema="soccer.rsi.fixed_control_contact_replay_complete.v1",
                rows=rows,
                integrated_substeps=48000,
                attempted_step_calls=48000,
            )
            target = root / "complete.json"
        else:
            assert Path(args[1]).name == "rsi_read_contact_exit_metrics.py"
            for row in rows:
                row["metrics"] = dict(
                    total_robot_downward_impulse_n_s=0.5,
                    maximum_linear_newton_residual_n=1e-12,
                    first_nonfoot_robot_contact_substep=None,
                )
            value = dict(schema="soccer.rsi.contact_exit_metrics_readback.v1", rows=rows)
            target = Path(cfg["metrics_output"])
        value["report_hash"] = hash_json(value)
        stage.write_once(target, value)

    monkeypatch.setattr(stage.subprocess, "run", child)
    if fail_preflight:
        with pytest.raises(subprocess.CalledProcessError):
            stage.main()
        failure = json.loads(Path(cfg["started_path"]).with_suffix(".failure.json").read_text())
        assert not failure["automatic_retry_authorized"] and len(calls) == 1
        assert not Path(cfg["complete_path"]).exists()
    else:
        stage.main()
        result = json.loads(Path(cfg["complete_path"]).read_text())
        assert len(calls) == 3 and result["new_fixed_control_replay_substeps"] == 48000
        assert result["source_scene_clusters"] == 4 and result["total_metrics_endpoints"] == 16
        assert result["new_neural_inferences"] == result["new_optimizer_updates"] == 0
        assert not result["teacher_qualified"] and not result["dataset_admission_authorized"]
    assert calls[0][-1] == "--preflight"
    with pytest.raises(ValueError, match="once-only"):
        stage.main()
    assert len(calls) == (1 if fail_preflight else 3)
