"""Workflow fixtures only; these do not claim physical execution."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import rsi_continue_retained_parent_curriculum as workflow


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = Path(workflow.__file__).resolve().parent.parent
    plan = source / "docs/rsi/protocols/current-retained-parent-smooth-curriculum-v367.json"
    args = argparse.Namespace(plan=plan, output_root=tmp_path / "output")
    for name in (
        "model",
        "parent_model",
        "parent_pilot",
        "transport_review",
        "g1_usd",
        "scene",
        "bank",
        "isaac_python",
        "model_root",
        "late_swing_policy",
        "core_root",
    ):
        path = tmp_path / name
        path.write_text("{}")
        setattr(args, name, path)
    args.parent_bank_root = tmp_path / "parent-bank"
    args.parent_bank_root.mkdir()
    for name in ("validation_summary.json", "independent_review.json"):
        (args.parent_bank_root / name).write_text("{}")
    args.parent_cpu_root = []
    for i in range(4):
        folder = tmp_path / f"cpu-{i}"
        folder.mkdir()
        (folder / "independent_review.json").write_text("{}")
        args.parent_cpu_root.append(folder)
    parent_hash = json.loads(plan.read_text())["retained_parent_model_hash"]
    args.parent_model.write_text(json.dumps(dict(model_hash=parent_hash)))
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _: args)
    monkeypatch.setattr(workflow.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(workflow, "_head", lambda _: "frozen-fixture")
    monkeypatch.setattr(workflow, "validate_model", lambda _: None)
    monkeypatch.setattr(workflow, "qualify_parent_bank", lambda *a: [{}, {}])
    monkeypatch.setattr(workflow, "required_storage", lambda *a: 1024)
    monkeypatch.setattr(
        workflow.shutil, "disk_usage", lambda _: SimpleNamespace(free=200 * 1024**3)
    )
    monkeypatch.setattr(
        workflow,
        "_sealed",
        lambda p: dict(
            report_hash="fixture-sealed", rows=[], consumed_gain=1, current_parent_retained=True
        ),
    )
    stages = []
    monkeypatch.setattr(workflow, "run_stage", lambda root, name, cmd: stages.append((name, cmd)))
    return args, stages


def test_qualified_workflow_has_ordered_stages_and_current_parent_baseline(setup):
    args, stages = setup
    workflow.main()
    assert [n for n, _ in stages] == [
        "zero-pilot",
        "zero-pilot-review",
        "curriculum-preflight",
        "all-failure-exploration",
        "audit-and-learn",
        "candidate-validation",
    ]
    collection = stages[3][1]
    assert collection[collection.index("--exploration-stream") + 1] == "1"
    assert "--compressed-sampling-models" in collection
    assert "--first-four-courses" not in collection
    validation = stages[-1][1]
    assert validation[validation.index("--baseline") + 1] == str(args.parent_model)
    assert validation.count("--baseline-cpu-root") == 4
    commitment = json.loads((args.output_root / "commitment.json").read_text())
    assert commitment["planned_physical_executions"] == 204  # Fixture's two failures.
    assert commitment["fresh_holdout_open_authorized"] is False
    assert commitment["system_disk_reserve_bytes"] == 100 * 1024**3


def test_parent_rejection_blocks_every_stage_and_output(setup, monkeypatch):
    args, stages = setup

    def reject(*a):
        raise ValueError("parent out of play")

    monkeypatch.setattr(workflow, "qualify_parent_bank", reject)
    with pytest.raises(ValueError, match="out of play"):
        workflow.main()
    assert not stages and not args.output_root.exists()


def test_whole_budget_is_required_before_zero_pilot(setup, monkeypatch):
    args, stages = setup
    monkeypatch.setattr(
        workflow.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3)
    )
    with pytest.raises(ValueError, match="entire"):
        workflow.main()
    assert not stages and not args.output_root.exists()


def test_failed_collection_never_learns_or_validates(setup, monkeypatch):
    args, stages = setup

    def stage(root, name, command):
        stages.append((name, command))
        if name == "all-failure-exploration":
            raise RuntimeError("native fixture failure")

    monkeypatch.setattr(workflow, "run_stage", stage)
    with pytest.raises(RuntimeError):
        workflow.main()
    assert stages[-1][0] == "all-failure-exploration"
    assert not (args.output_root / "result.json").exists()


def test_source_input_drift_prevents_result_publication(setup, monkeypatch):
    args, stages = setup

    def stage(root, name, command):
        stages.append((name, command))
        if name == "candidate-validation":
            args.parent_model.write_text("changed fixture input")

    monkeypatch.setattr(workflow, "run_stage", stage)
    with pytest.raises(ValueError, match="drift"):
        workflow.main()
    assert not (args.output_root / "result.json").exists()


def test_budget_includes_all_samples_and_two_pilots(tmp_path, monkeypatch):
    monkeypatch.setattr(
        workflow, "folder_bytes", lambda path: 1000 if "candidate" in str(path) else 500
    )
    expected = int(1.15 * (14 * (9500 + 8 * 100 * 1024**2) + 60 * 2500)) + 2 * 1024**3
    assert workflow.required_storage(tmp_path, [dict(seed=1, lane=0)], 14) == expected
