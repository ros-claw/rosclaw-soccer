"""Declared successor argument tests only; no training or physics is run here."""

import json

import pytest

from rosclaw_soccer.rsi.consolidated_adapter_learning import FEATURE_STATUS
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_run_consolidated_adapter_native_exam import native_command


def arguments(tmp_path, exam="pass80"):
    feature = tmp_path / "features"
    feature.mkdir()
    fp = feature / "complete.json"
    fp.write_text(
        json.dumps(
            dict(
                status=FEATURE_STATUS,
                records=[dict(rows=2)] * 274,
                own_feature_decisions=548,
                data_hash="sha256:bank",
            )
        )
    )
    tr = tmp_path / "training"
    tr.mkdir()
    tp = tr / "complete.json"
    tp.write_text(
        json.dumps(
            dict(
                checkpoint_kind="consolidated-adapter-v1",
                checkpoint_hash="sha256:checkpoint",
                data_hash="sha256:bank",
                feature_hash=digest(fp),
                input_file_hashes={str(fp): digest(fp)},
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            )
        )
    )
    cpu = tmp_path / "cpu"
    cpu.mkdir()
    (cpu / "complete.json").write_text(
        json.dumps(
            dict(
                status="DUAL_RECURRENT_TRAINED_CHECKPOINT_CPU_OWN_HISTORY_AUDIT_PASS_NOT_PHYSICS",
                checkpoint_kind="consolidated-adapter-v1",
                source_hash=digest(tp),
                source_head="head",
                checkpoint_hash="sha256:checkpoint",
                data_hash="sha256:bank",
                own_actor_decisions=548,
                physical_steps_executed=0,
                old_retention_proven=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            )
        )
    )
    result = dict(
        training_root=tr,
        training_hash=digest(tp),
        cpu_root=cpu,
        expected_head="head",
        output=tmp_path / "unused physical output",
        exam=exam,
    )
    if exam == "retention44":
        rr = tmp_path / "rehearsal"
        rr.mkdir()
        (rr / "complete.json").write_text("{}")
        result.update(rehearsal_root=rr, rehearsal_hash=digest(rr / "complete.json"))
    return result


@pytest.mark.parametrize("exam", ["pass80", "retention44"])
def test_declared_successor_only_invokes_existing_frozen_protocol(tmp_path, exam):
    args = arguments(tmp_path, exam)
    cmd = native_command(**args)
    assert cmd[2:4] == [
        "-m",
        "scripts.rsi_exam_team_pass_motor"
        if exam == "pass80"
        else "scripts.rsi_exam_trained_dual_motor",
    ]
    assert "consolidated-adapter-v1" in cmd
    assert cmd[cmd.index("--output") + 1] == str(args["output"])
    assert not any(k in cmd for k in ("--horizon", "--threshold", "--teacher", "--promote"))


@pytest.mark.parametrize("fault", ["count", "head", "physical", "permission", "failed", "exists"])
def test_missing_or_invalid_predecessor_cannot_start_physics(tmp_path, fault):
    args = arguments(tmp_path)
    p = args["cpu_root"] / "complete.json"
    c = json.loads(p.read_text())
    if fault == "count":
        c["own_actor_decisions"] = 2
    elif fault == "head":
        c["source_head"] = "other"
    elif fault == "physical":
        c["physical_steps_executed"] = 1
    elif fault == "permission":
        c["hardware_authorized"] = True
    elif fault == "failed":
        (args["cpu_root"] / "failed.json").write_text("{}")
    elif fault == "exists":
        args["output"].mkdir()
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        native_command(**args)


def audit_arguments(tmp_path, exam="pass80-own"):
    args = arguments(tmp_path)
    root = tmp_path / "native"
    root.mkdir()
    decl = dict(
        source_head="head",
        training_hash=args["training_hash"],
        checkpoint_kind="consolidated-adapter-v1",
        hardware_authorized=False,
        promotion_authorized=False,
    )
    dp = root / "declaration.json"
    dp.write_text(json.dumps(decl))
    status, n = (
        ("TEAM_PASS_CANDIDATE80_COLLECTED_PENDING_OWN_NATIVE_AND_RETENTION", 80)
        if exam == "pass80-own"
        else ("DUAL_RECURRENT_NATIVE_TRAIN_EXAM_CLOSED_PENDING_INDEPENDENT_AUDIT", 44)
    )
    (root / "complete.json").write_text(
        json.dumps(dict(decl, status=status, records=[{}] * n, declaration_hash=digest(dp)))
    )
    args.update(
        exam=exam, native_root=root, producer_head="head", expected_head="independent-reader"
    )
    return args


@pytest.mark.parametrize("exam", ["pass80-own", "retention44-own"])
def test_whole_independent_readback_can_follow_another_pinned_producer(tmp_path, exam):
    args = audit_arguments(tmp_path, exam)
    cmd = native_command(**args)
    assert cmd[3] == (
        "scripts.rsi_audit_team_pass_motor"
        if exam == "pass80-own"
        else "scripts.rsi_audit_dual_motor_native_exam"
    )
    assert cmd[cmd.index("--source-hash") + 1] == digest(args["native_root"] / "complete.json")
    assert cmd[cmd.index("--expected-head") + 1] == "independent-reader"


@pytest.mark.parametrize(
    "fault", ["count", "model", "producer", "declaration", "failed", "missing"]
)
def test_incomplete_or_wrong_native_source_cannot_trigger_independent_reexecution(tmp_path, fault):
    args = audit_arguments(tmp_path)
    p = args["native_root"] / "complete.json"
    r = json.loads(p.read_text())
    if fault == "count":
        r["records"].pop()
    elif fault == "model":
        r["training_hash"] = "other"
    elif fault == "producer":
        r["source_head"] = "other"
    elif fault == "declaration":
        r["declaration_hash"] = "other"
    elif fault == "failed":
        (args["native_root"] / "failed.json").write_text("{}")
    elif fault == "missing":
        args["producer_head"] = None
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        native_command(**args)
