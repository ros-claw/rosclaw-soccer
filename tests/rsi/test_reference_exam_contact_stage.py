"""Synthetic original cohort/physics qualification and child execution only.

Tests immutable source membership, finite staging and no implicit retries.
No real MuJoCo bodies, planner or learned-performance claims here.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_replay_first_touch_contacts as replay_script
from scripts import rsi_stage_reference_exam_contacts as stage
from tests.rsi.test_reference_student_exam_diagnostics import cohort, dense


def fixture(tmp_path, zero=False):
    if zero:
        from tests.rsi.test_reference_positive_zero import zero_cohort

        cfg, d, c, a = zero_cohort()
    else:
        cfg, d, c, a = cohort()
    root = tmp_path / "producer"
    root.mkdir()
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    reader = source / "scripts/rsi_read_reference_student_exam.py"
    reader.write_text("synthetic original reader, no physical qualification")
    cfg.update(
        output_root=str(root),
        source_root=str(source),
        core_root=str(tmp_path),
        source_head="mock-original-head",
    )
    d.update(cfg)
    d.update(prepared_pins={}, input_file_hashes={})

    def save(path, value, key="report_hash"):
        if key:
            value[key] = hash_json({k: v for k, v in value.items() if k != key})
        Path(path).write_text(json.dumps(value))
        return value

    save(root / "declaration.json", d)
    cases = []
    for i, r in enumerate(c["results"]):
        path = root / f"receipt-{i:04d}.json"
        save(path, r, "receipt_hash")
        a["rows"][i]["receipt_hash"] = r["receipt_hash"]
        cases.append(
            dict(
                case_id=r["job"].replace("-", "_"),
                receipt_path=str(path),
                receipt_file_hash=stage.digest(path),
                receipt_hash=r["receipt_hash"],
            )
        )
    c["declaration_hash"] = d["report_hash"]
    save(root / "complete.json", c)
    a.update(
        complete_hash=c["report_hash"],
        input_file_hashes={},
        reader_source_hash=stage.digest(reader),
    )
    audit_path = tmp_path / "readback.json"
    save(audit_path, a)
    config_path = tmp_path / "producer-config.json"
    save(config_path, cfg, key=None)
    replay = dict(
        replay_profile="reference-positive-zero-nine" if zero else "reference-student-thirteen",
        source_head="post-head",
        output_root=str(tmp_path / "dense"),
        new_replay_budget=9 if zero else 13,
        new_substep_budget=27000 if zero else 39000,
        behavior_model_hash=cfg["behavior_model_hash"],
        world_hash=cfg["mesh_task_contact_world_hash"],
        cases=cases,
        producer_config=str(config_path),
        producer_config_file_hash=stage.digest(config_path),
        producer_complete=str(root / "complete.json"),
        producer_declaration=str(root / "declaration.json"),
        source_terminal_readback=str(audit_path),
        source_terminal_readback_hash=a["report_hash"],
        audit_source=str(reader),
    )

    def sealed(path, key="report_hash"):
        value = json.loads(Path(path).read_text())
        if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("test seal changed")
        return value

    return cfg, d, c, a, replay, save, sealed


@pytest.mark.parametrize(
    "tamper", [None, "config", "reader", "outside", "order", "unverified", "retag"]
)
@pytest.mark.parametrize("zero", [False, True])
def test_reference_model_intent_requires_original_complete_audited_sources(tmp_path, tamper, zero):
    _, d, c, a, cfg, save, sealed = fixture(tmp_path, zero)
    if tamper == "config":
        Path(cfg["producer_config"]).write_text("{}")
    elif tamper == "reader":
        Path(cfg["audit_source"]).write_text("changed reader")
    elif tamper == "outside":
        cfg["cases"][0]["receipt_path"] = str(tmp_path / "borrowed.json")
    elif tamper == "order":
        cfg["cases"].reverse()
    elif tamper == "unverified":
        a["whole_input_and_witness_files_verified"] = False
        save(cfg["source_terminal_readback"], a)
        cfg["source_terminal_readback_hash"] = a["report_hash"]
    elif tamper == "retag":
        c["results"][1]["reference_student_model_hash"] = "wrong-model"
        save(cfg["producer_complete"], c)
        a["complete_hash"] = c["report_hash"]
        save(cfg["source_terminal_readback"], a)
        cfg["source_terminal_readback_hash"] = a["report_hash"]
    pins = {}
    if tamper:
        with pytest.raises(ValueError):
            replay_script.reference_exam_sources(cfg, sealed, pins)
    else:
        result = replay_script.reference_exam_sources(cfg, sealed, pins)
        assert len(result) == (9 if zero else 13) and result == c["results"]
        assert str(Path(cfg["audit_source"])) in pins


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("zero", [False, True])
def test_stage_finishes_once_or_preserves_failed_child_without_retry(
    tmp_path, monkeypatch, fail, zero
):
    producer, d, c, a, r, save, _ = fixture(tmp_path, zero)
    cfg = dict(
        diagnostic_profile=r["replay_profile"],
        mode="SIM_ONLY",
        hardware_authorized=False,
        promotion_authorized=False,
        neural_training_authorized=False,
        source_head="post-head",
        deadline_unix=10**20,
        new_substep_budget=r["new_substep_budget"],
        producer_unit="native",
        audit_unit="audit",
        producer_config=r["producer_config"],
        producer_config_file_hash=r["producer_config_file_hash"],
        readback=r["source_terminal_readback"],
        audit_source=r["audit_source"],
        replay_output_root=r["output_root"],
    )
    for k in ("started_path", "summary_output", "replay_config", "metrics_output", "complete_path"):
        cfg[k] = str(tmp_path / f"new-{k}.json")
    cfgpath = tmp_path / "stage.json"
    save(cfgpath, cfg, key=None)
    monkeypatch.setattr(sys, "argv", ["stage", "--config", str(cfgpath)])
    monkeypatch.setattr(
        stage.subprocess,
        "check_output",
        lambda args, **kw: "post-head" if "rev-parse" in args else "",
    )
    checked = []
    monkeypatch.setattr(stage, "require_terminal", checked.append)
    calls = []

    def child(args, **kw):
        assert kw["check"] is True and kw["timeout"] == 2400
        calls.append(args)
        if fail:
            raise subprocess.CalledProcessError(7, args)
        if len(calls) == 2:
            root = Path(cfg["replay_output_root"])
            root.mkdir()
            if zero:
                from tests.rsi.test_reference_positive_zero import dense as zero_dense

                _, _, replayed, metric = zero_dense()
            else:
                _, _, replayed, metric = dense()
            for rows in (replayed["rows"], metric["rows"]):
                for i, row in enumerate(rows):
                    row["source_receipt_hash"] = c["results"][i]["receipt_hash"]
            save(root / "complete.json", replayed)
            save(cfg["metrics_output"], metric)

    monkeypatch.setattr(stage.subprocess, "run", child)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            stage.main()
        assert len(calls) == 1 and Path(cfg["started_path"]).with_suffix(".failure.json").exists()
        assert not Path(cfg["complete_path"]).exists()
    else:
        stage.main()
        assert len(calls) == 3 and checked == ["native", "audit"]
        out = json.loads(Path(cfg["complete_path"]).read_text())
        assert out["new_fixed_control_replay_substeps"] == r["new_substep_budget"]
        assert out["new_closed_loop_policy_trials"] == 0 and not out["promotion_authorized"]
        built = json.loads(Path(cfg["replay_config"]).read_text())
        assert built["producer_config_file_hash"] == r["producer_config_file_hash"]
        assert replay_script.fixed_replay_scope(built) == (
            r["new_replay_budget"],
            r["new_substep_budget"],
        )
    with pytest.raises(ValueError, match="unused once-only"):
        stage.main()


@pytest.mark.parametrize("zero", [False, True])
def test_exact_reference_driver_preflight_uses_explicit_qualified_model_intent(
    tmp_path, monkeypatch, capsys, zero
):
    # The original producer/reader numerical engines are all mocked. This
    # tests the real replay preflight/qualification dispatcher, not dynamics.
    import mujoco

    from rosclaw_soccer.rsi.reference_student_exam_scope import zero_model_like
    from scripts import rsi_read_reference_student_exam as reader
    from scripts import rsi_reference_student_exam as native
    from scripts.rsi_fit_pre_parent_reference import FIT_SEED
    from tests.rsi.test_reference_student_exam_pipeline import fixture as native_fixture

    c, path, _, _ = native_fixture(tmp_path, monkeypatch, "normal")
    c.cfg["behavior_model_hash"] = c.assets.MEAN
    if zero:
        from rosclaw_soccer.rsi.reference_exam_profiles import ZERO, positive_zero_scope

        c.cfg.update(
            exam_profile=ZERO,
            max_attempts=9,
            new_physical_budget=9,
            new_independent_cpu_budget=9,
            combined_body_substep_budget=54000,
        )
        c.scope = positive_zero_scope(c.assets.MEAN)
        # The synthetic engine fixture has no raw Foundation captures.
        # Actual all-300 numerical zero checks have a separate non-mocked test.
        monkeypatch.setattr(reader, "require_zero_navigation_trace", lambda _: None)
    path.write_text(json.dumps(c.cfg))
    c.models["zero"] = zero_model_like(c.models["student"], seed=FIT_SEED)
    monkeypatch.setattr(sys, "argv", ["native", "--config", str(path)])
    native.main()
    audit_path = tmp_path / "full-audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["reader", "--config", str(path), "--unit", "fake", "--output", str(audit_path)],
    )
    reader.main()
    original_audit = json.loads(audit_path.read_text())
    original = json.loads((c.root / "complete.json").read_text())
    cfg = dict(
        replay_profile="reference-positive-zero-nine" if zero else "reference-student-thirteen",
        behavior_model_hash=c.assets.MEAN,
        source_head="test-head",
        output_root=str(tmp_path / "unused-dense"),
        new_replay_budget=9 if zero else 13,
        new_substep_budget=27000 if zero else 39000,
        mujoco_version=mujoco.__version__,
        world_hash=c.cfg["mesh_task_contact_world_hash"],
        producer_config=str(path),
        producer_config_file_hash=stage.digest(path),
        producer_complete=str(c.root / "complete.json"),
        producer_declaration=str(c.root / "declaration.json"),
        source_terminal_readback=str(audit_path),
        source_terminal_readback_hash=original_audit["report_hash"],
        audit_source=str(Path(reader.__file__).resolve()),
        hardware_authorized=False,
        promotion_authorized=False,
        neural_training_authorized=False,
        cases=[
            dict(
                case_id=j["job"].replace("-", "_"),
                receipt_path=str(c.root / f"receipt-{i:04d}.json"),
                receipt_file_hash=stage.digest(c.root / f"receipt-{i:04d}.json"),
                receipt_hash=r["receipt_hash"],
            )
            for i, (j, r) in enumerate(zip(c.scope["jobs"], original["results"], strict=True))
        ],
    )
    rp = tmp_path / "replay-config.json"
    rp.write_text(json.dumps(cfg))
    monkeypatch.setattr(
        replay_script.subprocess,
        "check_output",
        lambda args, **kw: "test-head" if "rev-parse" in args else "",
    )
    monkeypatch.setattr(sys, "argv", ["replay", "--config", str(rp), "--preflight"])
    replay_script.main()
    assert (
        f"FIXED_REPLAYS {9 if zero else 13} {27000 if zero else 39000} "
        "SUBSTEPS_PREFLIGHT_ZERO_MODEL_ALLOCATIONS" in capsys.readouterr().out
    )
    assert not Path(cfg["output_root"]).exists()
