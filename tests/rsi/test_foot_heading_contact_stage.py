"""Synthetic C3 child orchestration; never new physical or learning evidence."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe
from rosclaw_soccer.rsi.foot_heading_evidence import FIELDS, PREFIX
from rosclaw_soccer.rsi.foot_heading_scope import assess_heading_scope
from rosclaw_soccer.rsi.original_ik_planner_pulse import OriginalIkPlannerPulse
from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_load_heading_cohort as loader
from scripts import rsi_replay_first_touch_contacts as replay_script
from scripts import rsi_stage_pre_planner_contacts as stage
from tests.rsi.test_foot_heading_parity import baseline
from tests.rsi.test_foot_heading_scope import config, receipts
from tests.rsi.test_pre_planner_pulse_cohort import dense


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "native"
    root.mkdir()
    scope, rows = receipts()
    producer = dict(
        config(),
        output_root=str(root),
        core_root=str(tmp_path),
        behavior_model_hash=scope["source_plan"]["behavior_model_hash"],
        mesh_task_contact_world_hash="world",
    )

    def save(path, value, key="report_hash"):
        if key:
            value[key] = hash_json({k: v for k, v in value.items() if k != key})
        Path(path).write_text(json.dumps(value))

    pins = {}
    for i, r in enumerate(rows):
        trace = baseline()
        trace.update({PREFIX + k: np.zeros((300, 1, w)) for k, w in FIELDS.items()})
        if r["heading_enabled"]:
            trace["foot_heading_additional_applied_rad"][30:, 0, 0] = 0.05
        tp = root / f"trace-{i}.npz"
        np.savez_compressed(tp, **trace)
        pins[str(tp)] = stage.digest(tp)
        r["witnesses"] = dict(trace=dict(path=str(tp), sha256=pins[str(tp)]))
        save(root / f"receipt-{i:04d}.json", r, "receipt_hash")
    declaration = dict(scope=scope)
    complete = dict(results=rows, assessment=assess_heading_scope(scope, rows))
    audit = dict(report_hash="audit")
    producer_path = tmp_path / "producer.json"
    save(producer_path, producer, key=None)
    audit_path = tmp_path / "audit.json"
    save(audit_path, audit)
    cohort = dict(producer=producer, declaration=declaration, complete=complete, audit=audit)
    calls = []

    def load(spec, *, sealed, pins):
        calls.append(spec)
        assert spec["producer_config_file_hash"] == stage.digest(producer_path)
        assert spec["source_terminal_readback_hash"] == audit["report_hash"]
        pins.update(
            {r["witnesses"]["trace"]["path"]: r["witnesses"]["trace"]["sha256"] for r in rows}
        )
        return cohort

    monkeypatch.setattr(loader, "load_closed_heading", load)
    monkeypatch.setattr(
        stage.subprocess, "check_output", lambda args, **kw: "frozen" if "rev-parse" in args else ""
    )
    cfg = dict(
        mode="SIM_ONLY",
        diagnostic_profile="foot-heading-contact-six",
        source_head="frozen",
        hardware_authorized=False,
        promotion_authorized=False,
        neural_training_authorized=False,
        new_substep_budget=18000,
        deadline_unix=10**20,
        producer_unit="native",
        audit_unit="audit",
        producer_config=str(producer_path),
        producer_config_file_hash=stage.digest(producer_path),
        readback=str(audit_path),
        audit_source="original-owned-reader.py",
        started_path=str(tmp_path / "started.json"),
        replay_config=str(tmp_path / "replay.json"),
        replay_output_root=str(tmp_path / "dense"),
        metrics_output=str(tmp_path / "metrics.json"),
        complete_path=str(tmp_path / "complete.json"),
    )
    path = tmp_path / "config.json"
    save(path, cfg, key=None)
    monkeypatch.setattr(sys, "argv", ["stage", "--config", str(path)])
    return cfg, cohort, save, calls


@pytest.mark.parametrize("fail", [False, True])
def test_heading_stage_dispatches_exact_six_and_never_retries_failed_child(
    tmp_path, monkeypatch, fail
):
    cfg, c, save, loads = fixture(tmp_path, monkeypatch)
    calls = []

    def child(args, **kwargs):
        calls.append(args)
        assert kwargs["check"] is True and kwargs["timeout"] == 2400
        if fail:
            raise subprocess.CalledProcessError(7, args)
        replay, metrics = dense(c["declaration"], c["complete"])
        if len(calls) == 2:
            root = Path(cfg["replay_output_root"])
            root.mkdir()
            save(root / "complete.json", replay)
        elif len(calls) == 3:
            save(cfg["metrics_output"], metrics)

    monkeypatch.setattr(stage.subprocess, "run", child)
    before = copy.deepcopy(c)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            stage.main()
        assert len(calls) == 1
        assert Path(cfg["started_path"]).with_suffix(".failure.json").exists()
        assert not Path(cfg["complete_path"]).exists()
    else:
        stage.main()
        result = json.loads(Path(cfg["complete_path"]).read_text())
        built = json.loads(Path(cfg["replay_config"]).read_text())
        assert replay_script.fixed_replay_scope(built) == (6, 18000)
        assert len(calls) == 3 and len(loads) == 1
        assert [j["case_id"] for j in built["cases"]] == [
            j["job"].replace("-", "_") for j in c["declaration"]["scope"]["jobs"]
        ]
        assert result["schema"] == "soccer.rsi.foot_heading_contact_complete.v1"
        assert len(result["paired_motion_diagnostics"]) == 3
        assert result["new_fixed_control_replay_substeps"] == 18000
        assert result["new_neural_inferences"] == result["new_optimizer_updates"] == 0
        assert not result["teacher_qualified"] and not result["promotion_authorized"]
    assert c == before
    with pytest.raises(ValueError, match="unused"):
        stage.main()
    assert len(calls) == (1 if fail else 3)


@pytest.mark.parametrize("tamper", [None, "world", "outside", "qualification"])
def test_exact_heading_replay_preflight_passes_both_explicit_intents_per_case(
    tmp_path, monkeypatch, capsys, tamper
):
    import mujoco

    stage_cfg, c, save, loads = fixture(tmp_path, monkeypatch)
    world = tmp_path / "world.xml"
    world.write_text("synthetic witness: preflight must never compile this as MuJoCo")
    world_hash = stage.digest(world)
    c["producer"]["mesh_task_contact_world_hash"] = world_hash
    producer_path = Path(stage_cfg["producer_config"])
    save(producer_path, c["producer"], key=None)
    cases = []
    for i, row in enumerate(c["complete"]["results"]):
        row["qualification_alignment"] = dict(synthetic_case=i)
        row["witnesses"]["world"] = dict(path=str(world), sha256=world_hash)
        for kind in ("independent_review", "foundation_review"):
            path = tmp_path / f"{kind}-{i}.json"
            save(path, dict(schema="test.synthetic_metadata_not_physics", case=i))
            row["witnesses"][kind] = dict(path=str(path), sha256=stage.digest(path))
        rp = Path(c["producer"]["output_root"]) / f"receipt-{i:04d}.json"
        save(rp, row, "receipt_hash")
        cases.append(
            dict(
                case_id=row["job"].replace("-", "_"),
                receipt_path=str(rp),
                receipt_file_hash=stage.digest(rp),
                receipt_hash=row["receipt_hash"],
            )
        )
    cfg = dict(
        replay_profile="foot-heading-contact-six",
        source_head="frozen",
        behavior_model_hash=c["producer"]["behavior_model_hash"],
        world_hash=world_hash,
        output_root=str(tmp_path / "unused-preflight-root"),
        new_replay_budget=6,
        new_substep_budget=18000,
        mujoco_version=mujoco.__version__,
        producer_config=str(producer_path),
        producer_config_file_hash=stage.digest(producer_path),
        producer_unit="native",
        audit_unit="audit",
        source_terminal_readback=stage_cfg["readback"],
        source_terminal_readback_hash=c["audit"]["report_hash"],
        cases=cases,
        hardware_authorized=False,
        promotion_authorized=False,
        neural_training_authorized=False,
    )
    if tamper == "world":
        cfg["world_hash"] = "wrong"
    elif tamper == "outside":
        outside = tmp_path / "borrowed-receipt.json"
        outside.write_bytes(Path(cases[0]["receipt_path"]).read_bytes())
        cases[0]["receipt_path"] = str(outside)
    config_path = tmp_path / "replay-preflight.json"
    save(config_path, cfg, key=None)
    monkeypatch.setattr(sys, "argv", ["replay", "--config", str(config_path), "--preflight"])
    # Only orchestration dispatch is under test. Physical labels, whole source
    # closure and numerical qualification have their own non-mocked tests.
    outcomes = iter(c["complete"]["results"])

    def label(_):
        row = next(outcomes)
        return SimpleNamespace(to_dict=lambda: row["outcome"])

    monkeypatch.setattr(replay_script, "label_first_touch", label)
    checked = []

    def qualification(review, foundation, outcome, **kwargs):
        i = len(checked)
        assert review["case"] == foundation["case"] == i
        job = c["declaration"]["scope"]["jobs"][i]
        assert kwargs == dict(
            original_ik_planner_pulse_contract_hash=OriginalIkPlannerPulse(0.08).contract()[
                "contract_hash"
            ],
            foot_heading_contract_hash=FootHeadingContactProbe(job["heading_enabled"]).contract()[
                "contract_hash"
            ],
        )
        checked.append(kwargs)
        return (
            {}
            if tamper == "qualification"
            else c["complete"]["results"][i]["qualification_alignment"]
        )

    monkeypatch.setattr(replay_script, "require_first_touch_qualification", qualification)
    if tamper is None:
        replay_script.main()
        assert len(checked) == 6 and len(loads) == 1
        assert (
            "FIXED_REPLAYS 6 18000 SUBSTEPS_PREFLIGHT_ZERO_MODEL_ALLOCATIONS"
            in capsys.readouterr().out
        )
        assert len({r["foot_heading_contract_hash"] for r in checked}) == 2
        assert len({r["original_ik_planner_pulse_contract_hash"] for r in checked}) == 1
    else:
        with pytest.raises(ValueError):
            replay_script.main()
    assert not Path(cfg["output_root"]).exists()
