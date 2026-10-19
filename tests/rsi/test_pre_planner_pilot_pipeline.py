"""Six-body orchestration with physical/NN/receipt verification engines mocked."""

import copy
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.first_touch_campaign import FirstTouchCampaign
from rosclaw_soccer.rsi.planner_navigation_pulse_scope import (
    GROUPS,
    POSITIVE_PROFILE,
    PROFILE,
    pulse_scope,
)
from scripts import rsi_pre_planner_navigation_pilot as pilot
from scripts.rsi_read_pre_planner_navigation import read_journal
from tests.rsi.test_planner_navigation_pulse_evidence import zero_model
from tests.rsi.test_planner_navigation_pulse_scope import config


@pytest.mark.parametrize("tamper", [None, "changed_course", "changed_stream"])
@pytest.mark.parametrize("profile", [PROFILE, POSITIVE_PROFILE])
def test_real_prepare_compares_json_plan_identity_not_python_tuple_container(
    tmp_path, monkeypatch, tamper, profile
):
    runtime_plan = dict(seeds=(20261077, 20261078), jobs=[dict(course=(2.65, 0.1, 0.5))], stream=19)
    saved_plan = json.loads(json.dumps(runtime_plan))
    if tamper == "changed_course":
        saved_plan["jobs"][0]["course"][0] += 0.01
    elif tamper == "changed_stream":
        saved_plan["stream"] = 18
    candidate = dict(
        parameters=BoundedFirstTouchControl().contract()["parameters"],
        swing="bounded_contact_diagnostic",
    )
    model = zero_model()
    teachers = {12: dict(receipt_hash="original-teacher")}
    c = SimpleNamespace(
        source=Path(pilot.__file__).resolve().parents[1],
        root=tmp_path / "unused",
        cfg=dict(matched_zero_cohort={}, mesh_task_contact_world_hash=model["world_hash"]),
        models=dict(zero=model),
        scope=pulse_scope("sha256:" + "a" * 64, profile=profile),
        candidate=candidate,
        teachers=teachers,
        pins={},
        sealed=None,
        positive_plan=SimpleNamespace(contract=lambda: runtime_plan),
    )
    recorded = dict(
        config=dict(exam_profile="reference-positive-zero-nine-v1"),
        declaration=dict(
            candidate=candidate,
            models=dict(zero=model),
            reused_teachers={"12": teachers[12]},
            scope=dict(source_plans=dict(positive19=saved_plan)),
            mesh_task_contact_world_hash=model["world_hash"],
        ),
        complete=dict(
            report_hash="zero-complete", results=[dict(group=g, role="zero") for g in GROUPS]
        ),
    )
    monkeypatch.setattr(pilot, "load_inputs", lambda *a, **kw: c)
    monkeypatch.setattr(pilot, "load_cohort", lambda *a, **kw: recorded)
    if profile == POSITIVE_PROFILE:
        c.cfg["negative_predecessor"] = {"whole_closed_negative": "test"}
        from scripts import rsi_load_pre_planner_cohort as loader

        def predecessor(spec, **kwargs):
            assert spec == c.cfg["negative_predecessor"]
            assert kwargs["pins"] is c.pins
            return {"whole_closed_negative": "authenticated"}

        monkeypatch.setattr(loader, "require_negative_predecessor", predecessor)
    if tamper is None:
        assert pilot.prepare(tmp_path / "not-read-config") is c
        assert list(c.recorded_zeros) == list(GROUPS)
        if profile == POSITIVE_PROFILE:
            assert c.negative_predecessor == {"whole_closed_negative": "authenticated"}
    else:
        with pytest.raises(ValueError, match="source plan"):
            pilot.prepare(tmp_path / "not-read-config")
    assert not c.root.exists()


@pytest.mark.parametrize(
    "mode", ["complete", "body_failure", "off_corrupt", "early_probe_corrupt", "preflight"]
)
@pytest.mark.parametrize("profile", [PROFILE, POSITIVE_PROFILE])
def test_all_six_and_once_only_failure_before_following_bodies(
    tmp_path, monkeypatch, mode, profile
):
    baseline = {f"field_{i}": np.zeros((300, 1, 1)) for i in range(33)}
    source = tmp_path / "source-traces"
    source.mkdir()
    witnesses, pins = {}, {}
    for group in GROUPS:
        path = source / f"g{group}.npz"
        np.savez(path, **baseline)
        h = pilot.digest(path)
        pins[str(path)] = h
        witnesses[group] = dict(
            receipt_hash=f"original-{group}", witnesses=dict(trace=dict(path=str(path), sha256=h))
        )
    cfg = dict(config(), deadline_unix=10**30)
    cfg["exam_profile"] = profile
    if profile == POSITIVE_PROFILE:
        cfg["negative_predecessor"] = {"whole_closed_evidence": "mocked_preparation"}
    c = SimpleNamespace(
        cfg=cfg,
        root=tmp_path / "study",
        source=tmp_path,
        core=tmp_path,
        scope=pulse_scope("sha256:" + "a" * 64, profile=profile),
        models=dict(zero=dict(model_hash="mock-zero")),
        candidate=dict(parameters={}),
        recorded_zeros=witnesses,
        recorded_zero_complete_hash="old-whole",
        fit_stage=dict(report_hash="mock-fit"),
        pins=pins,
        nominal=dict(results=[{}]),
        assets=None,
        review=None,
        positive_plan=None,
        positive_manifest=None,
        contexts={g: None for g in GROUPS},
        p=SimpleNamespace(PINS={}, HEADS={}, stable=lambda: None),
    )
    c.cfg["mesh_task_contact_world_hash"] = "sha256:" + "b" * 64
    allocations, calls = [], []
    monkeypatch.setattr(pilot, "prepare", lambda _: c)
    monkeypatch.setattr(pilot, "admit_resources", lambda _: None)
    monkeypatch.setattr(FirstTouchCampaign, "verify_receipt", lambda *a, **kw: None)

    def allocate(*args, **kw):
        allocations.append(True)
        return SimpleNamespace(
            factory=None, compiler=None, auditor=None, reader=None, encoder_session=None
        )

    def body(folder, group, candidate, **kw):
        pulse = kw["planner_navigation_pulse"]
        assert kw["reference_student_model"] is c.models["zero"]
        calls.append((group, pulse.lateral_delta_m_s))
        if mode == "body_failure" and len(calls) == 4:
            raise ValueError("mock body fault")
        folder.mkdir()
        trace = copy.deepcopy(baseline)
        trace.update(
            {
                "planner_pulse_" + name: np.zeros((300, 1, 3))
                for name in ("baseline_command", "requested_residual", "applied_command")
            }
        )
        if pulse.lateral_delta_m_s:
            trace["field_0"][100:, 0, 0] = 0.01
            if mode == "early_probe_corrupt":
                trace["field_0"][99, 0, 0] = 0.01
        elif mode == "off_corrupt":
            trace["field_0"][299, 0, 0] = 0.01
        np.savez(folder / "physical_trace.npz", **trace)
        return dict(
            physical_substeps=3000,
            foundation_calls=300,
            sampling_draws=270,
            outcome=dict(task_pass=False, safe=len(calls) != 4),
            hardware_authorized=False,
            promotion_authorized=False,
        )

    monkeypatch.setattr(pilot, "allocate_native_first_touch_bundle", allocate)
    monkeypatch.setattr(pilot, "qualified_episode", body)
    monkeypatch.setattr(
        pilot, "_sealed", lambda *a, **kw: dict(default_locomotion_planner_events=[])
    )
    monkeypatch.setattr(
        pilot, "audit_encoder_reference_consumption", lambda *a, **kw: dict(mocked_encoder=True)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["pulse", "--config", str(tmp_path / "unused-config.json")]
        + (["--preflight"] if mode == "preflight" else []),
    )
    if mode in ("complete", "preflight"):
        pilot.main()
    else:
        with pytest.raises(ValueError):
            pilot.main()
    if mode == "preflight":
        assert not allocations and not calls and not c.root.exists()
        return
    assert len(allocations) == 1
    expected = {"complete": 6, "body_failure": 4, "off_corrupt": 1, "early_probe_corrupt": 2}[mode]
    assert len(calls) == expected
    delta = 0.08 if profile == POSITIVE_PROFILE else -0.08
    assert calls == [(g, d) for g in GROUPS for d in (0.0, delta)][:expected]
    with sqlite3.connect(c.root / "campaign.sqlite") as db:
        state = json.loads(
            db.execute("SELECT body FROM events ORDER BY seq DESC LIMIT 1").fetchone()[0]
        )["state"]
    assert state["attempts"] == expected
    if mode == "complete":
        report = json.loads((c.root / "complete.json").read_text())
        assert len(report["results"]) == 6
        assert report["new_native_bodies"] == report["new_independent_cpu_bodies"] == 6
        assert report["assessment"]["paired_results"]["newly_unsafe_pairs"] == 1
        assert not report["teacher_qualified"]
        assert state["phase"] == "PILOT_COMPLETE_EXPANDED_VALIDATION_REQUIRED"
        declaration = json.loads((c.root / "declaration.json").read_text())
        tip, checked = read_journal(c.root, declaration)
        assert checked == state and tip != "GENESIS"
        with sqlite3.connect(c.root / "campaign.sqlite") as db:
            db.execute("UPDATE events SET digest='corrupt' WHERE seq=1")
        with pytest.raises(ValueError, match="chain"):
            read_journal(c.root, declaration)
    else:
        assert state["phase"] == "FATAL_CONTRACT_VIOLATION"
        assert not (c.root / "complete.json").exists()
