"""Whole-cohort artifact protocol with stubbed labels, NOT actual G1 evidence."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.physics.g1_early_feedback_forecast import G1EarlyFeedbackForecast
from rosclaw_soccer.rsi.early_feedback_impact_score import early_feedback_impact_score
from scripts import rsi_read_early_impact_h300 as mod
from tests.rsi.test_early_feedback_readback import fixture
from tests.rsi.test_early_impact_h300_scope import scope


def sealed(path, value):
    d = dict(value)
    d["report_hash"] = mod.hash_json(d)
    path.write_text(json.dumps(d))
    return d


def protocol(monkeypatch, tmp_path, *, wrong_total=False, selection=False):
    root = tmp_path / "pilot"
    root.mkdir()
    rows, parents, jobs = [], [], []
    forecast, original, _ = fixture()
    original["fixture_task_marker"] = np.zeros((300, 1))
    original["foot_heading_state"] = np.zeros((300, 1, 2))
    if selection:
        forecast["foot_contact_peak_force_n_after_substep"][0, 0] = 2
        forecast["completed_contact_peak_force_n"][:10, 0] = 2
        original["force_n"][30, 0, 0] = 2
    score = early_feedback_impact_score(
        G1EarlyFeedbackForecast(
            "fixture-donor", forecast["end_integration_state"], forecast, (), ()
        ),
        np.zeros(29),
    )
    for i in range(6):
        job = {"case": i}
        jobs.append(job)
        t = {k: v.copy() for k, v in original.items()}
        t["fixture_task_marker"][:] = int(i >= 2)
        p = tmp_path / f"source-{i}.npz"
        np.savez_compressed(p, **t)
        outcome = dict(task_pass=i >= 2, safe=True)
        parent = dict(
            receipt_hash=f"source-{i}",
            outcome=outcome,
            witnesses=dict(trace=dict(path=str(p), sha256=mod.digest(p))),
        )
        parents.append(parent)
        t.update(
            early_impact_feedback_requested_offset_nm=np.zeros((300, 1, 29)),
            early_impact_feedback_applied_offset_nm=np.zeros((300, 1, 10, 29)),
            early_impact_feedback_forecast_steps=np.zeros((300, 1, 1), dtype=np.int64),
        )
        selected = selection and i == 0
        if selected:
            t["early_impact_feedback_forecast_steps"][30, 0, 0] = 80
        path = root / f"case-{i}-trace.npz"
        np.savez_compressed(path, **t)
        row = sealed(
            root / f"case-{i}-complete.json",
            dict(
                index=i,
                job=job,
                source_receipt_hash=parent["receipt_hash"],
                original_outcome=outcome,
                causal_rollout_outcome=outcome,
                first_requested_correction_frame=None,
                first_physically_applied_correction_frame=None,
                arrays=dict(path=str(path), sha256=mod.digest(path)),
                causal_review=dict(
                    native_integrated_steps=3000,
                    native_attempted_steps=3000,
                    forecast_integrated_steps=80 if selected else 0,
                    forecast_attempted_steps=80 if selected else 0,
                    source_future_trace_or_targets_consumed=False,
                ),
            ),
        )
        rows.append(row)
        folder = root / f"case-{i}-preaction-evidence"
        folder.mkdir()
        for frame in range(300):
            witnesses = []
            if selected and frame == 30:
                for j, name in enumerate(("baseline", "repeat_zero")):
                    branch = folder / f"frame-{frame:03d}-candidate-{j:02d}.npz"
                    np.savez_compressed(branch, **forecast)
                    witnesses.append(
                        dict(
                            name=name,
                            path=str(branch),
                            sha256=mod.digest(branch),
                            donor_hash="fixture-donor",
                            offset_nm=[0.0] * 29,
                            score=score,
                        )
                    )
            sealed(
                folder / f"frame-{frame:03d}.json",
                dict(
                    frame=frame,
                    persisted_before_current_native_substeps=True,
                    future_source_frame_or_target_used=False,
                    requested_offset_nm=[0.0] * 29,
                    raw_witnesses=witnesses,
                    selected_index=0 if witnesses else None,
                    status="NO_ACTION",
                    forecast_substeps=80 if witnesses else 0,
                ),
            )
    native = tmp_path / "native.json"
    sealed(native, dict(results=parents))
    native_decl = tmp_path / "native-declaration.json"
    sealed(native_decl, dict(scope=dict(jobs=jobs)))
    stage = tmp_path / "stage.json"
    sealed(
        stage,
        dict(
            effective_cohort_spec=dict(
                producer_complete=str(native), producer_declaration=str(native_decl)
            )
        ),
    )
    cfg = dict(scope(), output_root=str(root), stage_readback=str(stage))
    declaration = sealed(root / "declaration.json", dict(config=cfg, input_file_hashes={}))
    complete = root / "complete.json"
    sealed(
        complete,
        dict(
            schema="soccer.rsi.early_impact_h300_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            actual_causal_rollout_steps=18000,
            actual_forecast_steps=80 if selection else 0,
            reused_original_controls=6,
            original_passes=4,
            candidate_passes=6 if wrong_total else 4,
            rescues=0,
            regressions=0,
            added_unsafe=0,
            new_optimizer_updates=0,
            new_neural_policy_trained=False,
            independent_physics_qualified=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    output = tmp_path / "readback.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "readback",
            "--root",
            str(root),
            "--unit",
            "fixture.service",
            "--source-head",
            "pinned",
            "--complete-file-hash",
            mod.digest(complete),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(mod, "require_terminal", lambda unit: None)
    monkeypatch.setattr(
        mod.subprocess,
        "check_output",
        lambda args, **kw: "pinned" if args[1] == "rev-parse" else "",
    )
    monkeypatch.setattr(
        mod,
        "label_first_touch",
        lambda t: SimpleNamespace(
            to_dict=lambda: dict(task_pass=bool(t["fixture_task_marker"][0, 0]), safe=True)
        ),
    )
    return output


def test_all_six_no_action_reports_and_1800_preaction_records_checked(monkeypatch, tmp_path):
    output = protocol(monkeypatch, tmp_path)
    mod.main()
    d = json.loads(output.read_text())
    assert len(d["rows"]) == 6 and d["candidate_passes"] == 4 and d["rescues"] == 0
    assert not d["teacher_qualified"] and not d["independent_neural_implementation"]
    assert d["new_physical_steps"] == d["new_neural_inferences"] == 0


def test_claimed_task_gain_without_primitive_support_is_rejected(monkeypatch, tmp_path):
    output = protocol(monkeypatch, tmp_path, wrong_total=True)
    with pytest.raises(ValueError, match="totals"):
        mod.main()
    assert not output.exists()


def test_selected_bank_is_checked_against_actual_all_four_control_periods(monkeypatch, tmp_path):
    output = protocol(monkeypatch, tmp_path, selection=True)
    mod.main()
    d = json.loads(output.read_text())
    alignment = d["rows"][0]["selected_window_alignment"][0]
    assert alignment["full80ms_execution_alignment_verified"]
    assert alignment["physical_substeps_compared"] == 40
    assert d["actual_forecast_steps"] == 80
