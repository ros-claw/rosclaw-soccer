"""Whole24 artifact fixture with STUBBED M0/ABI, never G1 physics evidence."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
from rosclaw_soccer.rsi.local_impact_perturbation_plan import declare_local_impact_perturbations
from rosclaw_soccer.rsi.native_first_touch_initial_state import NativeFirstTouchInitialState
from scripts import rsi_read_local_impact_robustness as mod
from tests.rsi.test_early_feedback_readback import fixture
from tests.rsi.test_local_impact_robustness_scope import scope


def sealed(path, d):
    d = dict(d)
    d["report_hash"] = mod.hash_json(d)
    path.write_text(json.dumps(d))
    return d


def protocol(monkeypatch, tmp_path, *, corruption=None):
    root = tmp_path / "local"
    root.mkdir()
    course_bank = tuple(
        c
        for c in ((2.68, 0.117, 0.34), (2.645, 0.104, 0.277), (2.646, 0.133, 0.516))
        for _ in range(2)
    )
    plan = declare_local_impact_perturbations(course_bank)
    native_rows = []
    source_pins = {}
    outcome = dict(task_pass=False, safe=True)
    for i, course in enumerate(course_bank):
        # Real native commitments are byte-bound, NOT report_hash envelopes.
        path = tmp_path / f"commitment-{i}.json"
        path.write_text(json.dumps(dict(course=course, motor_policy_hash=f"policy{i}")))
        h = mod.digest(path)
        source_pins[str(path)] = h
        native_rows.append(
            dict(
                receipt_hash=f"native{i}",
                outcome=outcome,
                witnesses=dict(commitment=dict(path=str(path), sha256=h)),
            )
        )
    native_path = tmp_path / "native.json"
    sealed(native_path, dict(results=native_rows))
    stage = tmp_path / "stage.json"
    sealed(stage, dict(effective_cohort_spec=dict(producer_complete=str(native_path))))
    h300_root = tmp_path / "h300"
    h300_root.mkdir()
    h300 = sealed(
        h300_root / "complete.json",
        dict(rows=[dict(causal_rollout_outcome=outcome) for _ in range(6)]),
    )
    _, template, _ = fixture()
    template["foot_heading_state"] = np.zeros((300, 1, 2))
    rows = []
    for i, job in enumerate(plan["jobs"]):
        parent = native_rows[job["source_index"]]
        initial = NativeFirstTouchInitialState(
            tuple(job["course"]),
            np.asarray(G1SonicRunupController.default_angles, dtype=np.float64),
        )
        arms = []
        for j, arm in enumerate(("original_control", "prospective_early_impact")):
            trace = {k: v.copy() for k, v in template.items()}
            trace["canonical_qpos"][0, 0, :7] = (0, 0, 0.793, 1, 0, 0, 0)
            trace["canonical_qpos"][0, 0, 7:36] = G1SonicRunupController.default_angles
            trace["canonical_qpos"][0, 0, 36:] = (*job["course"][:2], 0.13, 1, 0, 0, 0)
            vx = job["course"][2]
            trace["canonical_qvel"][0, 0, 35:] = (vx, 0, 0, 0, vx / 0.11, 0)
            if j:
                trace.update(
                    early_impact_feedback_requested_offset_nm=np.zeros((300, 1, 29)),
                    early_impact_feedback_applied_offset_nm=np.zeros((300, 1, 10, 29)),
                    early_impact_feedback_forecast_steps=np.zeros((300, 1, 1), dtype=np.int64),
                )
                folder = root / f"pair-{i}-preaction-evidence"
                folder.mkdir()
                for f in range(300):
                    sealed(
                        folder / f"frame-{f:03d}.json",
                        dict(
                            frame=f,
                            persisted_before_current_native_substeps=True,
                            future_source_frame_or_target_used=False,
                            requested_offset_nm=[0.0] * 29,
                            raw_witnesses=[],
                            selected_index=None,
                            status="NO_ACTION",
                            forecast_substeps=0,
                        ),
                    )
            file = root / f"pair-{i}-{arm}-trace.npz"
            np.savez_compressed(file, **trace)
            record = dict(
                arm=arm,
                job=job,
                source_receipt_hash=parent["receipt_hash"],
                original_policy_hash=f"policy{job['source_index']}",
                owned_continuation_enabled=bool(j),
                arrays=dict(path=str(file), sha256=mod.digest(file)),
                first_requested_correction_frame=None,
                first_physically_applied_correction_frame=None,
                review=dict(
                    initial_state_contract=initial.contract(),
                    native_attempted_steps=3000,
                    native_integrated_steps=3000,
                    forecast_attempted_steps=0,
                    forecast_integrated_steps=0,
                    frames=300,
                    current_body_policy_recomputed=True,
                    source_future_trace_or_targets_consumed=False,
                ),
                outcome=outcome,
                teacher_qualified=False,
                hardware_authorized=False,
                promotion_authorized=False,
            )
            if corruption == "source" and i == 11 and j:
                record["original_policy_hash"] = "unrelated"
            if corruption == "initial" and i == 11 and j:
                record["review"]["initial_state_contract"] = {}
            arms.append(sealed(root / f"pair-{i}-{arm}-complete.json", record))
        rows.append(
            sealed(
                root / f"pair-{i}-complete.json",
                dict(index=i, job=job, arms=arms, original_prefix_matched=True),
            )
        )
    cfg = dict(
        scope(), output_root=str(root), stage_readback=str(stage), producer_root=str(h300_root)
    )
    d = dict(
        config=cfg,
        input_file_hashes=source_pins,
        plan=plan,
        reused_h300_complete_hash=h300["report_hash"],
        old_baseline_and_early_outcomes_reused=[
            dict(index=i, baseline=outcome, early=outcome) for i in range(6)
        ],
    )
    if corruption == "plan":
        d["plan"] = dict(plan, new_unique_physical_courses=12)
    d = sealed(root / "declaration.json", d)
    c = dict(
        schema="soccer.rsi.local_impact_robustness_complete.v1",
        declaration_hash=d["report_hash"],
        rows=rows,
        new_native_steps=72000,
        forecast_steps=0,
        new_optimizer_updates=0,
        **mod.paired_summary(
            rows
            if corruption != "source"
            else [
                dict(
                    r,
                    arms=[
                        dict(a, original_policy_hash=f"policy{r['job']['source_index']}")
                        for a in r["arms"]
                    ],
                )
                for r in rows
            ]
        ),
        hardware_authorized=False,
    )
    if corruption == "totals":
        c["candidate_passes"] = 12
    if corruption == "teacher":
        c["teacher_qualified"] = True
    complete = root / "complete.json"
    sealed(complete, c)
    if corruption == "raw":
        (root / "pair-11-prospective_early_impact-trace.npz").write_bytes(b"tampered")
    if corruption == "decision":
        path = root / "pair-11-preaction-evidence/frame-299.json"
        value = json.loads(path.read_text())
        value.pop("report_hash")
        value["frame"] = 298
        sealed(path, value)
    output = tmp_path / "readback.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "read",
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
    monkeypatch.setattr(mod, "original_arrays", lambda trace: trace)
    monkeypatch.setattr(
        mod, "label_first_touch", lambda trace: SimpleNamespace(to_dict=lambda: outcome)
    )
    return output


def test_all24_arms_and3600_decisions_retained_without_fake_teacher(monkeypatch, tmp_path):
    output = protocol(monkeypatch, tmp_path)
    mod.main()
    d = json.loads(output.read_text())
    assert len(d["rows"]) == 12
    assert d["actual_native_steps"] == 72000
    assert d["actual_forecast_steps"] == d["new_physical_steps"] == d["new_neural_inferences"] == 0
    assert d["new_unique_physical_courses"] == 6 and d["original_source_clusters"] == 3
    assert d["candidate_passes"] == d["rescues"] == 0
    assert (
        d["teacher_qualified"]
        is d["training_authorized"]
        is d["independent_physical_replay"]
        is False
    )


@pytest.mark.parametrize(
    "corruption", ["plan", "source", "initial", "totals", "teacher", "raw", "decision"]
)
def test_corrupted_last_pair_or_claim_not_silently_dropped(monkeypatch, tmp_path, corruption):
    output = protocol(monkeypatch, tmp_path, corruption=corruption)
    with pytest.raises(ValueError):
        mod.main()
    assert not output.exists()


def test_nonterminal_producer_not_read_as_completed(monkeypatch, tmp_path):
    output = protocol(monkeypatch, tmp_path)

    def nonterminal(unit):
        raise ValueError("producer still running")

    monkeypatch.setattr(mod, "require_terminal", nonterminal)
    with pytest.raises(ValueError, match="still running"):
        mod.main()
    assert not output.exists()


def test_original_control_torque_check_does_not_invent_action_fields(monkeypatch):
    monkeypatch.setattr(mod, "original_arrays", lambda trace: trace)
    raw = np.zeros((300, 1, 10, 29))
    trace = dict(
        pd_requested_torque_nm=raw, torque_nm=raw.copy(), actual_actuator_force_nm=raw.copy()
    )
    names = set(trace)
    mod.verify_original_control_torques(trace)
    assert set(trace) == names
    trace["actual_actuator_force_nm"][299, 0, 9, 28] = 0.01
    with pytest.raises(ValueError, match="clipped base PD"):
        mod.verify_original_control_torques(trace)
