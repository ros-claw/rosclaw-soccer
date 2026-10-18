"""Synthetic nine-body NAV baseline contracts; not physical success evidence."""

import copy
import json
import sys

import numpy as np
import pytest

from rosclaw_soccer.rsi.reference_exam_profiles import (
    ZERO,
    assess_reference,
    positive_zero_scope,
    profile_info,
    require_zero_navigation_trace,
)
from rosclaw_soccer.rsi.reference_student_exam_diagnostics import measured_readout, require_cohort
from scripts.rsi_replay_first_touch_contacts import fixed_replay_scope
from tests.rsi.test_pre_parent_reference_features import measured_trace
from tests.rsi.test_reference_student_exam_diagnostics import PARENT, cohort
from tests.rsi.test_reference_student_exam_pipeline import audit, fixture, pilot


def zero_cohort():
    cfg, d, c, a = cohort()
    cfg.update(
        exam_profile=ZERO,
        max_attempts=9,
        new_physical_budget=9,
        new_independent_cpu_budget=9,
        combined_body_substep_budget=54000,
    )
    scope = positive_zero_scope(PARENT)
    info = profile_info(ZERO)
    d.update(cfg, schema=info["declaration_schema"], scope=scope)
    rows = []
    for i, j in enumerate(scope["jobs"]):
        m = d["models"][j["role"]]
        rows.append(
            dict(
                j,
                receipt_hash=f"zero-{i}",
                candidate=d["candidate"],
                reference_student_model_hash=None if m is None else m["model_hash"],
                outcome=dict(task_pass=True, safe=True),
            )
        )
    paired = assess_reference(
        ZERO,
        scope,
        rows,
        teachers={int(g): r["outcome"] for g, r in d["reused_teachers"].items()},
        original_retention={},
    )
    c.update(
        schema=info["complete_schema"],
        results=rows,
        paired_results=paired,
        new_native_bodies=9,
        new_independent_cpu_bodies=9,
        native_substeps=27000,
        independent_cpu_substeps=27000,
    )
    a.update(
        schema=info["readback_schema"],
        source_native_bodies=9,
        causal_reference_frames_reconstructed=2400,
        paired_results=paired,
        fitted_student_policy_evaluated=False,
        matched_zero_all300_reference_increments_exactly_zero=True,
        rows=[
            dict(
                job=j,
                receipt_hash=r["receipt_hash"],
                outcome=r["outcome"],
                model_hash=r["reference_student_model_hash"],
            )
            for j, r in zip(scope["jobs"], rows, strict=True)
        ],
    )
    return cfg, d, c, a


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "budget",
        "scope",
        "drop",
        "model",
        "zero_norm",
        "student_role",
        "zero_capture",
        "claim",
        "pairing",
    ],
)
def test_whole_exact_nine_zero_baseline_not_an_extra_learned_student(tamper):
    cfg, d, c, a = zero_cohort()
    if tamper == "budget":
        c["native_substeps"] = 39000
    elif tamper == "scope":
        d["scope"]["jobs"][1]["role"] = "student"
    elif tamper == "drop":
        c["results"].pop()
    elif tamper == "model":
        a["rows"][1]["model_hash"] = "student"
    elif tamper == "zero_norm":
        d["models"]["zero"] = d["models"]["student"]
    elif tamper == "student_role":
        c["results"][1]["role"] = "student"
    elif tamper == "zero_capture":
        a["matched_zero_all300_reference_increments_exactly_zero"] = False
    elif tamper == "claim":
        a["fitted_student_policy_evaluated"] = True
    elif tamper == "pairing":
        c["paired_results"] = dict(c["paired_results"], fabricated=True)
    if tamper:
        with pytest.raises(ValueError):
            require_cohort(cfg, d, c, a)
    else:
        assert require_cohort(cfg, d, c, a) == c["paired_results"]
        assert c["paired_results"]["positive_teacher_vs_zero"]["pairs"] == 8
        assert not c["paired_results"]["fitted_student_policy_evaluated"]


def test_zero_trace_requires_all300_actual_requests_and_targets_navigation_only():
    trace = measured_trace()
    nav = trace["foundation_navigation_joint_target_rad"].copy()
    trace["pre_motor_joint_target_rad"] = nav.copy()
    trace["taskspace_joint_increment_rad"] = np.zeros((300, 1, 29))
    for field in ("requested_reference_increment_rad", "applied_reference_increment_rad"):
        trace["reference_student_" + field] = np.zeros((300, 1, 12))
    require_zero_navigation_trace(trace)
    trace["reference_student_applied_reference_increment_rad"][299, 0, 0] = 1e-16
    with pytest.raises(ValueError):
        require_zero_navigation_trace(trace)


def dense():
    cfg, d, c, a = zero_cohort()
    rows = [
        dict(
            case_id=j["job"].replace("-", "_"),
            source_receipt_hash=r["receipt_hash"],
            original_outcome=copy.deepcopy(r["outcome"]),
            metrics=dict(
                maximum_linear_newton_residual_n=1e-10,
                total_robot_downward_impulse_n_s=1.0,
                first_nonfoot_robot_contact_substep=None,
            ),
        )
        for j, r in zip(d["scope"]["jobs"], c["results"], strict=True)
    ]
    replay = dict(
        schema="soccer.rsi.fixed_control_contact_replay_complete.v1",
        integrated_substeps=27000,
        attempted_step_calls=27000,
        rows=copy.deepcopy(rows),
    )
    metrics = dict(schema="soccer.rsi.contact_exit_metrics_readback.v1", rows=rows)
    return d, c, replay, metrics


@pytest.mark.parametrize("tamper", [None, "budget", "order", "partial", "newton", "nan", "outcome"])
def test_nine_dense_replays_keep_all_original_outcomes_and_no_learning_claim(tamper):
    d, c, r, m = dense()
    if tamper == "budget":
        r["integrated_substeps"] = 39000
    elif tamper == "order":
        r["rows"].reverse()
    elif tamper == "partial":
        m["rows"].pop()
    elif tamper == "newton":
        m["rows"][0]["metrics"]["maximum_linear_newton_residual_n"] = 1e-6
    elif tamper == "nan":
        m["rows"][0]["metrics"]["total_robot_downward_impulse_n_s"] = float("nan")
    elif tamper == "outcome":
        m["rows"][0]["original_outcome"]["task_pass"] = False
    if tamper:
        with pytest.raises(ValueError):
            measured_readout(d, c, r, m)
    else:
        measured = measured_readout(d, c, r, m)
        assert measured["research_decision"].startswith("MATCHED_ZERO_CONTROLS")
        assert not measured["fitted_student_policy_evaluated"]
        assert measured["distinct_derived_course_count"] == 8


def test_fixed_nine_replay_not_an_extra_policy_search():
    cases = [
        dict(case_id=j["job"].replace("-", "_"), receipt_hash=str(i))
        for i, j in enumerate(positive_zero_scope(PARENT)["jobs"])
    ]
    cfg = dict(
        replay_profile="reference-positive-zero-nine",
        new_replay_budget=9,
        new_substep_budget=27000,
        behavior_model_hash=PARENT,
        cases=cases,
    )
    assert fixed_replay_scope(cfg) == (9, 27000)
    cfg["cases"].reverse()
    with pytest.raises(ValueError):
        fixed_replay_scope(cfg)


@pytest.mark.parametrize("mode", ["normal", "preflight", "off_corruption", "body_failure"])
def test_actual_nine_orchestration_with_all_physics_and_neural_engines_mocked(
    tmp_path, monkeypatch, mode
):
    c, path, allocations, calls = fixture(tmp_path, monkeypatch, mode)
    c.cfg.update(
        exam_profile=ZERO,
        max_attempts=9,
        new_physical_budget=9,
        new_independent_cpu_budget=9,
        combined_body_substep_budget=54000,
    )
    c.scope = positive_zero_scope(c.assets.MEAN)
    path.write_text(json.dumps(c.cfg))
    pilot.require_scope(c.cfg)
    monkeypatch.setattr(
        sys,
        "argv",
        ["exam", "--config", str(path)] + (["--preflight"] if mode == "preflight" else []),
    )
    if mode in ("off_corruption", "body_failure"):
        with pytest.raises(ValueError):
            pilot.main()
        assert not (c.root / "complete.json").exists()
        with pytest.raises(ValueError, match="unused"):
            pilot.main()
        return
    pilot.main()
    if mode == "preflight":
        assert not calls and not allocations and not c.root.exists()
        return
    assert len(calls) == 9 and all(k[1]["generated_seed_episode_binding"] for k in calls)
    assert sum(k[1]["reference_student_model"] is None for k in calls) == 1
    checked = []
    # This fixture lacks raw Foundation actions by design. Pure zero-NAV
    # arithmetic is tested above; here ALL physical/NN engines are mocks.
    monkeypatch.setattr(audit, "require_zero_navigation_trace", lambda t: checked.append(True))
    output = tmp_path / "zero-readback.json"
    monkeypatch.setattr(
        sys, "argv", ["audit", "--config", str(path), "--unit", "fake", "--output", str(output)]
    )
    audit.main()
    result = json.loads(output.read_text())
    assert result["schema"] == profile_info(ZERO)["readback_schema"]
    assert (
        result["source_native_bodies"] == 9
        and result["causal_reference_frames_reconstructed"] == 2400
    )
    assert len(checked) == 8 and not result["fitted_student_policy_evaluated"]
