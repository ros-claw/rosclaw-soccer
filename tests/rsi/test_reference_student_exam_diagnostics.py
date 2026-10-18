"""Synthetic whole-cohort and measured diagnostic contracts, no real bodies."""

import copy

import numpy as np
import pytest
from rosclaw.growth.causal_residual_memory import initial_parameters

from rosclaw_soccer.rsi.reference_student_exam_diagnostics import measured_readout, require_cohort
from rosclaw_soccer.rsi.reference_student_exam_scope import assess_exam, exam_scope, zero_model_like
from rosclaw_soccer.rsi.reference_student_execution import make_model
from scripts.rsi_fit_pre_parent_reference import FIT_SEED
from scripts.rsi_replay_first_touch_contacts import fixed_replay_scope
from tests.rsi.test_reference_student_exam_pipeline import config

PARENT, WORLD = "sha256:" + "a" * 64, "sha256:" + "b" * 64


def cohort():
    cfg = dict(config(), behavior_model_hash=PARENT, mesh_task_contact_world_hash=WORLD)
    p = initial_parameters(135, 12, hidden_dimension=64, seed=FIT_SEED)
    p["head_bias"][4] = 0.05
    student = make_model(
        p, mean=np.zeros(135), scale=np.ones(135), world_hash=WORLD, parent_model_hash=PARENT
    )
    models = dict(off=None, student=student, zero=zero_model_like(student, seed=FIT_SEED))
    scope = exam_scope(PARENT)
    jobs = scope["jobs"]
    teachers = {
        g: dict(task_pass=i % 2 == 0, safe=True)
        for i, g in enumerate((12, 13, 28, 29, 76, 77, 92, 93))
    }
    old = {g: dict(task_pass=True, safe=True) for g in (4, 32)}
    declaration = dict(
        cfg,
        schema="soccer.rsi.reference_student_exam_declaration.v1",
        report_hash="declaration",
        scope=scope,
        models=models,
        candidate=dict(test="synthetic"),
        reused_teachers={str(g): dict(outcome=o) for g, o in teachers.items()},
        reused_legacy_retention={str(g): dict(outcome=o) for g, o in old.items()},
    )
    receipts = []
    for i, j in enumerate(jobs):
        m = models[j["role"]]
        receipts.append(
            dict(
                j,
                receipt_hash=f"r{i}",
                candidate=declaration["candidate"],
                outcome=dict(task_pass=True, safe=True),
                **({} if m is None else dict(reference_student_model_hash=m["model_hash"])),
            )
        )
    pairs = assess_exam(scope, receipts, teachers=teachers, original_retention=old)
    complete = dict(
        schema="soccer.rsi.reference_student_exam_complete.v1",
        report_hash="complete",
        declaration_hash="declaration",
        results=receipts,
        paired_results=pairs,
        new_native_bodies=13,
        new_independent_cpu_bodies=13,
        native_substeps=39000,
        independent_cpu_substeps=39000,
        new_optimizer_updates=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
        fresh_exam=False,
    )
    audit = dict(
        schema="soccer.rsi.reference_student_exam_terminal_readback.v1",
        complete_hash="complete",
        rows=[
            dict(
                job=j,
                receipt_hash=r["receipt_hash"],
                outcome=copy.deepcopy(r["outcome"]),
                model_hash=None if models[j["role"]] is None else models[j["role"]]["model_hash"],
            )
            for j, r in zip(jobs, receipts, strict=True)
        ],
        source_native_bodies=13,
        causal_reference_frames_reconstructed=3600,
        new_physical_steps=0,
        new_optimizer_updates=0,
        whole_input_and_witness_files_verified=True,
        shared_payload_logical_seals_rebuilt=True,
        original_off_all33_arrays_bit_exact=True,
        all_pre_action_prefixes_bit_exact=True,
        official_52_case_exam_completed=False,
        successor_readiness_qualified=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
        fresh_exam=False,
        paired_results=copy.deepcopy(pairs),
    )
    return cfg, declaration, complete, audit


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "partial",
        "bool",
        "model",
        "world",
        "parent",
        "zero_norm",
        "outcome",
        "order",
        "old_dropped",
        "grant",
        "scope",
        "pairing",
    ],
)
def test_entire_frozen_model_pairing_and_old_failures_are_required(tamper):
    cfg, d, c, a = cohort()
    if tamper == "partial":
        c["results"].pop()
    elif tamper == "bool":
        a["new_optimizer_updates"] = False
    elif tamper == "model":
        a["rows"][1]["model_hash"] = "other"
    elif tamper == "world":
        cfg["mesh_task_contact_world_hash"] = PARENT
        d["mesh_task_contact_world_hash"] = PARENT
    elif tamper == "parent":
        cfg["behavior_model_hash"] = WORLD
    elif tamper == "zero_norm":
        z = d["models"]["zero"]
        mean = np.asarray(z["mean"])
        mean[0] = 0.2
        d["models"]["zero"] = make_model(
            z["parameters"], mean=mean, scale=z["scale"], world_hash=WORLD, parent_model_hash=PARENT
        )
    elif tamper == "outcome":
        a["rows"][1]["outcome"]["task_pass"] = False
    elif tamper == "order":
        a["rows"].reverse()
    elif tamper == "old_dropped":
        d["reused_teachers"].pop("13")
    elif tamper == "grant":
        a["promotion_authorized"] = True
    elif tamper == "scope":
        d["scope"]["source_scene_clusters"] = 13
    elif tamper == "pairing":
        a["paired_results"]["positive_teacher_vs_student"]["rescued_pairs"] = 99
    if tamper:
        with pytest.raises(ValueError):
            require_cohort(cfg, d, c, a)
    else:
        assert require_cohort(cfg, d, c, a) == c["paired_results"]


def dense():
    cfg, d, c, a = cohort()
    rows = [
        dict(
            case_id=j["job"].replace("-", "_"),
            source_receipt_hash=r["receipt_hash"],
            original_outcome=copy.deepcopy(r["outcome"]),
            metrics=dict(
                first_nonfoot_robot_contact_substep=None,
                total_robot_downward_impulse_n_s=0.2,
                maximum_linear_newton_residual_n=1e-12,
            ),
        )
        for j, r in zip(d["scope"]["jobs"], c["results"], strict=True)
    ]
    replay = dict(
        schema="soccer.rsi.fixed_control_contact_replay_complete.v1",
        integrated_substeps=39000,
        attempted_step_calls=39000,
        rows=copy.deepcopy(rows),
    )
    metrics = dict(schema="soccer.rsi.contact_exit_metrics_readback.v1", rows=rows)
    return d, c, replay, metrics


@pytest.mark.parametrize(
    "tamper", [None, "partial", "duplicate", "order", "outcome", "newton", "nan", "budget"]
)
def test_dense_readout_is_complete_finite_and_preserves_original_task_labels(tamper):
    d, c, r, m = dense()
    if tamper == "partial":
        m["rows"].pop()
    elif tamper == "duplicate":
        m["rows"][1] = m["rows"][0]
    elif tamper == "order":
        r["rows"].reverse()
    elif tamper == "outcome":
        m["rows"][1]["original_outcome"]["task_pass"] = False
    elif tamper == "newton":
        m["rows"][1]["metrics"]["maximum_linear_newton_residual_n"] = 1e-5
    elif tamper == "nan":
        m["rows"][1]["metrics"]["total_robot_downward_impulse_n_s"] = float("nan")
    elif tamper == "budget":
        r["integrated_substeps"] = 78000
    if tamper:
        with pytest.raises(ValueError):
            measured_readout(d, c, r, m)
    else:
        read = measured_readout(d, c, r, m)
        assert len(read["rows"]) == 13
        assert read["research_decision"].startswith("LOCAL_STUDENT_GAIN")
        assert not read["teacher_qualified"] and not read["fresh_exam"]


@pytest.mark.parametrize(
    "condition", ["unsafe", "nonfoot", "retention", "below", "mixed", "unchanged", "gain"]
)
def test_scientific_branching_never_auto_promotes(condition):
    d, c, r, m = dense()
    student_positive = [
        i
        for i, j in enumerate(d["scope"]["jobs"])
        if j["role"] == "student" and j["plan"] == "positive19"
    ]
    if condition == "unsafe":
        c["results"][1]["outcome"].update(safe=False, task_pass=False)
    elif condition == "nonfoot":
        m["rows"][1]["metrics"]["first_nonfoot_robot_contact_substep"] = 1900
    elif condition == "retention":
        c["results"][-1]["outcome"]["task_pass"] = False
    elif condition in ("below", "mixed", "unchanged"):
        for i in student_positive:
            g = c["results"][i]["group"]
            c["results"][i]["outcome"] = copy.deepcopy(d["reused_teachers"][str(g)]["outcome"])
            if condition == "below":
                c["results"][i]["outcome"]["task_pass"] = False
        if condition == "mixed":
            for i in (1, 2):
                c["results"][i]["outcome"]["task_pass"] = not c["results"][i]["outcome"][
                    "task_pass"
                ]
    for i, item in enumerate(c["results"]):
        r["rows"][i]["original_outcome"] = copy.deepcopy(item["outcome"])
        m["rows"][i]["original_outcome"] = copy.deepcopy(item["outcome"])
    c["paired_results"] = assess_exam(
        d["scope"],
        c["results"],
        teachers={int(k): v["outcome"] for k, v in d["reused_teachers"].items()},
        original_retention={int(k): v["outcome"] for k, v in d["reused_legacy_retention"].items()},
    )
    read = measured_readout(d, c, r, m)
    expected = {
        "unsafe": "REJECT",
        "nonfoot": "REJECT",
        "retention": "RETENTION_REGRESSION",
        "below": "STUDENT_BELOW",
        "mixed": "MIXED_POSITIVE",
        "unchanged": "NO_NET",
        "gain": "LOCAL_STUDENT",
    }[condition]
    assert read["research_decision"].startswith(expected)
    assert not read["promotion_authorized"] and not read["neural_training_authorized"]
    if condition == "nonfoot":
        assert read["rows"][1]["original_outcome"]["task_pass"] is True


@pytest.mark.parametrize("tamper", [None, "partial", "order", "budget", "duplicate"])
def test_fixed_thirteen_profile_is_not_thirteen_new_policy_trials(tamper):
    cases = [
        dict(case_id=j["job"].replace("-", "_"), receipt_hash=f"r{i}")
        for i, j in enumerate(exam_scope(PARENT)["jobs"])
    ]
    cfg = dict(
        replay_profile="reference-student-thirteen",
        new_replay_budget=13,
        new_substep_budget=39000,
        behavior_model_hash=PARENT,
        cases=cases,
    )
    if tamper == "partial":
        cases.pop()
    elif tamper == "order":
        cases.reverse()
    elif tamper == "budget":
        cfg["new_substep_budget"] = 78000
    elif tamper == "duplicate":
        cases[1]["receipt_hash"] = cases[0]["receipt_hash"]
    if tamper:
        with pytest.raises(ValueError):
            fixed_replay_scope(cfg)
    else:
        assert fixed_replay_scope(cfg) == (13, 39000)
