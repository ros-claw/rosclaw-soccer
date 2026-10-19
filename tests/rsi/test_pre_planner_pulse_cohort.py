import copy
import json

import pytest

from rosclaw_soccer.rsi.planner_navigation_pulse_cohort import (
    pulse_contact_readout,
    pulse_motion_readout,
    require_pulse_cohort,
)
from rosclaw_soccer.rsi.planner_navigation_pulse_scope import assess_pulse_scope, pulse_scope
from scripts.rsi_replay_first_touch_contacts import fixed_replay_scope
from tests.rsi.test_planner_navigation_pulse_scope import PARENT, config


def cohort():
    producer = dict(config(), behavior_model_hash=PARENT)
    scope = pulse_scope(PARENT)
    declaration = dict(
        producer,
        schema="soccer.rsi.pre_planner_pulse_six_declaration.v1",
        scope=scope,
        report_hash="decl",
        candidate={"test": "candidate"},
        zero_reference_model={"model_hash": "zero"},
    )
    receipts = [
        dict(
            j,
            receipt_hash=f"r{i}",
            outcome=dict(task_pass=False, safe=i != 1),
            qualification_alignment={"test": i},
            candidate=declaration["candidate"],
            reference_student_model_hash="zero",
            physical_substeps=3000,
            foundation_calls=300,
            sampling_draws=270,
            reference_student_inference_calls=300,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            teacher_qualified=False,
            private_fresh_accessed=False,
        )
        for i, j in enumerate(scope["jobs"])
    ]
    assessment = assess_pulse_scope(scope, receipts)
    complete = dict(
        schema="soccer.rsi.pre_planner_pulse_six_complete.v1",
        declaration_hash="decl",
        report_hash="complete",
        results=receipts,
        assessment=assessment,
        new_native_bodies=6,
        new_independent_cpu_bodies=6,
        native_substeps=18000,
        independent_cpu_substeps=18000,
        new_optimizer_updates=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    audit = dict(
        schema="soccer.rsi.pre_planner_pulse_six_terminal_readback.v1",
        complete_hash="complete",
        assessment=assessment,
        rows=[
            dict(
                job=j,
                receipt_hash=r["receipt_hash"],
                outcome=r["outcome"],
                alignment=r["qualification_alignment"],
            )
            for j, r in zip(scope["jobs"], receipts, strict=True)
        ],
        source_native_bodies=6,
        zero_reference_frames_reconstructed=1800,
        new_physical_steps=0,
        new_foundation_inferences=0,
        optimizer_updates=0,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return producer, declaration, complete, audit


def dense(d, c):
    rows = [
        dict(
            case_id=j["job"].replace("-", "_"),
            source_receipt_hash=r["receipt_hash"],
            original_outcome=r["outcome"],
            report_hash=f"dense{i}",
        )
        for i, (j, r) in enumerate(zip(d["scope"]["jobs"], c["results"], strict=True))
    ]
    replay = dict(
        schema="soccer.rsi.fixed_control_contact_replay_complete.v1",
        integrated_substeps=18000,
        attempted_step_calls=18000,
        rows=rows,
    )
    metrics = dict(
        schema="soccer.rsi.contact_exit_metrics_readback.v1",
        new_physical_steps=0,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        rows=[
            dict(
                case_id=r["case_id"],
                source_receipt_hash=r["source_receipt_hash"],
                original_outcome=r["original_outcome"],
                replay_result_hash=r["report_hash"],
                metrics=dict(
                    maximum_linear_newton_residual_n=0.0,
                    total_robot_downward_impulse_n_s=0.0,
                    first_nonfoot_robot_contact_substep=123 if i == 1 else None,
                ),
            )
            for i, r in enumerate(rows)
        ],
    )
    return replay, metrics


def test_whole_failed_cohort_and_dense_diagnostics_do_not_create_training_success():
    p, d, c, a = json.loads(json.dumps(cohort()))
    original = copy.deepcopy((p, d, c, a))
    assert require_pulse_cohort(p, d, c, a) == c["assessment"]
    replay, metrics = dense(d, c)
    r = pulse_contact_readout(d, c, replay, metrics)
    assert len(r["rows"]) == 6
    assert r["pulse_unsafe_endpoints"] == r["pulse_endpoints_with_nonfoot_over1n"] == 1
    assert r["new_closed_loop_policy_trials"] == 0
    assert not r["teacher_qualified"] and not r["dataset_admission_authorized"]
    assert (p, d, c, a) == original


@pytest.mark.parametrize(
    "mutation",
    [
        "partial",
        "order",
        "new_steps",
        "missing_cpu",
        "truthy",
        "alignment",
        "model",
        "label",
        "assessment",
        "retag",
        "candidate",
    ],
)
def test_whole_cohort_rejects_partial_retagged_or_forged_results(mutation):
    p, d, c, a = cohort()
    if mutation == "partial":
        c["results"].pop()
    elif mutation == "order":
        a["rows"].reverse()
    elif mutation == "new_steps":
        a["new_physical_steps"] = 1
    elif mutation == "missing_cpu":
        c["new_independent_cpu_bodies"] = 5
    elif mutation == "truthy":
        c["new_optimizer_updates"] = False
    elif mutation == "alignment":
        a["rows"][0]["alignment"] = {}
    elif mutation == "model":
        c["results"][0]["reference_student_model_hash"] = "other"
    elif mutation == "label":
        a["rows"][0]["outcome"] = dict(safe=True, task_pass=True)
    elif mutation == "assessment":
        c["assessment"] = {}
    elif mutation == "retag":
        c["results"][0]["group"] = 92
    else:
        c["results"][0]["candidate"] = {}
    with pytest.raises(ValueError):
        require_pulse_cohort(p, d, c, a)


@pytest.mark.parametrize(
    "mutation",
    [
        "partial",
        "duplicate",
        "order",
        "source",
        "result",
        "label",
        "nan",
        "newton",
        "counter",
        "new_trials",
    ],
)
def test_dense_readout_preserves_all_failures_and_original_newton_threshold(mutation):
    _, d, c, _ = cohort()
    replay, metrics = dense(d, c)
    if mutation == "partial":
        metrics["rows"].pop()
    elif mutation == "duplicate":
        metrics["rows"][1] = metrics["rows"][0]
    elif mutation == "order":
        replay["rows"].reverse()
    elif mutation == "source":
        replay["rows"][0]["source_receipt_hash"] = "other"
    elif mutation == "result":
        metrics["rows"][0]["replay_result_hash"] = "other"
    elif mutation == "label":
        metrics["rows"][0]["original_outcome"] = {}
    elif mutation == "nan":
        metrics["rows"][0]["metrics"]["total_robot_downward_impulse_n_s"] = float("nan")
    elif mutation == "newton":
        metrics["rows"][0]["metrics"]["maximum_linear_newton_residual_n"] = 1.01e-7
    elif mutation == "counter":
        replay["attempted_step_calls"] = 18001
    else:
        metrics["new_physical_steps"] = True
    with pytest.raises(ValueError):
        pulse_contact_readout(d, c, replay, metrics)


@pytest.mark.parametrize("mutation", [None, "partial", "order", "steps", "receipt", "rename"])
def test_fixed_six_recorded_torque_scope_never_becomes_an_open_search(mutation):
    _, d, c, _ = cohort()
    replay, _ = dense(d, c)
    cfg = dict(
        replay_profile="pre-planner-navigation-six",
        behavior_model_hash=PARENT,
        new_replay_budget=6,
        new_substep_budget=18000,
        cases=[
            dict(case_id=r["case_id"], receipt_hash=r["source_receipt_hash"])
            for r in replay["rows"]
        ],
    )
    if mutation == "partial":
        cfg["cases"].pop()
    elif mutation == "order":
        cfg["cases"].reverse()
    elif mutation == "steps":
        cfg["new_substep_budget"] += 1
    elif mutation == "receipt":
        cfg["cases"][1]["receipt_hash"] = cfg["cases"][0]["receipt_hash"]
    elif mutation == "rename":
        cfg["cases"][0]["case_id"] = "more_favorable_course"
    if mutation is None:
        assert fixed_replay_scope(cfg) == (6, 18000)
    else:
        with pytest.raises(ValueError):
            fixed_replay_scope(cfg)


@pytest.mark.parametrize(
    "mutation",
    [None, "running", "source", "audit_hash", "journal", "receipt", "witness", "foreign_output"],
)
def test_loader_authenticates_both_terminal_units_owned_files_and_journal(
    tmp_path, monkeypatch, mutation
):
    from scripts import rsi_load_pre_planner_cohort as module

    p, d, c, a = cohort()
    source, root = tmp_path / "source", tmp_path / "output"
    (source / "scripts").mkdir(parents=True)
    root.mkdir()
    reader = source / "scripts/rsi_read_pre_planner_navigation.py"
    reader.write_text("# frozen reader\n")
    witness = tmp_path / "witness"
    witness.write_bytes(b"original")
    p.update(source_root=str(source), source_head="frozen", output_root=str(root))
    d.update(p, input_file_hashes={str(witness): module.digest(witness)}, prepared_pins={})
    a.update(
        report_hash="audit",
        reader_source_hash=module.digest(reader),
        ledger_tip="tip",
        input_file_hashes={},
    )
    pp = tmp_path / "producer.json"
    pp.write_text(json.dumps(p))
    documents = {
        str(root / "declaration.json"): d,
        str(root / "complete.json"): c,
        str(tmp_path / "audit.json"): a,
    }
    for i, r in enumerate(c["results"]):
        documents[str(root / f"receipt-{i:04d}.json")] = copy.deepcopy(r)
    for name, doc in documents.items():
        from pathlib import Path

        Path(name).write_text(json.dumps(doc))
    cfg = dict(
        producer_unit="native.service",
        audit_unit="audit.service",
        producer_config=str(pp),
        producer_config_file_hash=module.digest(pp),
        producer_declaration=str(root / "declaration.json"),
        producer_complete=str(root / "complete.json"),
        source_terminal_readback=str(tmp_path / "audit.json"),
        source_terminal_readback_hash="audit",
        audit_source=str(reader),
    )
    called = []

    def terminal(unit):
        called.append(unit)
        if mutation == "running":
            raise ValueError("not terminal")

    monkeypatch.setattr(module, "require_terminal", terminal)
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda argv, **kw: (
            ("other" if mutation == "source" else "frozen") if argv[1] == "rev-parse" else ""
        ),
    )
    state = {"completed": {r["job"]: r for r in c["results"]}}
    monkeypatch.setattr(
        module, "read_journal", lambda *args: ("other" if mutation == "journal" else "tip", state)
    )
    if mutation == "audit_hash":
        cfg["source_terminal_readback_hash"] = "other"
    elif mutation == "receipt":
        documents[str(root / "receipt-0000.json")]["candidate"] = {}
    elif mutation == "witness":
        witness.write_bytes(b"changed")
    elif mutation == "foreign_output":
        cfg["producer_complete"] = str(tmp_path / "complete.json")
    pins = {}

    def sealed(path, key="report_hash"):
        pins[str(path)] = module.digest(path)
        return documents[str(path)]

    if mutation is None:
        r = module.load_closed_pulse(cfg, sealed=sealed, pins=pins)
        assert r["assessment"] == c["assessment"]
        assert called == ["native.service", "audit.service"]
        assert pins[str(witness)] == module.digest(witness)
    else:
        with pytest.raises(ValueError):
            module.load_closed_pulse(cfg, sealed=sealed, pins=pins)


@pytest.mark.parametrize("mutation", [None, "early_motion", "joint_reference", "nan_command"])
def test_motion_readout_measures_navigation_and_body_without_inventing_skill_gain(mutation):
    import numpy as np

    from tests.rsi.test_reference_feedback_divergence import traces

    off, _ = traces()
    off["navigation_command"] = np.zeros((300, 1, 3))
    # Synthetic padding satisfies the original33-array envelope, not a native rollout.
    for i in range(33 - sum(not k.startswith("reference_student_") for k in off)):
        off[f"synthetic_original_{i}"] = np.zeros((300, 1, 1))
    pulse = copy.deepcopy(off)
    pulse["navigation_command"][100:160, 0, 1] = -0.08
    pulse["joint_position_rad"][101:, 0, 0] = 0.01
    if mutation == "early_motion":
        pulse["joint_position_rad"][99, 0, 0] = 1e-12
    elif mutation == "joint_reference":
        pulse["reference_student_requested_reference_increment_rad"][150, 0, 0] = 0.001
    elif mutation == "nan_command":
        pulse["navigation_command"][100, 0, 1] = np.nan
    if mutation is None:
        r = pulse_motion_readout(off, pulse)
        assert r["first_navigation_difference_frame"] == 100
        assert r["before_action_prefix"]["compared_frames"] == 100
        assert (
            r["measured_reference_body_ball_channels"]["channels"]["joint_position_before"][
                "first_frame_above"
            ]["0.001"]
            == 101
        )
        assert r["new_physical_steps"] == 0
        assert r["action_authority_is_not_task_improvement"]
    else:
        with pytest.raises(ValueError):
            pulse_motion_readout(off, pulse)
