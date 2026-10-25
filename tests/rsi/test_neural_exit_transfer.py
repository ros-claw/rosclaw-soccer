import copy
import json

import pytest

from rosclaw_soccer.rsi.neural_exit_transfer import (
    EXIT_RETENTION_INDICES,
    EXIT_SOURCE_INDICES,
    EXIT_TRANSFER_INDICES,
    EXIT_TRANSFER_WITH_CONTROLS,
    require_protected_exit_source,
)


def evidence():
    flags = dict(hardware_authorized=False, promotion_authorized=False, goal_complete=False)
    outcomes = [dict(task_pass=i != 2) for i in EXIT_SOURCE_INDICES]
    audit = dict(
        status="NEURAL_EXIT_RAW_HISTORY_AND_CPU_NN_VERIFIED_NOT_TEACHER",
        rescues=[1],
        regressions=[],
        qualified_for_expansion=False,
        new_physical_steps=0,
        neural_optimizer_updates=0,
        actual_native_steps=28800,
        cpu_velocity_nn_readback_calls=1440,
        results=[
            dict(course_index=i, C=o, C_healthy=True)
            for i, o in zip(EXIT_SOURCE_INDICES, outcomes, strict=True)
        ],
        producer_terminal=dict(
            source_head="source", manager_completion_verified=True, producer_restarted=False
        ),
        **flags,
    )
    producer = dict(
        status="OBSERVED_NEURAL_EXIT_PILOT_COMPLETE_NOT_TEACHER_QUALIFIED",
        source_head="source",
        actual_worlds=8,
        actual_native_steps=28800,
        neural_optimizer_updates=0,
        qualified_for_expansion=False,
        **flags,
    )
    rows = [
        dict(
            course_index=i,
            arm="C",
            post_contact_exit=True,
            protect_passing_entry_prediction=True,
            exit_gate_uses_source_course_or_outcome=False,
            outcome=o,
            healthy_full_horizon=True,
            physical_teacher_qualified=False,
            **flags,
        )
        for i, o in zip(EXIT_SOURCE_INDICES, outcomes, strict=True)
    ]
    return audit, producer, rows


def test_fixed_transfer_is_balanced_source_disjoint_with_separate_controls():
    assert len(EXIT_TRANSFER_INDICES) == 16
    assert not set(EXIT_TRANSFER_INDICES) & set(EXIT_SOURCE_INDICES)
    assert [sum(i // 10 == c for i in EXIT_TRANSFER_INDICES) for c in range(4)] == [4] * 4
    assert EXIT_RETENTION_INDICES == (0, 30)
    assert len(EXIT_TRANSFER_WITH_CONTROLS) == len(set(EXIT_TRANSFER_WITH_CONTROLS)) == 18
    assert 2 * len(EXIT_TRANSFER_WITH_CONTROLS) * 3600 == 129600
    assert 2 * len(EXIT_TRANSFER_WITH_CONTROLS) * 37 * 1280 == 1704960


def test_only_protected_audited_source_unlocks_new_sim_study_not_promotion():
    audit, producer, rows = evidence()
    require_protected_exit_source(audit, producer, rows)
    assert audit["qualified_for_expansion"] is False
    rows[1]["protect_passing_entry_prediction"] = False
    with pytest.raises(ValueError):
        require_protected_exit_source(audit, producer, rows)


@pytest.mark.parametrize(
    "defect",
    [
        dict(regressions=[0]),
        dict(rescues=[]),
        dict(qualified_for_expansion=True),
        dict(new_physical_steps=1),
        dict(cpu_velocity_nn_readback_calls=1439),
        dict(actual_native_steps=28799),
        dict(hardware_authorized=True),
        dict(promotion_authorized=True),
        dict(results=[]),
    ],
)
def test_incomplete_or_unsafe_source_cannot_unlock_transfer(defect):
    audit, producer, rows = evidence()
    with pytest.raises(ValueError):
        require_protected_exit_source(audit | defect, producer, rows)


@pytest.mark.parametrize("defect", ["body", "outcome", "source", "unprotected", "real"])
def test_all_four_actual_rows_and_completion_must_be_bound(defect):
    audit, producer, rows = copy.deepcopy(evidence())
    if defect == "body":
        rows[3]["healthy_full_horizon"] = False
    elif defect == "outcome":
        rows[1]["outcome"] = dict(task_pass=False)
    elif defect == "source":
        producer["source_head"] = "other"
    elif defect == "unprotected":
        rows[0]["protect_passing_entry_prediction"] = False
    else:
        rows[0]["hardware_authorized"] = True
    with pytest.raises(ValueError):
        require_protected_exit_source(audit, producer, rows)


def test_file_gate_requires_unchanged_audit_bound_actual_c_rows(tmp_path):
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_run_neural_exit_pilot import admit_protected_exit_source

    audit, producer, rows = evidence()
    root = tmp_path / "producer"
    root.mkdir()
    bound = {}
    producer["results"] = []
    for row in rows:
        index = row["course_index"]
        path = root / f"course-{index:02d}" / "C.json"
        path.parent.mkdir()
        path.write_text(json.dumps(row))
        expected = digest(path)
        bound[str(path)] = expected
        producer["results"].append(
            dict(
                course_index=index,
                arms=dict(C=dict(path=str(path.relative_to(root)), hash=expected)),
            )
        )
    cp = root / "complete.json"
    cp.write_text(json.dumps(producer))
    bound[str(cp)] = digest(cp)
    audit["input_file_hashes"] = bound
    audit["producer_terminal"]["output_root"] = str(root)
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    expected_audit = digest(path)
    pins = {}
    admit_protected_exit_source(path, expected_audit, pins)
    assert pins[str(path)] == expected_audit
    changed = root / "course-01" / "C.json"
    row = json.loads(changed.read_text())
    row["protect_passing_entry_prediction"] = False
    changed.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="audit-bound"):
        admit_protected_exit_source(path, expected_audit, {})


@pytest.mark.parametrize("defect", [None, "negative", "bool", "padding", "nn_calls", "cap"])
def test_forecast_budget_separates_requested_from_early_failed_integrations(defect):
    from scripts.rsi_review_neural_exit_pilot import verify_forecast_accounting

    counters = dict(
        forecast_calls=37,
        attempted_native_steps=47360,
        integrated_native_steps=46000,
        future_nn_calls=4600,
        maximum_native_steps=47360,
    )
    if defect == "negative":
        counters["forecast_calls"] = -1
    elif defect == "bool":
        counters["future_nn_calls"] = True
    elif defect == "padding":
        counters["integrated_native_steps"] = 47370
    elif defect == "nn_calls":
        counters["future_nn_calls"] = 4601
    elif defect == "cap":
        counters["maximum_native_steps"] += 1280
    row = dict(entry_control=80, forecast_counters=counters)
    if defect:
        with pytest.raises(ValueError):
            verify_forecast_accounting(row)
    else:
        assert verify_forecast_accounting(row) == (47360, 46000)
