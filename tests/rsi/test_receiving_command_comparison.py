"""Synthetic review fixtures are not new physical evidence."""

from copy import deepcopy

import pytest

from rosclaw_soccer.rsi.receiving_command_comparison import (
    BC_OWN,
    BC_SOURCE,
    INDICES,
    TASK_OWN,
    TASK_SOURCE,
    admit_pair,
    compare_records,
)

H = "sha256:" + "a" * 64
T = "sha256:" + "b" * 64


def pair(task=False):
    rows = []
    for i in range(12):
        rows.append(
            dict(
                index=i,
                source_case_index=INDICES[i // 2],
                course=dict(state_id=26, target=[1, 2, 0.11]),
                sampling_seed=100 + i // 2,
                controls=389,
                native_steps=3890,
                actor_calls=1945,
                observed_recovery_controls=100,
                all_actor_actions_actually_executed=True,
                receiver_command_own_body_and_actual_feedback=True,
                receiver_command_teacher_actions_executed=False,
                receiver_command_updates_during_episode=0,
                receiver_motor_optimizer_updates=0,
                assisted=False,
                external_teacher=False,
                controller_kind=(
                    "receiving-neural-task-command-dagger-native-v2"
                    if task
                    else "receiving-neural-command-bc-native-v1"
                ),
                A_recovery=dict(controls=100, neural_rows=100, native_steps=1000),
                artifacts=dict(initial=dict(hash="sha256:" + str(i // 6 + (i >= 10)) * 64)),
                outcome=dict(
                    formal_native_steps=3000,
                    B_legal_functional_receive_before_deadline=True,
                    B_first_functional_controlled_receive=200,
                    combined_precision_and_functional_receive=False,
                    strict_clean_foot_capture_regraded=False,
                    whole_body_healthy=True,
                    passer_clean_foot=True,
                    passer_full100NNrecovery=True,
                    B_excluded_contact=False,
                    incoming_arrival_speed_mps=2.5,
                    incoming=dict(
                        A_first_foot_strike_substep=1000,
                        B_first_ball_contact_substep=1800,
                        incoming_target_error_3d_before_B_contact_m=0.2,
                    ),
                ),
            )
        )
    s = dict(
        status=TASK_SOURCE if task else BC_SOURCE,
        source_head="a" * 40,
        source_hash=None,
        independent_whole_world_replay=False,
        command_training_hash=T,
        command_seed=732023,
        source_case_indices=list(INDICES),
        initialization_hash=H,
        training_hash=H,
        Fresh_verified=False,
        promotion_authorized=False,
        hardware_authorized=False,
        goal_complete=False,
        optimizer_updates=0,
        all_actual_native_zero_identity=False,
        records=rows,
        counts=dict(native_steps=46680, actor_calls=23340),
    )
    if task:
        s["command_stage"] = "task-dagger-v2"
    o = deepcopy(s)
    o.update(
        status=TASK_OWN if task else BC_OWN, source_hash=H, independent_whole_world_replay=True
    )
    o["records"] = [{k: v for k, v in r.items() if k != "artifacts"} for r in o["records"]]
    return s, o


@pytest.mark.parametrize("task", [False, True])
def test_exact_independent_pair_and_clock_admitted(task):
    s, o = pair(task)
    assert len(admit_pair(s, o, source_hash=H, training_hash=T, seed=732023, task=task)) == 12


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "status",
        "own",
        "count",
        "seed",
        "train",
        "deadline",
        "steps",
        "teacher",
        "recovery",
        "future",
        "nan",
    ],
)
def test_pair_faults_fail_closed(fault):
    s, o = pair(True)
    if fault == "hash":
        o["source_hash"] = T
    elif fault == "status":
        s["status"] = BC_SOURCE
    elif fault == "own":
        o["records"][0]["sampling_seed"] += 1
    elif fault == "count":
        s["counts"]["actor_calls"] += 1
    elif fault == "seed":
        s["command_seed"] = 732024
    elif fault == "train":
        s["command_training_hash"] = H
    else:
        r = s["records"][0]
        if fault == "deadline":
            r["outcome"]["formal_native_steps"] = 4000
        elif fault == "steps":
            r["native_steps"] = 3000
        elif fault == "teacher":
            r["receiver_command_teacher_actions_executed"] = True
        elif fault == "recovery":
            r["A_recovery"]["neural_rows"] = 0
        elif fault == "future":
            r["outcome"]["incoming"]["A_first_foot_strike_substep"] = -1
        else:
            r["outcome"]["incoming_arrival_speed_mps"] = float("nan")
        o["records"] = [
            {k: v for k, v in r.items() if k != "artifacts"} for r in deepcopy(s["records"])
        ]
    with pytest.raises(ValueError):
        admit_pair(s, o, source_hash=H, training_hash=T, seed=732023, task=True)


def test_strict_means_original_precision_not_regraded_false_marker():
    s, _ = pair()
    parent = s["records"]
    child = deepcopy(parent)
    o = child[0]["outcome"]
    o["incoming"]["incoming_target_error_3d_before_B_contact_m"] = 0.1
    o["combined_precision_and_functional_receive"] = True
    report = compare_records(parent, child)
    assert report["metrics"]["strict"]["rescues"] == [0]
    assert report["local_strict_gain_without_any_measured_loss"] is True
    assert report["goal_complete"] is False
    assert report["promotion_authorized"] is False


def test_gains_cannot_hide_receiving_or_safety_regression():
    s, _ = pair()
    child = deepcopy(s["records"])
    child[0]["outcome"]["incoming"]["incoming_target_error_3d_before_B_contact_m"] = 0.05
    child[0]["outcome"]["combined_precision_and_functional_receive"] = True
    child[2]["outcome"]["B_legal_functional_receive_before_deadline"] = False
    child[4]["outcome"]["whole_body_healthy"] = False
    d = compare_records(s["records"], child)
    assert d["metrics"]["functional"]["regressions"] == [2]
    assert d["metrics"]["healthy"]["regressions"] == [4]
    assert d["local_strict_gain_without_any_measured_loss"] is False


@pytest.mark.parametrize("fault", ["noise", "initial", "course", "precision"])
def test_mismatch_or_relaxed_precision_rejected(fault):
    s, _ = pair()
    child = deepcopy(s["records"])
    if fault == "noise":
        child[0]["sampling_seed"] += 1
    elif fault == "initial":
        child[0]["artifacts"]["initial"]["hash"] = T
    elif fault == "course":
        child[0]["course"]["state_id"] = 25
    else:
        child[0]["outcome"]["combined_precision_and_functional_receive"] = True
    with pytest.raises(ValueError):
        compare_records(s["records"], child)
