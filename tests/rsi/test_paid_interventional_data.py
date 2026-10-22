import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.paid_interventional_data import admitted_windows, append_unique


def fixture():
    arms = []
    summary = []
    for name, cut in dict(teacher=0, phase_start=62, middle=80, late=96, student=289).items():
        rows = [
            dict(
                case=i,
                healthy_success=name in ("teacher", "phase_start") or name == "middle" and i == 4,
            )
            for i in (4, 5)
        ]
        arms.append(dict(arm=name, switch_control=cut, repeats=[rows, copy.deepcopy(rows)]))
        summary.append(
            dict(
                arm=name,
                cases=[
                    dict(
                        **row, quality=row["healthy_success"], stand_healthy=True, stand_stable=True
                    )
                    for row in rows
                ],
            )
        )
    config = dict(
        protocol="paid.TRAIN.own_state_teacher_handoff.v1",
        cases=[4, 5],
        repeats=2,
        arms=dict(teacher=0, phase_start=62, middle=80, late=96, student=289),
    )
    common = dict(
        config=config,
        optimizer_updates=0,
        Fresh_rows_read=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    p = dict(
        **common,
        status="PAID_OWN_STATE_TEACHER_HANDOFF_CLOSED_PENDING_AUDIT",
        episodes=20,
        actual_native_steps=77800,
        actual_neural_rows=7780,
        planner_R2_head_rows=5780,
        IK_queries=5780,
        arms=arms,
    )
    a = dict(
        **common,
        status="PAID_OWN_STATE_TEACHER_NN_PLANNER_IK_ALL_NATIVE_AUDIT_PASS",
        source_hash="hash",
        counts=dict(episodes=20, IK=5780),
        own_native_steps=77800,
        own_actor_neural_rows=7780,
        own_R2_planner_head_rows=5780,
        summary=summary,
    )
    return p, a


def test_only_actual_post_deviation_successful_teacher_can_supply_new_rows():
    p, a = fixture()
    w = admitted_windows(p, a, "hash")
    assert [(v["arm"], v["case"], v["switch_control"]) for v in w] == [("middle", 4, 80)]


@pytest.mark.parametrize(
    "fault", ["audit", "hash", "count", "success", "recovery", "authority", "bank"]
)
def test_no_failed_counterfactual_or_unproven_recovery_labels(fault):
    p, a = fixture()
    if fault == "audit":
        a["status"] = "PENDING"
    elif fault == "hash":
        a["source_hash"] = "other"
    elif fault == "count":
        a["counts"]["episodes"] = 19
    elif fault == "success":
        p["arms"][2]["repeats"][0][0]["healthy_success"] = False
    elif fault == "recovery":
        a["summary"][2]["cases"][0]["stand_stable"] = False
    elif fault == "authority":
        p["hardware_authorized"] = True
    else:
        p["config"] = {**p["config"], "cases": [5, 6]}
    with pytest.raises(ValueError):
        admitted_windows(p, a, "hash")


def old():
    return dict(
        features=np.zeros((1, 288), np.float32),
        target=np.zeros((1, 29), np.float32),
        parent_delta=np.zeros((1, 29), np.float32),
        teacher=np.array([False]),
        active=np.array([False]),
        group=np.zeros(1, np.int32),
    )


def test_original_arrays_unchanged_and_new_duplicate_does_not_inflate_data():
    baseline = old()
    x = np.zeros((2, 288), np.float32)
    x[1, 0] = 1
    z = dict(
        features=x, target=np.zeros((2, 29), np.float32), parent_delta=np.zeros((2, 29), np.float32)
    )
    merged, groups = append_unique(baseline, [(z, [80, 81], dict(case=4))])
    for k in baseline:
        assert np.array_equal(merged[k][:1], baseline[k])
    assert len(merged["features"]) == 2 and groups[0]["source_frames"] == [81]
    assert groups[0]["duplicate_rows_skipped"] == 1


def test_identical_state_different_action_is_rejected_not_averaged():
    z = dict(
        features=np.zeros((1, 288), np.float32),
        target=np.ones((1, 29), np.float32) * 0.01,
        parent_delta=np.zeros((1, 29), np.float32),
    )
    with pytest.raises(ValueError, match="conflicting"):
        append_unique(old(), [(z, [63], {})])


@pytest.mark.parametrize("fault", [None, "memory", "planner", "not_applied", "not_executed"])
def test_teacher_window_replays_whole_student_history_not_zero_memory_at_cut(fault):
    from types import SimpleNamespace

    import torch
    from test_paid_successor_features import fixture as history_fixture

    from rosclaw_soccer.rsi.paid_interventional_data import executed_features

    base, raw = history_fixture()
    raw["NN_action"][1:] = 0.1
    raw["NN_obs"][2, 125:154] = raw["NN_action"][1]
    raw.update(
        PLAN_R2_nominal_action=np.zeros((3, 29), np.float32),
        PLAN_teacher_action=raw["NN_action"].copy(),
        PLAN_teacher_applied=np.array([False, True, True]),
        PLAN_context_frame=np.arange(3),
        PLAN_switch_control=np.ones(3, int),
    )
    parent = SimpleNamespace(
        base=base, action_scale=torch.ones(29), residual=lambda x: torch.full((len(x), 29), 0.02)
    )
    if fault == "memory":
        raw["NN_h_before"][1].fill(0)
    elif fault == "planner":
        raw["PLAN_R2_nominal_action"][1, 0] = 0.01
    elif fault == "not_applied":
        raw["PLAN_teacher_applied"][1] = False
    elif fault == "not_executed":
        raw["PLAN_teacher_action"][1, 0] = 0.2
    if fault:
        with pytest.raises(ValueError):
            executed_features(parent, raw, 1)
    else:
        z, frames = executed_features(parent, raw, 1)
        assert frames == [1, 2] and np.allclose(z["features"][:, 160], [0.2, 0.3])
        assert np.allclose(z["target"], 0.08) and np.allclose(z["parent_delta"], 0.02)


def test_dirty_curator_cannot_allocate_models_or_write_data(tmp_path, monkeypatch):
    import sys

    from scripts import rsi_curate_paid_interventions as runner

    args = ["run", "--expected-head", "frozen", "--output", str(tmp_path / "out")]
    for name in ("original", "pilot", "audit"):
        args += ["--" + name + "-root", str(tmp_path / name), "--" + name + "-hash", "hash"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"dirty")
    with pytest.raises(ValueError, match="frozen"):
        runner.main()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("prefix_fault", [False, True])
def test_complete_success_labels_actual_student_prefix_not_counterfactual_teacher(prefix_fault):
    from types import SimpleNamespace

    import torch
    from test_paid_successor_features import fixture as history_fixture

    from rosclaw_soccer.rsi.paid_interventional_data import successful_behavior_features

    base, raw = history_fixture()
    raw["NN_action"][1:] = 0.1
    raw["NN_obs"][2, 125:154] = raw["NN_action"][1]
    raw.update(
        NN_nominal_action=raw["NN_action"].copy(),
        PLAN_R2_nominal_action=np.zeros((3, 29), np.float32),
        PLAN_teacher_action=raw["NN_action"].copy(),
        PLAN_teacher_applied=np.array([False, True, True]),
        PLAN_context_frame=np.arange(3),
        PLAN_switch_control=np.ones(3, int),
    )
    # Never-executed prefix teacher suggestions are not successful labels.
    raw["PLAN_teacher_action"][0] = 99
    parent = SimpleNamespace(
        base=base, action_scale=torch.ones(29), residual=lambda x: torch.full((len(x), 29), 0.02)
    )
    if prefix_fault:
        raw["NN_nominal_action"][0, 0] = 0.1
        with pytest.raises(ValueError, match="actual executed"):
            successful_behavior_features(parent, raw, 1)
    else:
        z, frames = successful_behavior_features(parent, raw, 1)
        assert frames == [0, 1, 2]
        assert np.allclose(z["features"][:, 160], [0.1, 0.2, 0.3])
        assert np.allclose(z["target"][0], -0.02)
        assert np.allclose(z["target"][1:], 0.08)
