"""Sampler/statistic unit tests, not physical pass or Fresh qualification."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.pass_fresh_curriculum import CONTEXTS, pass_fresh_courses, pass_fresh_screen


def templates():
    return [
        dict(
            state_id=i,
            context=CONTEXTS[i // 10],
            task=t,
            assisted=False,
            foot="right" if i < 10 else "left",
            center=105,
            target=[4.0, float(t) / 10, 0.11],
        )
        for i in range(40)
        for t in (0, 1)
    ]


def courses(seed=42):
    return pass_fresh_courses(templates(), {i: np.array([1.0, 0.0]) for i in CONTEXTS}, seed)


def test_balanced_new_states_are_outside_consumed_lateral_line_without_expanding_domain():
    result = courses()
    assert len(result) == 96 and result == courses() and result != courses(43)
    for ctx in CONTEXTS:
        states = [c for c in result if c["context"] == ctx and c["task"] == 0]
        assert len(states) == 12
        assert sum(c["side"] > 0 for c in states) == 6
        xy = np.array([c["ball_planar_offset_world"] for c in states])
        assert (np.abs(xy[:, 0]) >= 0.005).all()
        assert (np.linalg.norm(xy, axis=1) < 0.05).all()
        assert len({tuple(c["ball_planar_offset_world"]) for c in states}) == 12
    assert all(c["assisted"] is False and not any(sum(c["latent"], [])) for c in result)
    assert all(
        result[i]["ball_planar_offset_world"] == result[i + 1]["ball_planar_offset_world"]
        and result[i]["side"] == result[i + 1]["side"]
        for i in range(0, 96, 2)
    )


@pytest.mark.parametrize("fault", ["seed", "missing", "assisted", "direction", "goal"])
def test_bad_complete_declaration_is_rejected_not_clipped_or_replaced(fault):
    rows = templates()
    dirs = {i: np.array([1.0, 0.0]) for i in CONTEXTS}
    seed = 42
    if fault == "seed":
        seed = True
    elif fault == "missing":
        rows.pop()
    elif fault == "assisted":
        rows[0]["assisted"] = True
    elif fault == "direction":
        dirs[1] = np.array([2.0, 0.0])
    elif fault == "goal":
        rows[0]["target"][0] = 9
    with pytest.raises(ValueError):
        pass_fresh_courses(rows, dirs, seed)


def outcomes(good):
    return dict(
        whole_body_healthy=True,
        passer_full100NNrecovery=True,
        passer_clean_foot=True,
        B_legal_functional_receive_before_deadline=good,
        B_excluded_contact=False,
        incoming=dict(
            A_first_foot_strike_substep=100, B_first_ball_contact_substep=200 if good else None
        ),
    )


def rows(good):
    return [dict(course=c, outcome=outcomes(good(i))) for i, c in enumerate(courses())]


def test_screen_counts_initial_states_not_frames_or_goal_tasks_as_independent_samples():
    r = pass_fresh_screen(rows(lambda i: False), rows(lambda i: i % 3 == 0))
    assert r["candidate_controlled"] == 32 and r["baseline_controlled"] == 0
    assert r["local_pass_screen_passed"] and r["independent_initial_states"] == 48
    assert r["goal_trials"] == 96 and r["paired_state_stratified_bootstrap95"][0] > 0
    assert not r["promotion_authorized"] and not r["zero_global_forgetting_proven"]


@pytest.mark.parametrize("fault", ["fall", "nonfoot", "Barm", "inputs", "duplicate", "missing"])
def test_new_hard_safety_failures_or_incomplete_pairing_do_not_hide_behind_net_gain(fault):
    a, b = rows(lambda i: False), rows(lambda i: i % 3 == 0)
    if fault == "fall":
        b[0]["outcome"]["whole_body_healthy"] = False
    elif fault == "nonfoot":
        b[0]["outcome"]["passer_clean_foot"] = False
    elif fault == "Barm":
        b[0]["outcome"]["B_excluded_contact"] = True
    elif fault == "inputs":
        b[0]["course"]["side"] *= -1
    elif fault == "duplicate":
        b[1] = copy.deepcopy(b[0])
    elif fault == "missing":
        b.pop()
    if fault in ("inputs", "duplicate", "missing"):
        with pytest.raises(ValueError):
            pass_fresh_screen(a, b)
    else:
        assert not pass_fresh_screen(a, b)["local_pass_screen_passed"]
