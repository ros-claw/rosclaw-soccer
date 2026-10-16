import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification


def fixture(*, minimum=0.8, forward=1.2):
    force = np.zeros((300, 1, 6))
    force[200, 0, 0] = 2
    height = np.full((300, 1, 10), 0.8)
    height[295, 0, 1] = minimum
    ball = np.zeros((300, 1, 3))
    ball[200:261, 0, 0] = np.linspace(0, forward, 61)
    ball[261:, 0, 0] = forward
    outcome = label_first_touch(
        dict(force_n=force, pelvis_z_per_substep_m=height, ball_position_after_step_m=ball)
    )
    review = dict(
        physical_substeps=3000,
        actual_recurrent_sampling_draws_reconstructed=270,
        high_quality=forward >= 1,
        safety_passed=minimum >= 0.65,
        clean_foot_only=True,
        minimum_pelvis_z_m=float(minimum),
    )
    foundation = dict(recomputed_executed_calls=300)
    return review, foundation, outcome


@pytest.mark.parametrize(
    "minimum,forward", [(0.8, 1.2), (0.62, 1.2), (0.8, 0.5), (0.62, 0.5), (0.65, 1.2)]
)
def test_quality_and_safety_are_composed_without_crashing_legal_failure(minimum, forward):
    review, foundation, outcome = fixture(minimum=minimum, forward=forward)
    before = copy.deepcopy((review, foundation))
    alignment = require_first_touch_qualification(review, foundation, outcome)
    assert alignment["m0_task_pass"] is outcome.task_pass
    assert alignment["legacy_cpu_ball_quality"] is (forward >= 1)
    assert not alignment["task_or_safety_contract_changed"]
    assert not alignment["promotion_authorized"]
    assert (review, foundation) == before
    if minimum < 0.65:
        assert outcome.event_status == "UNSAFE_CONTACT" and not outcome.task_pass


@pytest.mark.parametrize(
    "document,key,value",
    [
        (0, "physical_substeps", 2999),
        (0, "physical_substeps", 3000.0),
        (0, "actual_recurrent_sampling_draws_reconstructed", True),
        (1, "recomputed_executed_calls", 299),
        (1, "recomputed_executed_calls", 300.0),
        (0, "high_quality", 1),
        (0, "safety_passed", 1),
        (0, "clean_foot_only", 1),
        (0, "minimum_pelvis_z_m", float("nan")),
        (0, "minimum_pelvis_z_m", float("inf")),
        (0, "minimum_pelvis_z_m", True),
        (0, "minimum_pelvis_z_m", "0.8"),
        (0, "minimum_pelvis_z_m", 0.7),
        (0, "safety_passed", False),
        (0, "high_quality", False),
        (0, "clean_foot_only", False),
    ],
)
def test_incomplete_or_forged_replay_fails_closed(document, key, value):
    review, foundation, outcome = fixture()
    [review, foundation][document][key] = value
    with pytest.raises(ValueError):
        require_first_touch_qualification(review, foundation, outcome)


def test_quality_true_cannot_promote_failed_ball_outcome():
    review, foundation, outcome = fixture(forward=0.5)
    review["high_quality"] = True
    with pytest.raises(ValueError, match="differs from primitive M0"):
        require_first_touch_qualification(review, foundation, outcome)


@pytest.mark.parametrize("field", ["physical_substeps", "safety_passed", "minimum_pelvis_z_m"])
def test_missing_evidence_rejected(field):
    review, foundation, outcome = fixture()
    del review[field]
    with pytest.raises(ValueError):
        require_first_touch_qualification(review, foundation, outcome)


@pytest.mark.parametrize("first", [239, 240, 260, 299, None])
@pytest.mark.parametrize("minimum", [0.62, 0.65, 0.8])
@pytest.mark.parametrize("nonfoot", [False, True])
@pytest.mark.parametrize("ball_kind", ["good", "short", "side", "out"])
def test_composition_matches_actual_legacy_rule_at_task_boundaries(
    first, minimum, nonfoot, ball_kind
):
    from rosclaw_soccer.rsi.independent_first_touch_bank import post_contact_displacement
    from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality

    force = np.zeros((300, 1, 6))
    if first is not None:
        force[first, 0, 0] = 2
    if nonfoot:
        force[100, 0, 4] = 2
    height = np.full((300, 1, 10), 0.8)
    height[295, 0, 1] = minimum
    ball = np.zeros((300, 1, 3))
    if first is not None:
        frames = np.arange(300)
        travel = np.clip((frames - first) / 60, 0, 1)
        ball[:, 0, 0] = travel * (0.5 if ball_kind == "short" else 1.2)
        if ball_kind == "side":
            ball[:, 0, 1] = travel * 0.8
    if ball_kind == "out":
        ball[-1, 0, 1] = 5.0
    trace = dict(force_n=force, pelvis_z_per_substep_m=height, ball_position_after_step_m=ball)
    outcome = label_first_touch(trace)
    any_contact = np.flatnonzero(np.any(force[:, 0] > 1, axis=1))
    legacy_first = int(any_contact[0]) if len(any_contact) else None
    clean = legacy_first is not None and not nonfoot
    row = dict(
        clean_foot_only=clean,
        minimum_pelvis_z_m=minimum,
        maximum_lateral_excursion_m=float(np.abs(ball[:, 0, 1]).max()),
        **post_contact_displacement(ball[:, 0], legacy_first),
    )
    review = dict(
        physical_substeps=3000,
        actual_recurrent_sampling_draws_reconstructed=270,
        high_quality=high_quality(row),
        safety_passed=minimum >= 0.65,
        clean_foot_only=clean,
        minimum_pelvis_z_m=minimum,
    )
    alignment = require_first_touch_qualification(
        review, dict(recomputed_executed_calls=300), outcome
    )
    assert alignment["m0_task_pass"] is outcome.task_pass
