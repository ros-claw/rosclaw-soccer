"""Synthetic qualification accounting, never physical-pilot evidence."""

from copy import deepcopy

import pytest

from scripts.rsi_review_adapter_continuation_pilot import screen_rows


def fixture(successes=8, unsafe=0):
    bank, rows = [], []
    for i in range(20):
        course = dict(
            context=20,
            state_id=20 + i // 2,
            task=i % 2,
            side=0.55,
            foot="left",
            center=103,
            target=[0, 1, 0.11],
            ball_planar_offset_world=[0, 0],
            assisted=False,
        )
        old = dict(
            whole_body_healthy=True,
            passer_full100NNrecovery=True,
            passer_clean_foot=True,
            B_legal_functional_receive_before_deadline=False,
            B_excluded_contact=False,
            incoming=dict(A_first_foot_strike_substep=100, B_first_ball_contact_substep=200),
        )
        bank.append(dict(role="new_unassisted", source_course=course, measured_outcome=old))
        for prefix in (0, 80):
            o = deepcopy(old)
            o["B_legal_functional_receive_before_deadline"] = i < successes
            o["passer_clean_foot"] = i < 20 - unsafe
            rows.append(
                dict(
                    index=len(rows),
                    course=course | dict(assisted=True),
                    outcome=o,
                    student_prefix_controls=prefix,
                    actual_continuation_controlled_pass=i < successes and i < 20 - unsafe,
                )
            )
    limits = dict(
        prefix80_controlled_pass_at_least=8, prefix80_new_nonfoot_vs_original_baseline_at_most=2
    )
    return rows, bank, limits


@pytest.mark.parametrize(
    "successes,unsafe,passed", [(8, 2, True), (7, 0, False), (8, 3, False), (10, 0, True)]
)
def test_frozen_joint_screen_requires_both_gain_and_contact_hygiene(successes, unsafe, passed):
    rows, bank, limits = fixture(successes, unsafe)
    result = screen_rows(rows, bank, limits)
    assert result["pilot_research_screen_passed"] is passed
    assert result["physically_qualified_corrections"] == successes
    assert result["negative_prefix80_trajectories_preserved"] == 20 - successes
    assert result["prefix80"]["new_nonfoot"] == unsafe
    assert result["fit_eligible_source_indices"] == list(range(1, 2 * successes, 2))


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "different_target", "false_label", "unassisted"]
)
def test_incomplete_mismatched_or_fabricated_qualification_is_rejected(fault):
    rows, bank, limits = fixture()
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[1] = deepcopy(rows[0])
    elif fault == "different_target":
        rows[0]["course"]["target"] = [99, 0, 0]
    elif fault == "false_label":
        rows[0]["actual_continuation_controlled_pass"] = False
    else:
        rows[0]["course"]["assisted"] = False
    with pytest.raises(ValueError):
        screen_rows(rows, bank, limits)


def test_existing_dirty_baseline_is_not_miscounted_as_a_new_contact_regression():
    rows, bank, limits = fixture(8, 3)
    bank[-1]["measured_outcome"]["passer_clean_foot"] = False
    result = screen_rows(rows, bank, limits)
    assert result["prefix80"]["clean"] == 17
    assert result["prefix80"]["new_nonfoot"] == 2
    assert result["pilot_research_screen_passed"] is True
