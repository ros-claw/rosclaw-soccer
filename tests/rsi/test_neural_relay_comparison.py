import copy

import pytest

from rosclaw_soccer.rsi.neural_relay_comparison import compare_relay_arms


def records():
    rows = []
    for context in (1, 4, 20, 21):
        for lateral in (-0.55, 0.55):
            for start in (0, 80):
                for memory in ("new_skill_zero", "own_shadow_carry"):
                    for bridge in (0, 5, 15, 30):
                        rows.append(
                            dict(
                                index=len(rows),
                                course=dict(
                                    context=context,
                                    lateral_m=lateral,
                                    arm=dict(
                                        start=start, memory=memory, reference_bridge_controls=bridge
                                    ),
                                ),
                                termination="WORLD_HORIZON_COMPLETE",
                                result=dict(
                                    handoff_executed=True,
                                    outgoing_foot_strike=True,
                                    minimum_return_goal_error_3d_m=0.5,
                                    return_0p1m_and_healthy=False,
                                ),
                            )
                        )
    return rows


def test_paired_changes_cluster_by_initial_condition_not_arm_or_frame():
    rows = records()
    for index in (0, 1, 5, 17):
        rows[index]["result"]["return_0p1m_and_healthy"] = True
    result = compare_relay_arms(rows)
    assert result["total_executions"] == 128
    assert result["distinct_initial_conditions"] == 8
    candidate = result["summaries"][1]
    assert candidate["precise_healthy"] == 3
    assert candidate["precise_net_change"] == 2
    assert candidate["precise_rescues"] == [5, 17]
    assert candidate["initial_conditions_with_any_precise_result"] == 2
    assert candidate["clustered_conditions_improved"] == 2
    assert not result["Fresh_verified"] and not result["goal_complete"]
    # Another horizon can LOSE the original baseline, not just get ignored.
    assert result["summaries"][2]["precise_regressions"] == [2]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_course", "wrong_flag", "nan"])
def test_incomplete_or_invalid_data_never_becomes_positive_summary(fault):
    rows = records()
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif fault == "wrong_course":
        rows[0]["course"]["context"] = 55
    elif fault == "wrong_flag":
        rows[0]["result"]["return_0p1m_and_healthy"] = 1
    else:
        rows[0]["result"]["minimum_return_goal_error_3d_m"] = float("nan")
    with pytest.raises(ValueError):
        compare_relay_arms(rows)


def test_team_entry_factorial_distinguishes_delivery_from_successful_return():
    rows = records()
    flag_map = {0: (False, False), 5: (False, True), 15: (True, False), 30: (True, True)}
    for row in rows:
        arm = row["course"]["arm"]
        assigned, align = flag_map[arm["reference_bridge_controls"]]
        arm.update(
            reference_bridge_controls=0, assigned_receiver_pass=assigned, align_successor=align
        )
        row["result"]["incoming_delivery"] = dict(
            incoming_error_observed=True,
            incoming_target_error_3d_before_B_contact_m=0.04 if assigned else 1.0,
            B_first_ball_contact_substep=100 if align else None,
        )
        row["result"]["return_0p1m_and_healthy"] = assigned and align
    result = compare_relay_arms(rows, team_entry=True)
    base, aligned, delivered, both = result["summaries"]
    assert (
        base["precise_healthy"] == aligned["precise_healthy"] == delivered["precise_healthy"] == 0
    )
    assert delivered["incoming_0p1m_deliveries"] == 32
    assert delivered["B_contact_observed"] == 0
    assert both["precise_healthy"] == 32
    assert both["clustered_conditions_improved"] == 8
    assert not both["incoming_diagnostics_are_received_pass_qualification"]
    assert "reference_bridge_controls" not in both
    assert both["assigned_receiver_pass"] and both["align_successor"]
