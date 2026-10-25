import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.partial_motor_memory import (
    NOMINAL_MEMORY_TRANSFER_INDICES,
    decode_motor_options,
    encode_motor_option,
    partial_option_from_success,
    require_source_retrieval_report,
)


def rows():
    return [
        dict(kind="aim_command", world_point=[8.0, -2.0, 0.11]),
        dict(
            kind="executed_target_synergy",
            coefficients=[[0.01] * 6, [-0.01] * 6],
            projection_mode="per_axis",
        ),
    ]


def test_runtime_options_are_immutable_data_without_source_id_or_clock():
    source = rows()
    options = decode_motor_options(source, np.eye(6, 29))
    assert [encode_motor_option(o) for o in options] == source
    source[1]["coefficients"][0][0] = 0.05
    assert options[1].coefficients[0][0] == 0.01
    assert options[1].start_control == 0
    assert options[1].projection_mode == "per_axis"


@pytest.mark.parametrize("extra", ["course_index", "outcome", "start_control", "source_hash"])
def test_no_source_metadata_may_become_a_runtime_option(extra):
    data = rows()
    data[1][extra] = 22
    with pytest.raises(ValueError, match="no source IDs"):
        decode_motor_options(data, np.eye(6, 29))


@pytest.mark.parametrize("defect", ["count", "order", "bounds", "nan", "mode"])
def test_invalid_or_expanded_memory_fails_closed(defect):
    data = rows()
    if defect == "count":
        data.append(copy.deepcopy(data[0]))
    elif defect == "order":
        data.reverse()
    elif defect == "bounds":
        data[1]["coefficients"][0][0] = 0.07
    elif defect == "nan":
        data[1]["coefficients"][0][0] = float("nan")
    else:
        data[1]["projection_mode"] = "whole_tick"
    with pytest.raises(ValueError):
        decode_motor_options(data, np.eye(6, 29))


def test_partial_success_cannot_be_curated_from_failure_or_different_interface():
    row = dict(
        outcome=dict(task_pass=False, safe=True),
        healthy_full_horizon=True,
        task_scoring_goal_world_point=[8.0, 0.0, 0.11],
        controller_events=[],
    )
    with pytest.raises(ValueError, match="actual task success"):
        partial_option_from_success("aim_command", row)


def actual_row(kind):
    goal = [8.0, -2.0, float(np.float32(0.11))]
    installation = (
        dict(
            command_world_point=goal,
            past_history_modified=False,
            measured_ball_modified=False,
            physical_state_modified=False,
            task_scoring_goal_modified=False,
        )
        if kind == "aim_command"
        else dict(
            coefficients=rows()[1]["coefficients"],
            projection_mode="per_axis",
            start_control=85,
            reference_modified=False,
            neural_weights_modified=False,
            hardware_authorized=False,
        )
    )
    return dict(
        outcome=dict(task_pass=True, safe=True),
        healthy_full_horizon=True,
        task_scoring_goal_world_point=[8.0, 0.0, 0.11],
        desired_command_world_point=goal,
        entry_control=85,
        controller_events=[
            dict(
                selected="selected",
                selected_basis="aim-command" if kind == "aim_command" else "executed-target",
                installation=installation,
                candidate_windows=[dict(name="selected", aim_world_point=[8.0, -2.0, 0.11])],
            )
        ],
    )


@pytest.mark.parametrize("kind", ["aim_command", "executed_target_synergy"])
def test_only_declared_actually_installed_success_is_curated(kind):
    source = actual_row(kind)
    before = copy.deepcopy(source)
    option = partial_option_from_success(kind, source)
    assert source == before
    assert option == rows()[0 if kind == "aim_command" else 1]
    if kind == "aim_command":
        assert option["world_point"][2] == 0.11  # Not installed float32 .1099999994.
        source["controller_events"][0]["installation"]["measured_ball_modified"] = True
    else:
        source["controller_events"][0]["installation"]["projection_mode"] = "whole_tick"
    with pytest.raises(ValueError):
        partial_option_from_success(kind, source)


def source_retrieval():
    return dict(
        status="FULL_NN_PARTIAL_MEMORY_RAW_AND_RETRIEVAL_VERIFIED",
        actual_worlds=12,
        new_physical_steps=0,
        qualified_for_expansion=False,
        source_memory_retrieval_reproduced=True,
        memory_retrieval_is_not_independent_generalization=True,
        regressions=[],
        rescues=[22, 27],
        results=[dict(course_index=i, C_healthy=True) for i in (0, 2, 8, 22, 27, 30)],
    )


def test_nominal_transfer_is_fixed_balanced_and_source_replay_does_not_qualify_teacher():
    assert len(NOMINAL_MEMORY_TRANSFER_INDICES) == 16
    assert [sum(i // 10 == k for i in NOMINAL_MEMORY_TRANSFER_INDICES) for k in range(4)] == [4] * 4
    report = source_retrieval()
    require_source_retrieval_report(
        report, dict(sources=[dict(source_course_index=22), dict(source_course_index=27)])
    )
    assert report["qualified_for_expansion"] is False


@pytest.mark.parametrize(
    "defect", ["other_rescues", "promoted", "regression", "partial", "unhealthy"]
)
def test_transfer_needs_actual_reproduction_of_both_known_options_not_just_any_two_wins(defect):
    report = source_retrieval()
    if defect == "other_rescues":
        report["rescues"] = [2, 8]
    elif defect == "promoted":
        report["qualified_for_expansion"] = True
    elif defect == "regression":
        report["regressions"] = [0]
    elif defect == "partial":
        report["results"].pop()
    else:
        report["results"][0]["C_healthy"] = False
    with pytest.raises(ValueError, match="BOTH"):
        require_source_retrieval_report(
            report, dict(sources=[dict(source_course_index=22), dict(source_course_index=27)])
        )
