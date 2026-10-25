"""Two data-only partial motor options, not a qualified teacher or neural policy.

Runtime options deliberately have no course ID, outcome, source clock or
successful-state lookup. A separate SIM controller must test them on its own
current measured body. Source provenance belongs to the memory report only.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.aim_goal_command import AimGoalCommand
from rosclaw_soccer.rsi.executed_target_synergy import ExecutedTargetSynergy

MotorOption = AimGoalCommand | ExecutedTargetSynergy
NOMINAL_MEMORY_TRANSFER_INDICES = (0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33)


def require_source_retrieval_report(report: dict[str, Any], memory_report: dict[str, Any]) -> None:
    """Admit a new SIM transfer study, never qualify a teacher from source replay."""
    expected = {s["source_course_index"] for s in memory_report["sources"]}
    if (
        len(expected) != 2
        or report["status"] != "FULL_NN_PARTIAL_MEMORY_RAW_AND_RETRIEVAL_VERIFIED"
        or report["actual_worlds"] != 12
        or report["new_physical_steps"] != 0
        or report["qualified_for_expansion"] is not False
        or report["source_memory_retrieval_reproduced"] is not True
        or report["memory_retrieval_is_not_independent_generalization"] is not True
        or report["regressions"]
        or not expected.issubset(report["rescues"])
        or [r["course_index"] for r in report["results"]] != [0, 2, 8, 22, 27, 30]
        or not all(r["C_healthy"] is True for r in report["results"])
    ):
        raise ValueError(
            "independent exact reproduction of BOTH source options with zero loss required"
        )


def encode_motor_option(option: MotorOption) -> dict[str, Any]:
    if type(option) is AimGoalCommand:
        option.target()
        return dict(kind="aim_command", world_point=list(option.world_point))
    if type(option) is ExecutedTargetSynergy and option.projection_mode == "per_axis":
        option.offsets()
        return dict(
            kind="executed_target_synergy",
            coefficients=[list(r) for r in option.coefficients],
            projection_mode="per_axis",
        )
    raise ValueError("only the two explicit qualified partial motor option types are supported")


def decode_motor_options(rows: Any, basis_hg: np.ndarray) -> tuple[MotorOption, MotorOption]:
    if (
        type(rows) is not list
        or len(rows) != 2
        or any(type(row) is not dict for row in rows)
        or not isinstance(basis_hg, np.ndarray)
        or basis_hg.shape != (6, 29)
        or basis_hg.dtype != np.float64
        or not np.isfinite(basis_hg).all()
    ):
        raise ValueError("exact two data-only options and qualified HG29 rank6 basis required")
    aim, target = rows
    if (
        set(aim) != {"kind", "world_point"}
        or aim["kind"] != "aim_command"
        or set(target) != {"kind", "coefficients", "projection_mode"}
        or target["kind"] != "executed_target_synergy"
        or target["projection_mode"] != "per_axis"
    ):
        raise ValueError(
            "bounded typed memory schema; no source IDs, clocks or outcomes at runtime"
        )
    goal = AimGoalCommand(tuple(aim["world_point"]))
    goal.target()
    coefficients = np.asarray(target["coefficients"])
    if coefficients.shape != (2, 6) or coefficients.dtype.kind != "f":
        raise ValueError("two immutable rank6 coefficient rows required")
    plan = ExecutedTargetSynergy(
        0,  # A placeholder; the caller MUST rebase to the current measured entry.
        tuple(tuple(float(x) for x in r) for r in basis_hg),
        (tuple(float(x) for x in coefficients[0]), tuple(float(x) for x in coefficients[1])),
        "per_axis",
    )
    plan.offsets()
    return goal, plan


def partial_option_from_success(kind: str, row: dict[str, Any]) -> dict[str, Any]:
    """Curate a verified actual success, without promoting its failed source family."""
    if (
        row["outcome"]["task_pass"] is not True
        or row["outcome"]["safe"] is not True
        or row["healthy_full_horizon"] is not True
        or row["task_scoring_goal_world_point"] != [8.0, 0.0, 0.11]
        or len(row["controller_events"]) != 1
    ):
        raise ValueError("complete healthy actual task success under the unchanged exam required")
    event = row["controller_events"][0]
    installation = event["installation"]
    if kind == "aim_command" and event["selected_basis"] == "aim-command":
        if installation["command_world_point"] != row["desired_command_world_point"] or any(
            installation[k]
            for k in (
                "past_history_modified",
                "measured_ball_modified",
                "physical_state_modified",
                "task_scoring_goal_modified",
            )
        ):
            raise ValueError("original measured state and separate desired aim command required")
        # Preserve the original declared double-precision command, rather than
        # turning installed float32 .11 into a new out-of-domain decimal value.
        matches = [w for w in event["candidate_windows"] if w["name"] == event["selected"]]
        if len(matches) != 1 or matches[0]["aim_world_point"] is None:
            raise ValueError("one exact selected and actually installed aim command required")
        command = AimGoalCommand(tuple(matches[0]["aim_world_point"]))
        if command.target().tolist() != row["desired_command_world_point"]:
            raise ValueError("declared memory command must decode to the actual installed point")
        return encode_motor_option(command)
    if kind == "executed_target_synergy" and event["selected_basis"] == "executed-target":
        if (
            installation["projection_mode"] != "per_axis"
            or installation["start_control"] != row["entry_control"]
            or any(
                installation[k]
                for k in ("reference_modified", "neural_weights_modified", "hardware_authorized")
            )
        ):
            raise ValueError("unchanged reference and actually installed per-axis target required")
        return dict(
            kind=kind,
            coefficients=installation["coefficients"],
            projection_mode="per_axis",
        )
    raise ValueError("actual success must use the declared partial motor interface")
