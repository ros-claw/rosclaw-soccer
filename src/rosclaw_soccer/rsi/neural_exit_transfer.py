"""Frozen source-disjoint SIM exit transfer, not a new action or promotion gate."""

from typing import Any

EXIT_SOURCE_INDICES = (0, 1, 2, 30)
EXIT_TRANSFER_INDICES = (3, 4, 5, 6, 10, 11, 12, 13, 20, 21, 22, 23, 31, 32, 33, 34)
EXIT_RETENTION_INDICES = (0, 30)
EXIT_TRANSFER_WITH_CONTROLS = tuple(sorted(EXIT_TRANSFER_INDICES + EXIT_RETENTION_INDICES))


def require_protected_exit_source(
    audit: dict[str, Any], producer: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    """Caller must authenticate all artifact hashes before allocating physics."""
    if (
        type(audit) is not dict
        or type(producer) is not dict
        or type(rows) is not list
        or type(audit.get("results")) is not list
        or any(type(r) is not dict for r in audit["results"] + rows)
    ):
        raise ValueError("ordinary complete source evidence required")
    for report in (audit, producer):
        if any(
            report.get(k) is not False
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        ):
            raise ValueError("source is unpromoted SIM research only")
    if (
        audit.get("status") != "NEURAL_EXIT_RAW_HISTORY_AND_CPU_NN_VERIFIED_NOT_TEACHER"
        or audit.get("rescues") != [1]
        or audit.get("regressions") != []
        or audit.get("qualified_for_expansion") is not False
        or audit.get("new_physical_steps") != 0
        or audit.get("neural_optimizer_updates") != 0
        or audit.get("cpu_velocity_nn_readback_calls") != 1440
        or audit.get("actual_native_steps") != 28800
        or [r.get("course_index") for r in audit.get("results", [])] != list(EXIT_SOURCE_INDICES)
        or not all(r.get("C_healthy") is True for r in audit["results"])
        or producer.get("status") != "OBSERVED_NEURAL_EXIT_PILOT_COMPLETE_NOT_TEACHER_QUALIFIED"
        or producer.get("actual_worlds") != 8
        or producer.get("actual_native_steps") != 28800
        or producer.get("neural_optimizer_updates") != 0
        or producer.get("qualified_for_expansion") is not False
        or len(rows) != 4
    ):
        raise ValueError("complete protected source gain, zero loss and body/NN audit required")
    terminal = audit.get("producer_terminal", {})
    if (
        terminal.get("source_head") != producer.get("source_head")
        or terminal.get("manager_completion_verified") is not True
        or terminal.get("producer_restarted") is not False
    ):
        raise ValueError("actual source producer completion binding required")
    for index, row, result in zip(EXIT_SOURCE_INDICES, rows, audit["results"], strict=True):
        if (
            row.get("course_index") != index
            or row.get("arm") != "C"
            or row.get("post_contact_exit") is not True
            or row.get("protect_passing_entry_prediction") is not True
            or row.get("exit_gate_uses_source_course_or_outcome") is not False
            or row.get("outcome") != result.get("C")
            or row.get("healthy_full_horizon") is not True
            or row.get("physical_teacher_qualified") is not False
        ):
            raise ValueError("all four audited actual C rows must use protected current prediction")
        if any(row.get(k) is not False for k in ("hardware_authorized", "promotion_authorized")):
            raise ValueError("SIM research only; no source promotion")
