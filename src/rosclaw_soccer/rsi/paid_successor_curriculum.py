"""Explicit TRAIN curriculum versions; never relabel failed trajectories as successes."""

from typing import Any


def curriculum_cases(declaration: dict[str, Any]) -> list[int]:
    cases = declaration.get("teacher_cases")
    protocol = declaration.get("curriculum_protocol", "successor-four-teacher-v1")
    if type(cases) is not list or any(type(i) is not int for i in cases):
        raise ValueError("integer-only declared TRAIN case identities required")
    if cases == [2, 3, 10, 11] and protocol == "successor-four-teacher-v1":
        return [2, 3, 10, 11]
    if cases == [2, 3, 8, 9, 10, 11] and protocol == "successor-six-teacher-v2":
        return [2, 3, 8, 9, 10, 11]
    raise ValueError("exact declared four/six-teacher TRAIN curriculum required")


def validate_train_curriculum(data: dict[str, Any], declaration: dict[str, Any]) -> None:
    cases = curriculum_cases(declaration)
    rows = data["trajectories"]
    if len(rows) != 36 + len(cases):
        raise ValueError("all teachers and all36 protected successes required")
    teacher = [r for r in rows if r["role"] == "teacher"]
    protected = [r for r in rows if r["role"] == "protected_success"]
    if (
        sorted(r["case"] for r in teacher) != cases
        or any(r["skill"] != 2 for r in teacher)
        or [sum(r["skill"] == k for r in protected) for k in range(3)] != [16, 14, 6]
        or any(type(r["case"]) is not int or not 0 <= r["case"] < 16 for r in rows)
        or len({(r["source_path"], r["case"], r["skill"]) for r in rows}) != len(rows)
        or data["protected_successes_by_kick"] != [16, 14, 6]
        or data["bank"] != "train"
        or data["DEV_episodes_fit"] != 0
        or declaration["DEV_episodes_fit"] != 0
    ):
        raise ValueError("genuine TRAIN teacher and protected-success identities required")
    cursor = 0
    for row in rows:
        if (
            type(row["rows"]) is not int
            or not 1 <= row["rows"] <= 300
            or row["start"] != cursor
            or row["stop"] != cursor + row["rows"]
        ):
            raise ValueError("complete unfiltered contiguous feature sequence identity required")
        cursor = row["stop"]
    if (
        cursor != data["rows"]
        or sum(r["rows"] for r in teacher) != data["teacher_rows"]
        or sum(r["rows"] for r in protected) != data["protected_rows"]
    ):
        raise ValueError("actual teacher/protection row accounting differs")


def qualified_additional_teacher(
    report: dict[str, Any],
    review: dict[str, Any],
    declaration: dict[str, Any],
    *,
    report_path: str,
    report_hash: str,
    parent_path: str,
    parent_hash: str,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if (
        report.get("status") != "PAID_SUCCESSOR_CONTACT_TEACHER_CLOSED_PENDING_AUDIT"
        or review.get("status")
        != "PAID_SUCCESSOR_CONTACT_ALL_CANDIDATES_NATIVE_NEURAL_IK_CEM_AUDIT_PASS"
        or review["input_file_hashes"].get(report_path) != report_hash
        or report["input_file_hashes"].get(parent_path) != parent_hash
        or declaration.get("source_cases") != cases
        or declaration.get("pairs") != [[8, 9]]
        or declaration.get("random_seed") != 20261022
        or report.get("cem_distribution_updates") != 2
        or review["counts"].get("branches") != 52
        or review["counts"].get("own_neural") != report["actual_neural_rows"]
        or review.get("own_native_steps") != report["actual_native_steps"]
        or any(
            obj.get(k) is not False
            for obj in (report, review)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("complete seed0 TRAIN005 teacher and all52 independent audit required")
    searches, audits = report["searches"], review["summaries"]
    if (
        len(searches) != 1
        or len(audits) != 1
        or searches[0]["pair"] != [8, 9]
        or searches[0]["selected_rank"][:5] != [1, 1, 1, 2, 2]
        or audits[0]["pair"] != [8, 9]
        or len(audits[0]["selected"]) != 2
        or any(
            not r["quality"] or not r["stand_healthy"] or not r["stand_stable"]
            for r in audits[0]["selected"]
        )
    ):
        raise ValueError("common clean stable2/2 teacher required")
    rows: list[dict[str, Any]] = searches[0]["selected"]
    if [r["case"] for r in rows] != [8, 9] or any(
        r["kick"]["body_failed"]
        or r["kick"]["forbidden"]
        or not r["stand"]["healthy"]
        or not r["stand"]["stability_observed"]
        for r in rows
    ):
        raise ValueError("actual selected2/2 successful kick-and-stand labels required")
    return rows
