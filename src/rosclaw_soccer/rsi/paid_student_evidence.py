"""Admit closed student-world evidence for more verification, never promotion."""

from typing import Any


def qualify_student_worlds(
    report: dict[str, Any],
    declaration: dict[str, Any],
    audit: dict[str, Any],
    *,
    report_path: str,
    report_hash: str,
    declaration_hash: str,
) -> None:
    if (
        report.get("status") != "PAID_SUCCESSOR_NEURAL_STUDENT_LIFECYCLE_CLOSED_PENDING_AUDIT"
        or audit.get("status")
        != "PAID_SUCCESSOR_STUDENT_ALL_NATIVE_PRIMITIVES_AND_OWN_NEURAL_AUDIT_PASS"
        or audit["input_file_hashes"].get(report_path) != report_hash
        or report.get("declaration_hash") != declaration_hash
        or report.get("source_head") != declaration.get("source_head")
        or report.get("bank") != declaration.get("bank")
        or declaration.get("bank") not in ("train", "dev")
        or len(declaration.get("cases", [])) != 16
        or len(report.get("worlds", [])) != 16
        or [w.get("case") for w in report["worlds"]] != list(range(16))
        or declaration.get("skills") != 3
        or declaration.get("repeats") != 2
        or declaration.get("body_resets") != 0
        or declaration.get("standing_controls") != 100
        or report.get("all_first_and_full_repeat_arrays_exact") is not True
        or report.get("summary") != audit.get("summary")
        or audit["counts"].get("worlds") != 32
        or not isinstance(report.get("actual_neural_rows"), int)
        or report["actual_neural_rows"] <= 0
        or audit["counts"].get("native") != report.get("actual_native_steps")
        or audit["counts"].get("neural") != report["actual_neural_rows"]
        or audit["counts"].get("own_neural") != report["actual_neural_rows"]
        or audit.get("own_native_steps") != 0
        or any(
            obj.get(k) is not False
            for obj in (report, declaration, audit)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("complete independently audited student-world evidence required")
