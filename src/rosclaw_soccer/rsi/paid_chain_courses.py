"""Explicit TRAIN/DEV course and qualified-prefix contracts for SIM chains."""

from typing import Any


def chain_parent_profile(
    declaration: dict[str, Any],
    exam: dict[str, Any],
    candidate: dict[str, Any],
    course_bank: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[int, dict[str, Any]]]]]:
    """DEV has only a qualified FIRST action, never invent a second proof.

    These are predeclared evaluation banks, not runtime policy routing. DEV is
    already consumed development evidence, NOT a sealed Fresh evaluation.
    """
    if course_bank not in ("train", "dev"):
        raise ValueError("explicit train or dev course bank required")
    cases = declaration["old_cases" if course_bank == "train" else "DEV_cases"]
    if not isinstance(cases, list) or len(cases) != 16:
        raise ValueError("all sixteen unchanged declared conditions required")
    if course_bank == "train":
        sources = {
            "baseline": exam["baseline_continuous"],
            "learned": candidate["continuous"]["results"],
        }
        prefixes = {
            arm: [{0: row["first"], 1: row["second"]} for row in rows]
            for arm, rows in sources.items()
        }
    else:
        sources = {
            "baseline": exam["baseline_isolated"]["dev16"],
            "learned": candidate["isolated"]["dev16"]["results"],
        }
        prefixes = {arm: [{0: row} for row in rows] for arm, rows in sources.items()}
    for rows in prefixes.values():
        if len(rows) != 16:
            raise ValueError("complete qualified prefix coverage required")
        for case, prefix in zip(cases, rows, strict=True):
            for row in prefix.values():
                if (
                    not isinstance(row, dict)
                    or row.get("source") != case.get("source")
                    or not isinstance(row.get("raw_path"), str)
                    or not isinstance(row.get("raw_hash"), str)
                ):
                    raise ValueError("exact source-bound parent action proof required")
    return cases, prefixes
