"""Fixed zero-offset six-option protocol; not new actor or promotion authority."""

import math
from typing import Any

ORIGINAL_NINE_ZERO_C_INDICES = (0, 2, 4, 6, 7, 8)


def zero_c_six_options() -> tuple[tuple[int, tuple[float, float, float], str], ...]:
    return tuple((law, (0.0, 0.0, 0.0), side) for side in ("right", "left") for law in (0, 3, 4))


def choose_zero_c_six(rows: list[dict[str, Any]]) -> int:
    if len(rows) != 6:
        raise ValueError("all fixed zero-C six options required")
    for index, (row, (law, offset, side)) in enumerate(
        zip(rows, zero_c_six_options(), strict=True)
    ):
        if (
            type(row["index"]) is not int
            or type(row["law_index"]) is not int
            or row["index"] != index
            or row["law_index"] != law
            or row["offset_root_m"] != list(offset)
            or row["contact_reference_foot"] != side
        ):
            raise ValueError("ordered three-law/two-reference zero-C protocol required")
        score = row["score"]
        if type(score["qualified"]) is not bool or (
            score["qualified"]
            and (type(score["cost"]) not in (float, int) or not math.isfinite(score["cost"]))
        ):
            raise ValueError("typed original task gate and finite qualified cost required")
    valid = [i for i, r in enumerate(rows) if r["score"]["qualified"]]
    return min(valid, key=lambda i: (rows[i]["score"]["cost"], i)) if valid else 0


def matched_old_six_selection(row: dict[str, Any]) -> dict[str, int]:
    """Fail if reducing old9 to zero-C6 changes the selected actual endpoint."""
    if row["status"] != "STUDENT30_PRIVILEGED_RECOVERY_EXECUTED" or len(row["candidates"]) != 9:
        raise ValueError("complete original nine-option actual recovery case required")
    subset = [
        dict(row["candidates"][old], index=i) for i, old in enumerate(ORIGINAL_NINE_ZERO_C_INDICES)
    ]
    selected = choose_zero_c_six(subset)
    original = ORIGINAL_NINE_ZERO_C_INDICES[selected]
    if original != row["selected"]:
        raise ValueError("old six-option baseline differs from recorded selected actual endpoint")
    return dict(six_index=selected, original_nine_index=original)
