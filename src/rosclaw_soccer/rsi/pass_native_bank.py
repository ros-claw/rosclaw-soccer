"""Resolve unchanged physical exam fixtures separately from new learning data."""

from collections.abc import Callable
from typing import Any


def original_pass_bank(
    data: dict[str, Any], kind: str, resolve: Callable[[str], dict[str, Any]]
) -> dict[str, Any]:
    if kind == "consolidated-refinement-v1":
        if (
            data["status"] != "QUALIFIED_RESCUE_REFINEMENT_BANK_READY_PENDING_OWN_FEATURE_AUDIT"
            or len(data["records"]) != data["fit_plan"]["fit_histories"]
        ):
            raise ValueError("exact new refinement bank required, not relabeled bank274")
        retained = resolve(data["retained_bank_hash"])
        if retained["parent_lineage_hash"] != data["parent_lineage_hash"]:
            raise ValueError("unchanged original physical task lineage required")
        return original_pass_bank(retained, "consolidated-adapter-v1", resolve)
    if kind == "team-pass-consolidated-v1":
        original = data
    elif kind in ("team-pass-corrected-v2", "team-pass-selected-v3", "consolidated-adapter-v1"):
        if (
            data["status"] != "TEAM_PASS_ACTUAL_STUDENT_CORRECTION_BANK274_READY_NOT_TRAINED"
            or len(data["records"]) != 274
        ):
            raise ValueError("unchanged legacy correction bank274 required")
        original = resolve(data["base_data_hash"])
    else:
        raise ValueError("explicit pass model type required")
    if (
        original["status"] != "TEAM_PASS_WHOLE_HISTORY_BANK177_READY_NOT_TRAINED"
        or len(original["records"]) != 177
    ):
        raise ValueError("original whole physical TRAIN80 fixture witness required")
    return original
