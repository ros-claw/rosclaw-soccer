"""Fail-closed evidence gates for a finite, SIM-only standing campaign queue."""

from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.sim.contracts import hash_bytes


def standing_gate(exam_root: Path, audit_root: Path, *, variant: int, bank: str) -> dict[str, Any]:
    for root in (exam_root, audit_root):
        if (root / "failed.json").exists():
            raise ValueError("failed campaign requires diagnosis, never automatic retry")
    ep, ap = exam_root / "complete.json", audit_root / "complete.json"
    e, a = load_json_artifact(ep), load_json_artifact(ap)
    h = hash_bytes(ep.read_bytes())
    if (
        e.get("status") != "PAID_NEURAL_STANDING_LIFECYCLE_CLOSED_PENDING_AUDIT"
        or a.get("status") != "PAID_NEURAL_STANDING_LIFECYCLE_NATIVE_AND_NEURAL_AUDIT_PASS"
        or a.get("input_file_hashes", {}).get(str(ep)) != h
        or e.get("variant") != variant
        or e.get("course_bank") != bank
        or e.get("skills") != 3
        or len(e.get("worlds", [])) != 16
        or a.get("summary") != e.get("summary")
        or a.get("counts", {}).get("worlds") != 32
        or a.get("counts", {}).get("native") != e.get("actual_native_steps")
        or a.get("counts", {}).get("neural") != e.get("actual_neural_rows")
        or a.get("counts", {}).get("own_neural") != e.get("actual_neural_rows")
        or a.get("own_native_steps") != 0
        or any(
            obj.get(key) is not False
            for obj in (e, a)
            for key in (
                "hardware_authorized",
                "promotion_authorized",
                "old_exam_result",
                "team_game",
            )
        )
        or e.get("optimizer_updates") != 0
        or a.get("optimizer_updates") != 0
    ):
        raise ValueError("exact closed all16 independently audited standing campaign required")
    return dict(
        exam_hash=h,
        audit_hash=hash_bytes(ap.read_bytes()),
        summary=e["summary"],
        variant=variant,
        bank=bank,
    )
