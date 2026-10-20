"""Explicit research receipt boundary; never alias old ball-aware certificates."""

from typing import Any


def require_foot_only_learning(report: dict[str, Any]) -> None:
    warmup = report.get("status") == "FOOT_ONLY_VALUE_WARMUP_PPO_NOT_SKILL_QUALIFIED"
    if report.get("status") not in (
        "FOOT_ONLY_VALUE_WARMUP_PPO_NOT_SKILL_QUALIFIED",
        "FOOT_ONLY_PPO_PILOT_NOT_SKILL_QUALIFIED",
    ):
        raise ValueError("explicit new foot-only research learner required")
    expected = {
        "gpu_world_steps": 245760 if warmup else 122880,
        "iterations": 32 if warmup else 16,
        "worlds": 32,
        "seed": 4601,
        "new_native_steps": 0,
    }
    if any(type(report.get(k)) is not int or report[k] != v for k, v in expected.items()):
        raise ValueError("complete predeclared foot-only learning budget required")
    updates = 656 if warmup else 336
    if report.get("optimizer_counts") != dict(attempted=updates, completed=updates):
        raise ValueError("complete counted foot-only optimizer updates required")
    for key in ("hardware_authorized", "promotion_authorized", "goal_complete"):
        if report.get(key) is not False:
            raise ValueError("non-promoted SIM research boundary required")
    if warmup:
        boundary = report.get("value_warmup_boundary", {})
        if (
            boundary.get("actor_bitexact") is not True
            or boundary.get("critic_changed") is not True
            or boundary.get("optimizer_updates") != 320
            or boundary.get("physical_substeps") != 3840
        ):
            raise ValueError("actual frozen-actor value recalibration boundary required")


def require_foot_only_development(
    report: dict[str, Any], *, learner_path: str, learner_hash: str
) -> None:
    if (
        report.get("status") != "FOOT_ONLY_PAIRED_PUBLIC_DEV_NOT_SKILL_QUALIFIED"
        or report.get("all_initial_states_paired_exact") is not True
        or report.get("known_public_dev_not_blind_fresh") is not True
        or report.get("optimizer_updates") != 0
        or report.get("hardware_authorized") is not False
        or report.get("input_file_hashes", {}).get(learner_path) != learner_hash
    ):
        raise ValueError("same foot-only candidate's complete paired public DEV required")
    rows = report.get("results", [])
    if (
        len(rows) != 4
        or {(r.get("source"), r.get("arm")) for r in rows}
        != {
            (f"soccer-standard-{s}", arm)
            for s in ("007_left.npz", "010_right.npz")
            for arm in ("initial", "final")
        }
        or any(r.get("completed_count") != 32 for r in rows)
    ):
        raise ValueError("both public sources and both complete paired arms required")
