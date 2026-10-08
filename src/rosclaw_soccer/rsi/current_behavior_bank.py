"""Football-specific complete-bank admission, not physics or promotion proof.

Callers must authenticate whole files and independently reconstruct the actual
behavior density. This structural check never selects successful trajectories.
"""

from typing import Any


def _hash(value: Any) -> bool:
    return (
        type(value) is str
        and value.startswith("sha256:")
        and len(value) == 71
        and all(c in "0123456789abcdef" for c in value[7:])
    )


def validate_current_bank_rows(
    jobs: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    behavior_model_hash: str,
) -> dict[str, Any]:
    """Require forty consumed courses and four current-policy draws each.

    All records, including failed and unsafe draws, are retained. Passing this
    function grants no training, runtime selection, Fresh or promotion rights.
    """
    if (
        type(jobs) is not list
        or type(records) is not list
        or len(jobs) != 160
        or len(records) != 160
        or not _hash(behavior_model_hash)
    ):
        raise ValueError("complete 160-row bank and exact current behavior hash required")
    courses: list[tuple[int, int, int]] = []
    seeds: set[int] = set()
    quality = safe = 0
    for index, (job, row) in enumerate(zip(jobs, records, strict=True)):
        if type(job) is not dict or type(row) is not dict:
            raise ValueError("complete declared job and record mappings required")
        fields = (
            ("group", index, index + 1),
            ("context_id", index // 4, index // 4 + 1),
            ("sample_index", index % 4, index % 4 + 1),
            ("seed", 0, 2**32),
            ("lane", 0, 16),
            ("baseline_course_index", 0, 52),
            ("sampling_seed", 0, 2**32),
        )
        if any(
            type(job.get(key)) is not int
            or not low <= job[key] < high
            or type(row.get(key)) is not int
            or row[key] != job[key]
            for key, low, high in fields
        ):
            raise ValueError("ordered exact course, context and draw identities required")
        course = (job["seed"], job["lane"], job["baseline_course_index"])
        if index % 4 == 0:
            courses.append(course)
        elif course != courses[-1]:
            raise ValueError("all four draws must share one physical course")
        if job["sampling_seed"] in seeds:
            raise ValueError("independent declared draws required")
        seeds.add(job["sampling_seed"])
        if (
            job.get("actual_behavior_model_hash") != behavior_model_hash
            or row.get("actual_behavior_model_hash") != behavior_model_hash
            or row.get("view_hash") != job.get("view_hash")
            or not _hash(job.get("view_hash"))
            or row.get("frame_sample_count") != 270
            or type(row.get("frame_sample_count")) is not int
            or row.get("physical_replay_performed_here") is not True
            or row.get("execution_origin") != "NEW"
            or any(
                row.get(key) is not False
                for key in ("private_fresh_accessed", "promotion_authorized", "hardware_authorized")
            )
        ):
            raise ValueError("actual current behavior and SIM-only execution bindings required")
        outcome = row.get("outcome")
        if type(outcome) is not dict or any(
            type(outcome.get(key)) is not bool for key in ("high_quality", "safety_passed")
        ):
            raise ValueError("explicit measured quality and safety labels required")
        quality += outcome["high_quality"]
        safe += outcome["safety_passed"]
    if len({c[:2] for c in courses}) != 40 or len({c[2] for c in courses}) != 40:
        raise ValueError("forty distinct consumed physical courses required")
    return dict(
        episodes=160,
        contexts=40,
        frame_rows=43200,
        high_quality_episodes=quality,
        safe_episodes=safe,
        all_success_and_failure_rows_retained=True,
        physics_validated_here=False,
        behavior_density_reconstructed_here=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
