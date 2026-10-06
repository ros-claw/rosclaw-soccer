"""Declare complete consumed-course coverage before observing new outcomes.

This is a football curriculum planner, not a physics verifier or promotion gate.
The caller authenticates the source manifest and actual behavior policy. No
success labels are read; failures cannot silently disappear from the new bank.
"""

from typing import Any


def declare_jobs(
    records: list[dict[str, Any]], *, samples_per_context: int, stream: int
) -> list[dict[str, Any]]:
    """Expand all forty original consumed contexts into independent draws.

    Require the complete historical four-draw declaration, including consistent
    course identities, before deriving a new exploration stream. Learning rows,
    replay steps and repeated draws are not additional independent contexts.
    """
    if (
        type(records) is not list
        or len(records) != 160
        or type(samples_per_context) is not int
        or not 4 <= samples_per_context <= 16
        or type(stream) is not int
        or not 0 <= stream < 32
    ):
        raise ValueError("complete forty-context source and bounded new declaration required")
    contexts = []
    used_seeds: set[int] = set()
    for index, row in enumerate(records):
        if type(row) is not dict or any(
            type(row.get(key)) is not int or not low <= row[key] < high
            for key, low, high in (
                ("seed", 0, 2**32),
                ("lane", 0, 16),
                ("baseline_course_index", 0, 52),
                ("sampling_seed", 0, 2**32),
                ("sample_index", 0, 4),
            )
        ):
            raise ValueError("bounded complete source course and draw identities required")
        if row["sample_index"] != index % 4 or row["sampling_seed"] in used_seeds:
            raise ValueError("ordered four distinct source draws per context required")
        used_seeds.add(row["sampling_seed"])
        identity = (row["seed"], row["lane"], row["baseline_course_index"])
        if index % 4 == 0:
            contexts.append(identity)
        elif identity != contexts[-1]:
            raise ValueError("draws of a course must share the original physical identity")
    if len({c[:2] for c in contexts}) != 40 or len({c[2] for c in contexts}) != 40:
        raise ValueError("forty distinct consumed physical courses required")
    jobs: list[dict[str, Any]] = []
    for context_id, (seed, lane, course) in enumerate(contexts):
        for sample_index in range(samples_per_context):
            # Preserve the existing football exploration namespace. The same
            # formula is independently tested against sampling_seed().
            sampling_seed = 202610335 + stream * 20000000 + course * 100 + sample_index
            if sampling_seed in used_seeds:
                raise ValueError("new exploration must not reuse source behavior draws")
            jobs.append(
                dict(
                    group=len(jobs),
                    context_id=context_id,
                    seed=seed,
                    lane=lane,
                    baseline_course_index=course,
                    sample_index=sample_index,
                    sampling_seed=sampling_seed,
                )
            )
    return jobs
