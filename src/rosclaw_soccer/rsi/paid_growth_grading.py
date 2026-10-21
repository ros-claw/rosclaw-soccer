"""Local football research grading, never canonical or hardware promotion."""


def contact_growth_gates(
    *,
    old_preserved: bool,
    second_quality: int,
    stable: int,
    quality_regressions: list[int],
    stability_regressions: list[int],
    reference_quality: int,
    grown_reference: bool,
) -> tuple[bool, bool]:
    if any(
        type(x) is not int or not 0 <= x <= 16 for x in (second_quality, stable, reference_quality)
    ):
        raise ValueError("complete16-cohort integer counts required")
    if (
        type(old_preserved) is not bool
        or type(grown_reference) is not bool
        or (grown_reference and reference_quality < 12)
    ):
        raise ValueError("explicit qualified grown reference required")
    retained = old_preserved and not quality_regressions and not stability_regressions
    gain = retained and second_quality >= 8 and second_quality > reference_quality
    # Stable replication of a qualified grown12 policy is a qualification,
    # NOT a new 12->14 performance improvement. Keep these claims separate.
    qualified = retained and second_quality >= 12 and stable == 16 and (gain or grown_reference)
    return gain, qualified
