import pytest

from rosclaw_soccer.rsi.paid_growth_grading import contact_growth_gates


def grade(quality, reference, **kwargs):
    return contact_growth_gates(
        old_preserved=kwargs.get("old", True),
        second_quality=quality,
        stable=kwargs.get("stable", 16),
        quality_regressions=kwargs.get("regressions", []),
        stability_regressions=[],
        reference_quality=reference,
        grown_reference=reference >= 12,
    )


def test_stable_reproduction_of_grown_skills_is_not_new_score_growth():
    assert grade(12, 12) == (False, True)
    assert grade(13, 12) == (True, True)
    assert grade(12, 8) == (True, True)
    assert grade(8, 8) == (False, False)
    assert grade(9, 8) == (True, False)


def test_new_contacts_do_not_offset_lost_prior_skills_or_body_stability():
    assert grade(13, 12, regressions=[5]) == (False, False)
    assert grade(12, 12, old=False) == (False, False)
    assert grade(13, 12, stable=15) == (True, False)


@pytest.mark.parametrize("quality", [-1, 17, True, float("nan")])
def test_noncohort_counts_rejected(quality):
    with pytest.raises(ValueError):
        grade(quality, 12)
