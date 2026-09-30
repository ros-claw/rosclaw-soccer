"""Candidate risk labels are physical regressions, not merely low rewards."""

from scripts.rsi_evaluate_causal_risk_distillation_v297 import new_harm


def _arm(*, clean: bool = True, excursion: float = 2.0) -> dict:
    return {"clean_foot_only": clean, "maximum_lateral_excursion_m": excursion}


def test_new_nonfoot_or_out_of_play_is_harm() -> None:
    assert new_harm(_arm(), _arm(clean=False)) is True
    assert new_harm(_arm(), _arm(excursion=4.01)) is True
    assert new_harm(_arm(), _arm()) is False


def test_existing_nonfoot_or_out_of_play_not_counted_as_new_harm() -> None:
    assert new_harm(_arm(clean=False), _arm(clean=False)) is False
    assert new_harm(_arm(excursion=4.2), _arm(excursion=4.1)) is False
