"""Predeclared hard safety gates for the intermediate approach pilot."""

from scripts.rsi_collect_approach_gain_pilot_v292 import COURSES
from scripts.rsi_collect_approach_intermediate_gain_v293 import gate


def _rows() -> list[dict]:
    rows = []
    for index, (seed, lane) in enumerate(COURSES):
        base = {
            "high_quality": index >= 6,
            "clean_foot_only": True,
            "maximum_lateral_excursion_m": 2.0,
            "minimum_pelvis_z_m": 0.7,
        }
        candidate = dict(base)
        if index < 3:
            candidate["high_quality"] = True
        rows.append({"seed": seed, "lane": lane, "arms": {"gain_08": base, "gain_10": candidate}})
    return rows


def test_three_rescues_with_foot_and_protection_pass() -> None:
    result = gate(_rows(), [])
    assert result["previous_failures_rescued"] == 3
    assert result["development_gate_passed"] is True


def test_nonfoot_regression_fails_even_with_rescues() -> None:
    rows = _rows()
    rows[0]["arms"]["gain_10"]["clean_foot_only"] = False
    assert gate(rows, [])["development_gate_passed"] is False


def test_missing_case_and_out_of_play_fail_closed() -> None:
    rows = _rows()
    assert gate(rows[:-1], [])["development_gate_passed"] is False
    rows[1]["arms"]["gain_10"]["maximum_lateral_excursion_m"] = 4.1
    assert gate(rows, [])["development_gate_passed"] is False
