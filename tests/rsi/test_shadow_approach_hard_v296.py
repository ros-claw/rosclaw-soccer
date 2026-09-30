"""The high-speed negative-side course and selected safety gate remain fixed."""

from scripts.rsi_collect_shadow_approach_hard_v296 import COURSES, preflight, score


def _rows() -> list[dict]:
    rows = []
    for index, (seed, lane) in enumerate(COURSES):
        baseline = {
            "high_quality": False,
            "clean_foot_only": True,
            "maximum_lateral_excursion_m": 3.0,
            "minimum_pelvis_z_m": 0.7,
            "reward": 0.0,
        }
        aggressive = dict(baseline)
        if index < 3:
            aggressive["high_quality"] = True
            aggressive["reward"] = 2.0
        rows.append(
            {
                "seed": seed,
                "lane": lane,
                "selected_gain": 1.2,
                "selected_arm": "gain_12",
                "arms": {"gain_08": baseline, "gain_12": aggressive},
            }
        )
    return rows


def test_hard_catalog_and_three_gain_gate() -> None:
    assert preflight().startswith("sha256:")
    assert len(COURSES) == 15
    assert score(_rows(), [])["hard_fresh_gate_passed"] is True


def test_nonfoot_or_out_of_play_rejects() -> None:
    rows = _rows()
    rows[0]["arms"]["gain_12"]["clean_foot_only"] = False
    assert score(rows, [])["hard_fresh_gate_passed"] is False
    rows = _rows()
    rows[0]["arms"]["gain_12"]["maximum_lateral_excursion_m"] = 4.1
    assert score(rows, [])["hard_fresh_gate_passed"] is False


def test_incomplete_or_duplicate_course_rejects() -> None:
    rows = _rows()
    assert score(rows[:-1], [])["hard_fresh_gate_passed"] is False
    rows[-1]["seed"] = rows[0]["seed"]
    rows[-1]["lane"] = rows[0]["lane"]
    assert score(rows, [])["hard_fresh_gate_passed"] is False
