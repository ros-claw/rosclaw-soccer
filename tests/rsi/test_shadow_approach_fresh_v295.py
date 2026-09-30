"""Frozen fresh-course selection and fail-closed paired safety gate."""

from scripts.rsi_collect_shadow_approach_fresh_v295 import (
    CHALLENGE,
    COURSES,
    PROTECTION,
    preflight,
    score,
)


def _rows() -> list[dict]:
    rows = []
    for index, (seed, lane) in enumerate(COURSES):
        base = {
            "high_quality": False,
            "clean_foot_only": True,
            "maximum_lateral_excursion_m": 2.0,
            "minimum_pelvis_z_m": 0.7,
            "reward": 0.0,
            "contact_body_indices": [0],
            "first_contact_frame": 70,
            "forward_60_m": 2.0,
            "lateral_60_m": 0.0,
            "command_active_frames": 0,
        }
        candidate = dict(base)
        if index < 2:
            candidate["high_quality"] = True
            candidate["reward"] = 2.0
        rows.append(
            {
                "seed": seed,
                "lane": lane,
                "selected_gain": 1.2,
                "selected_arm": "gain_12",
                "arms": {"gain_08": base, "gain_12": candidate},
            }
        )
    return rows


def test_fresh_catalog_and_scoring_gate() -> None:
    assert preflight().startswith("sha256:")
    assert len(CHALLENGE) == 17
    assert len(PROTECTION) == 4
    assert score(_rows(), [])["fresh_gate_passed"] is True


def test_nonfoot_and_new_out_fail_even_with_quality_gain() -> None:
    rows = _rows()
    rows[0]["arms"]["gain_12"]["clean_foot_only"] = False
    assert score(rows, [])["fresh_gate_passed"] is False
    rows = _rows()
    rows[0]["arms"]["gain_12"]["maximum_lateral_excursion_m"] = 4.1
    assert score(rows, [])["fresh_gate_passed"] is False


def test_missing_course_and_positive_control_drift_fail_closed() -> None:
    rows = _rows()
    assert score(rows[:-1], [])["fresh_gate_passed"] is False
    rows = _rows()
    rows[-1]["arms"]["gain_12"]["command_active_frames"] = 1
    assert score(rows, [])["fresh_gate_passed"] is False
