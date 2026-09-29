"""Course-relative update must never reward contaminated contact as a save."""

from rsi_r1_kinematic_ranked_update_v203 import _quality


def _row(*, clean: bool, controlled: bool, distance: float, speed: float) -> dict:
    return {
        "safe": True,
        "fault_agents": [],
        "clean": clean,
        "strict_controlled": controlled,
        "own_nonfoot_frames": [] if clean else [37],
        "tail_maximum_foot_distance_m": distance,
        "tail_maximum_ball_speed_mps": speed,
    }


def test_contact_contamination_is_below_any_clean_reception() -> None:
    contaminated = _row(clean=False, controlled=False, distance=0.1, speed=0.1)
    clean = _row(clean=True, controlled=False, distance=2.0, speed=3.0)
    assert _quality(clean) > _quality(contaminated)


def test_clean_tail_quality_and_strict_success_are_ordered() -> None:
    weak = _row(clean=True, controlled=False, distance=0.7, speed=0.6)
    better = _row(clean=True, controlled=False, distance=0.4, speed=0.3)
    strict = _row(clean=True, controlled=True, distance=0.4, speed=0.3)
    assert _quality(strict) > _quality(better) > _quality(weak)
