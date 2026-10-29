from rosclaw_soccer.training.airborne_receiving_courses import (
    airborne_receiving_course_bank,
    disjoint_focal_airborne_validation_bank,
    dissipation_focal_airborne_validation_bank,
    focal_nearest_airborne_course_bank,
)
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses


def test_dissipation_validation_bank_is_bounded_and_disjoint_from_prior_banks():
    bank = dissipation_focal_airborne_validation_bank()
    assert len(bank) == 8
    assert [c.seed for c in bank] == list(range(96600, 96608))
    assert preflight_receiving_courses(bank).startswith("sha256:")
    old = (
        airborne_receiving_course_bank()
        + focal_nearest_airborne_course_bank()
        + disjoint_focal_airborne_validation_bank()
    )
    assert {c.seed for c in bank}.isdisjoint({c.seed for c in old})
    assert {(c.agent_id, c.position_local_m, c.velocity_local_mps) for c in bank}.isdisjoint(
        {(c.agent_id, c.position_local_m, c.velocity_local_mps) for c in old}
    )
