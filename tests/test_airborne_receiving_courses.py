from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.training.airborne_receiving_courses import (
    airborne_receiving_ball_launch,
    airborne_receiving_course_bank,
)
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse, receiving_ball_launch
from scripts.rsi_airborne_receiving_baseline import exact_arrays


def test_fixed_bank_is_unique_preflighted_and_mirrors_team_frame():
    bank = airborne_receiving_course_bank()
    assert len(bank) == 8
    assert [x.seed for x in bank] == list(range(96300, 96308))
    assert preflight_receiving_courses(bank).startswith("sha256:")
    a, b = bank[0], bank[4]
    pa, va = airborne_receiving_ball_launch(a, origin=(0.0, 0.0, 0.0), radius_m=0.115)
    pb, vb = airborne_receiving_ball_launch(b, origin=(0.0, 0.0, 0.0), radius_m=0.115)
    assert pb == (-pa[0], -pa[1], pa[2])
    assert vb == (-va[0], -va[1], va[2])


def test_invalid_and_duplicate_states_rejected_without_clipping():
    bank = airborne_receiving_course_bank()
    for candidate in (
        replace(bank[0], position_local_m=(1.0, 0.0, 0.1)),
        replace(bank[0], velocity_local_mps=(float("nan"), 0.5, -1.0)),
        replace(bank[0], position_local_m=(True, 0.0, 0.5)),
        replace(bank[0], seed=True),
    ):
        with pytest.raises(ValueError):
            preflight_receiving_courses((candidate,))
    with pytest.raises(ValueError, match="duplicate"):
        preflight_receiving_courses((bank[0], replace(bank[0], seed=999)))


def test_legacy_ground_ball_contract_is_unchanged():
    course = ReceivingCourse("blue.defender", 1, 1.25, 0.08)
    assert receiving_ball_launch(course, origin=(0.0, 0.0, 0.0), radius_m=0.115) == (
        (-0.9, -0.08, 0.115),
        (1.25, 0.0, 0.0),
    )
    with pytest.raises(ValueError):
        preflight_receiving_courses((course, airborne_receiving_course_bank()[0]))


def test_exact_replay_fails_on_missing_changed_or_different_dtype():
    baseline = {"q": np.array([1.0, 2.0]), "tag": np.array(["ball"])}
    assert exact_arrays(baseline, {k: v.copy() for k, v in baseline.items()})
    assert not exact_arrays(baseline, {"q": baseline["q"]})
    assert not exact_arrays(baseline, {"q": np.array([1.0, 2.1]), "tag": baseline["tag"]})
    assert not exact_arrays(
        baseline, {"q": baseline["q"].astype(np.float32), "tag": baseline["tag"]}
    )
