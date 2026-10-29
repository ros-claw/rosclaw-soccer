from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.training.airborne_receiving_courses import (
    airborne_receiving_ball_launch,
    airborne_receiving_course_bank,
    disjoint_focal_airborne_validation_bank,
    focal_nearest_airborne_course_bank,
)
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
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


def test_focal_nearest_bank_is_distinct_and_fixed():
    bank = focal_nearest_airborne_course_bank()
    assert len(bank) == 8
    assert [x.seed for x in bank] == list(range(96400, 96408))
    assert preflight_receiving_courses(bank) != preflight_receiving_courses(
        airborne_receiving_course_bank()
    )
    assert {x.position_local_m[0] for x in bank} == {0.60, 0.65}


def test_validation_bank_is_fixed_and_physically_disjoint_from_collection():
    bank = disjoint_focal_airborne_validation_bank()
    collection = focal_nearest_airborne_course_bank()
    assert len(bank) == 8
    assert [x.seed for x in bank] == list(range(96500, 96508))
    assert not {x.seed for x in bank} & {x.seed for x in collection}
    assert not {(x.agent_id, x.position_local_m, x.velocity_local_mps) for x in bank} & {
        (x.agent_id, x.position_local_m, x.velocity_local_mps) for x in collection
    }
    assert preflight_receiving_courses(bank) != preflight_receiving_courses(collection)


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
    with pytest.raises(ValueError, match="explicit airborne"):
        simulate_r0_receiving_course(
            asset_root=Path("nonexistent"),
            reference_policy_path=Path("nonexistent"),
            course=course,
            scenario_id="s199.test",
            airborne_standard_ball=True,
        )


def test_exact_replay_fails_on_missing_changed_or_different_dtype():
    baseline = {"q": np.array([1.0, 2.0]), "tag": np.array(["ball"])}
    assert exact_arrays(baseline, {k: v.copy() for k, v in baseline.items()})
    assert not exact_arrays(baseline, {"q": baseline["q"]})
    assert not exact_arrays(baseline, {"q": np.array([1.0, 2.1]), "tag": baseline["tag"]})
    assert not exact_arrays(
        baseline, {"q": baseline["q"].astype(np.float32), "tag": baseline["tag"]}
    )
