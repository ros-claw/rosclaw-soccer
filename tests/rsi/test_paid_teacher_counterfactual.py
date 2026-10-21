from types import SimpleNamespace

import numpy as np
import pytest

from scripts.rsi_audit_paid_teacher import collision_counterfactual


@pytest.mark.parametrize("mode", ["no-robot-ball", "no-foot-ball"])
def test_removes_only_declared_ball_pairs_and_preserves_floor_self_contacts(mode):
    bodies = np.array([0, 31, 11, 12, 1, 11])
    m = SimpleNamespace(
        npair=0,
        ngeom=6,
        geom_bodyid=bodies,
        geom_contype=np.array([1, 1, 1, 1, 1, 0]),
        geom_conaffinity=np.array([1, 1, 1, 1, 1, 0]),
    )
    collision_counterfactual(m, mode=mode, ball=31, feet=[11, 12], robot={1, 11, 12})
    for i in range(6):
        for j in range(i):
            if bodies[i] == bodies[j]:
                continue
            allowed = bool(
                (m.geom_contype[i] & m.geom_conaffinity[j])
                | (m.geom_contype[j] & m.geom_conaffinity[i])
            )
            expected = i != 5 and j != 5
            pair = {int(bodies[i]), int(bodies[j])}
            if 31 in pair and (pair & {11, 12} or (mode == "no-robot-ball" and 1 in pair)):
                expected = False
            assert allowed == expected


def test_unrecognized_original_contact_masks_are_rejected_not_repaired():
    m = SimpleNamespace(
        npair=0,
        ngeom=1,
        geom_bodyid=np.array([1]),
        geom_contype=np.array([3]),
        geom_conaffinity=np.array([1]),
    )
    with pytest.raises(ValueError, match="unexpected"):
        collision_counterfactual(m, mode="no-foot-ball", ball=31, feet=[11, 12], robot={1, 11, 12})


@pytest.mark.parametrize("ball_pair", [False, True])
def test_explicit_floor_pairs_preserved_but_ball_pairs_rejected(ball_pair):
    m = SimpleNamespace(
        npair=1,
        ngeom=3,
        geom_bodyid=np.array([0, 11, 31]),
        geom_contype=np.ones(3, dtype=int),
        geom_conaffinity=np.ones(3, dtype=int),
        pair_geom1=np.array([2 if ball_pair else 0]),
        pair_geom2=np.array([1]),
    )
    original = (m.pair_geom1.copy(), m.pair_geom2.copy())
    if ball_pair:
        with pytest.raises(ValueError, match="explicit pair"):
            collision_counterfactual(m, mode="no-foot-ball", ball=31, feet=[11], robot={11})
        assert np.array_equal(m.geom_contype, np.ones(3))
    else:
        collision_counterfactual(m, mode="no-foot-ball", ball=31, feet=[11], robot={11})
    assert np.array_equal(m.pair_geom1, original[0])
    assert np.array_equal(m.pair_geom2, original[1])
