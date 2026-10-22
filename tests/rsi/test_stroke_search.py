from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.stroke_search import (
    checked_latent,
    pair_rank,
    sample_generation,
    update_elites,
)
from scripts.rsi_stroke_search_common import stroke_action


def test_bounded_sampling_is_deterministic_and_records_every_rejected_draw():
    args = (np.zeros((2, 6)), np.full((2, 6), 0.4))
    a, draws = sample_generation(np.random.default_rng(55), *args)
    b, other = sample_generation(np.random.default_rng(55), *args)
    assert draws == other and len(a) == len(b) == 16
    assert sum(x["accepted"] for x in draws) == 16
    assert all(abs(z).max() <= 1 for z in a)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_elites_keep_baseline_ties_and_never_collapse_spread():
    points = [np.zeros((2, 6))] * 17
    mean, std, elite = update_elites(points, [(1, 0, 0, 0.0)] * 17)
    assert elite == [0, 1, 2, 3] and not mean.any()
    assert (std == 0.1).all()
    with pytest.raises(ValueError):
        update_elites(points[:-1], [(1, 0, 0, 0.0)] * 16)
    with pytest.raises(ValueError):
        checked_latent(np.ones((2, 6)) * 1.01)


def test_joint_target_proposal_mapping_and_total_authority_include_existing_mean():
    order = np.roll(np.arange(29), 5)
    scale = np.linspace(0.075, 0.55, 29).astype(np.float32)
    ep = SimpleNamespace(frame=104, order=order, scale=scale)
    action, delta, clipped = stroke_action(
        ep, np.zeros(29, np.float32), np.zeros(29, np.float32), 104, np.ones((2, 6))
    )
    np.testing.assert_allclose(action[order] * scale, delta, atol=2e-8)
    assert not clipped
    policy_scale = scale[np.argsort(order)]
    mean = (np.full(29, 0.195) / policy_scale).astype(np.float32)
    action, delta, clipped = stroke_action(ep, mean, np.zeros(29, np.float32), 104, np.ones((2, 6)))
    assert clipped and (abs(action * policy_scale) <= 0.200002).all()
    unchanged = np.ones(29, bool)
    unchanged[order[delta != 0]] = False
    np.testing.assert_array_equal(action[unchanged], mean[unchanged])
    ep.frame = 79
    action, _, _ = stroke_action(ep, mean, np.zeros(29, np.float32), 104, np.ones((2, 6)))
    np.testing.assert_array_equal(action, mean)


def test_rank_never_rewards_a_fall_over_a_healthy_failed_pair():
    def row(i, failed):
        return dict(
            context=i,
            healthy_quality=False,
            result=dict(
                body_failed=failed,
                forbidden=True,
                nonfoot_impulse_Ns=0.1,
                postcontact=dict(
                    correct_foot_first=True,
                    post_contact_quality_observed=True,
                    goal_progress_1p2s_m=3.0,
                    goal_lateral_1p2s_m=0.2,
                    maximum_pelvis_retreat_after_contact_m=0.0,
                ),
            ),
        )

    assert pair_rank([row(4, False), row(20, False)]) > pair_rank([row(4, True), row(20, False)])
    with pytest.raises(ValueError):
        pair_rank([row(20, False), row(4, False)])
