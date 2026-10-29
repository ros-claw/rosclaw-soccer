import numpy as np

from rosclaw_soccer.rsi.sim_target_feasibility import TargetFeasibility
from scripts.rsi_review_omnicontact_native import rebuild_projected_target


def test_independent_readback_matches_causal_filter_over_history():
    limits = np.tile([-0.3, 0.3], (29, 1))
    state = np.zeros(29)
    source = TargetFeasibility(state.copy(), limits)
    rng = np.random.default_rng(726)
    for proposal in rng.normal(size=(100, 29)):
        expected = rebuild_projected_target(proposal, state, limits)
        actual = source.propose(proposal)
        np.testing.assert_array_equal(actual, expected)
        source.commit(actual)
        state = expected
