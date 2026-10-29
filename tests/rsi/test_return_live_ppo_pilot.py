import numpy as np
import pytest

from scripts.rsi_return_live_ppo_pilot import SCOPE, eligible_mask


def test_eligibility_uses_only_complete_scoped_admitted_windows():
    ids = ("blue.playmaker", "red.finisher", "red.goalkeeper")
    segment = dict(
        residual_active=np.ones((160, 3), dtype=bool),
        residual_exploration_mask=np.ones((160, 3), dtype=bool),
    )
    trials = [
        dict(agent_id=SCOPE[0], censored=False, frames=100, start_frame=4),
        dict(agent_id=SCOPE[1], censored=True, frames=80, start_frame=40),
        dict(agent_id=ids[2], censored=False, frames=100, start_frame=10),
    ]
    mask = eligible_mask(segment, ids, trials)
    assert mask.sum(axis=0).tolist() == [100, 0, 0]
    assert not mask[:4].any() and not mask[104:].any()


def test_rejects_wrong_masks_and_out_of_bounds_complete_trial():
    segment = dict(
        residual_active=np.ones((120, 2), dtype=bool),
        residual_exploration_mask=np.ones((120, 2), dtype=bool),
    )
    with pytest.raises(ValueError, match="bounds"):
        eligible_mask(
            segment, SCOPE, [dict(agent_id=SCOPE[0], censored=False, frames=100, start_frame=21)]
        )
    segment["residual_active"] = np.ones((120, 2), dtype=int)
    with pytest.raises(ValueError, match="masks"):
        eligible_mask(segment, SCOPE, [])
