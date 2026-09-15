import numpy as np
import pytest

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.training.near_ball_imitation import fit_private_motor_window


def case():
    ids = tuple(
        f"{team}.{role}"
        for team in ("blue", "red")
        for role in ("defender", "finisher", "goalkeeper", "playmaker")
    )
    parent = NearBallResidualPolicy.initialize(ids, "sha256:" + "a" * 64).with_task_geometry()
    rng = np.random.default_rng(2084)
    return (
        parent,
        rng.normal(0, 0.1, (20, 58)),
        np.full((20, 12), 0.1),
        rng.normal(0, 0.1, (30, 58)),
    )


def test_imitation_replays_and_freezes_encoders_critics_and_other_players():
    pytest.importorskip("torch")
    parent, obs, raw, anchors = case()
    kw = dict(
        agent_id="blue.playmaker",
        observations=obs,
        raw_actions=raw,
        anchor_observations=anchors,
        steps=16,
    )
    child, record = fit_private_motor_window(parent, **kw)
    repeated, replay = fit_private_motor_window(parent, **kw)
    assert child.policy_hash == repeated.policy_hash and record == replay
    assert record["demonstration_mse_after"] < record["demonstration_mse_before"]
    assert not record["on_policy"] and not record["promotion_eligible"]
    assert child.parent_hash == parent.policy_hash
    for key in parent.weights:
        if key not in ("w2", "b2"):
            np.testing.assert_array_equal(child.weights[key], parent.weights[key])
        else:
            np.testing.assert_array_equal(
                child.weights[key][[0, 1, 2, 4, 5, 6, 7]],
                parent.weights[key][[0, 1, 2, 4, 5, 6, 7]],
            )


@pytest.mark.parametrize("kind", ["empty", "nan", "wrong_width", "mismatch", "foreign", "budget"])
def test_invalid_off_policy_data_is_rejected(kind):
    parent, obs, raw, anchors = case()
    kw = dict(
        agent_id="blue.playmaker", observations=obs, raw_actions=raw, anchor_observations=anchors
    )
    if kind == "empty":
        kw["observations"] = obs[:0]
    if kind == "nan":
        obs[0, 0] = np.nan
    if kind == "wrong_width":
        kw["observations"] = obs[:, :-1]
    if kind == "mismatch":
        kw["raw_actions"] = raw[:-1]
    if kind == "foreign":
        kw["agent_id"] = "outsider"
    if kind == "budget":
        kw["steps"] = True
    with pytest.raises(ValueError):
        fit_private_motor_window(parent, **kw)
