"""Pure move/clock contract tests, not real G1/SONIC qualification."""

from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.feedback_branch_control import BODY_SHAPES, G1FeedbackBranchControl


def completed():
    c = object.__new__(G1FeedbackBranchControl)
    c.frame, c.consumed, c._continuation_taken = 30, True, False
    c._completed_motor_delta = np.full(12, 0.01)
    c.previous = np.zeros(12)
    c.tracker = SimpleNamespace(_next_frame=32)
    c.decoder = SimpleNamespace(
        _active_frame=None,
        _memory=SimpleNamespace(last_frame=31),
        _recurrent=SimpleNamespace(next_index=2),
        _policy_hash="fixed",
    )
    c.ik = SimpleNamespace(next_frame=31, pending=(31, np.zeros(6)))
    c.policy = {"policy_hash": "fixed"}
    c.body = {k: np.zeros((32, 1, *shape)) for k, shape in BODY_SHAPES.items()}
    return c


def test_one_owner_moves_without_caller_supplied_future_state():
    old = completed()
    tracker, decoder, ik = old.tracker, old.decoder, old.ik
    new = old.continue_owned()
    assert (new.tracker, new.decoder, new.ik) == (tracker, decoder, ik)
    assert new.frame == 31 and not new.consumed and not new._continuation_taken
    assert np.array_equal(new.previous, np.full(12, 0.01))
    assert not hasattr(old, "tracker") and not hasattr(old, "decoder")
    assert not hasattr(old, "ik") and not hasattr(old, "body")
    with pytest.raises(ValueError, match="privately owned"):
        old.continue_owned()
    with pytest.raises(ValueError, match="one complete"):
        old.next_target(None, np.zeros(6))
    with pytest.raises(ValueError, match="privately owned"):
        new.continue_owned()


@pytest.mark.parametrize(
    "bad", ["unconsumed", "missing_delta", "nan_delta", "clock", "policy", "history", "planner"]
)
def test_invalid_or_incomplete_successor_is_not_moved(bad):
    c = completed()
    if bad == "unconsumed":
        c.consumed = False
    elif bad == "missing_delta":
        c._completed_motor_delta = None
    elif bad == "nan_delta":
        c._completed_motor_delta[0] = np.nan
    elif bad == "clock":
        c.decoder._recurrent.next_index += 1
    elif bad == "policy":
        c.policy["policy_hash"] = "other"
    elif bad == "history":
        c.body["joint_position_rad"] = c.body["joint_position_rad"][:-1]
    else:
        c.frame = 38
    with pytest.raises(ValueError):
        c.continue_owned()
    assert hasattr(c, "decoder") and not c._continuation_taken
