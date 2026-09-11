"""A cancelled promise is not a cancelled physical pass in flight."""

from dataclasses import replace

import pytest

from rosclaw_soccer.growth.pass_handoff import PassHandoff


@pytest.mark.parametrize("directed", [False, True])
def test_withdraw_unlaunched_releases_pending_promise(directed):
    pending = PassHandoff(
        "blue.finisher",
        "blue.playmaker",
        1.0,
        launch_target_xy=(4.0, 1.0) if directed else None,
    )
    assert pending.withdraw_unlaunched(source_still_committed=True) is pending
    withdrawn = pending.withdraw_unlaunched(source_still_committed=False)
    assert withdrawn.interrupted and withdrawn.expired(1.1)
    assert not pending.interrupted
    assert not withdrawn.can_activate(1.1, progressed=True)


@pytest.mark.parametrize("directed", [False, True])
def test_measured_launch_survives_source_option_change(directed):
    pending = PassHandoff(
        "blue.finisher",
        "blue.playmaker",
        1.0,
        launch_target_xy=(4.0, 1.0) if directed else None,
    )
    if directed:
        flying = pending.observe_directed_launch(
            agent_id=pending.source,
            foot=True,
            force_n=10.0,
            time_sec=1.1,
            ball_xy=(3.0, 1.0),
            ball_velocity_xy=(1.0, 0.0),
        )
    else:
        flying = pending.observe_contact(
            agent_id=pending.source,
            foot=True,
            force_n=10.0,
            time_sec=1.1,
        )
    assert flying.source_foot_contact_sec == 1.1
    assert flying.withdraw_unlaunched(source_still_committed=False) is flying
    assert flying.can_activate(1.2, progressed=True)
    interrupted = replace(flying, interrupted=True)
    assert interrupted.withdraw_unlaunched(source_still_committed=True).interrupted


@pytest.mark.parametrize("bad", [None, 0, 1, "true", []])
def test_invalid_commitment_predicate_rejected(bad):
    pending = PassHandoff("red.finisher", "red.playmaker", 0.0)
    with pytest.raises(ValueError):
        pending.withdraw_unlaunched(source_still_committed=bad)
