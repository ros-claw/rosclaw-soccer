from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.growth.pass_failure_feedback import diagnose_passes
from rosclaw_soccer.growth.pass_handoff import PassHandoff


def test_motion_and_zero_force_or_shin_contact_cannot_launch_a_pass():
    pending = PassHandoff("red.playmaker", "red.finisher", 1.0)
    assert not pending.can_activate(1.5, progressed=True)
    for foot, force in ((False, 100.0), (True, 0.0)):
        observed = pending.observe_contact(
            agent_id=pending.source, foot=foot, force_n=force, time_sec=1.2
        )
        assert not observed.can_activate(1.5, progressed=True)
    launched = pending.observe_contact(agent_id=pending.source, foot=True, force_n=10, time_sec=1.3)
    assert launched.can_activate(1.5, progressed=True)
    assert not launched.can_activate(1.5, progressed=False)


def test_interruption_timeout_and_unstable_receiver_release_the_task():
    pending = PassHandoff("blue.playmaker", "blue.finisher", 1.0, 1.2)
    assert not pending.expired(4.0)
    assert pending.expired(4.2)
    assert pending.expired(2.0, receiver_stable=False)
    interrupted = pending.observe_contact(
        agent_id="red.defender", foot=False, force_n=5, time_sec=1.5
    )
    assert interrupted.expired(1.5) and not interrupted.can_activate(2.0, progressed=True)
    assert pending.source_foot_contact_sec == 1.2
    assert (
        pending.observe_contact(
            agent_id=pending.source, foot=True, force_n=5, time_sec=2.5
        ).source_foot_contact_sec
        == 1.2
    )


def test_invalid_or_stale_contact_does_not_create_a_valid_handoff():
    pending = PassHandoff("red.playmaker", "red.finisher", 1.0)
    with pytest.raises(ValueError):
        replace(pending, receiver="blue.finisher")
    with pytest.raises(ValueError):
        pending.observe_contact(
            agent_id=pending.source, foot=True, force_n=float("nan"), time_sec=2
        )
    with pytest.raises(ValueError):
        pending.can_activate(0.5, progressed=True)
    assert not pending.observe_contact(
        agent_id=pending.source, foot=True, force_n=50, time_sec=4
    ).can_activate(4, progressed=True)


def test_pass_flight_clock_starts_at_contact_and_nonfoot_interruption_rejects():
    n = 61
    trace = {
        "time": np.arange(n) * 0.1,
        "pass_source_agent_code": np.zeros(n),
        "pass_target_agent_code": np.zeros(n),
        "ball_contact_agent_code": np.zeros(n),
        "ball_contact_effector_code": np.ones(n),
        "ball_contact_force_n": np.ones(n),
        "ball_nonfoot_contact_agent_code": np.zeros(n),
        "ball_nonfoot_contact_force_n": np.zeros(n),
        "ball_pose": np.zeros((n, 7)),
        "ball_velocity": np.zeros((n, 6)),
        "red_finisher_left_foot_position": np.tile([1.0, 0, 0], (n, 1)),
        "red_finisher_right_foot_position": np.tile([1.0, 0.1, 0], (n, 1)),
    }
    trace["pass_source_agent_code"][:25] = 2
    trace["pass_target_agent_code"][:25] = 1
    trace["ball_contact_agent_code"][20] = 2
    trace["ball_contact_agent_code"][45] = 1
    trace["ball_pose"][:, 0] = np.clip((np.arange(n) - 20) / 25, 0, 1)
    trace["ball_velocity"][:, 0] = 0.4
    ids = ("red.finisher", "red.playmaker")
    assert not diagnose_passes(trace, ids)[0]["physical_receive_confirmed"]
    assert diagnose_passes(trace, ids, launch_relative=True)[0]["physical_receive_confirmed"]
    trace["ball_nonfoot_contact_agent_code"][30] = 2
    trace["ball_nonfoot_contact_force_n"][30] = 5
    result = diagnose_passes(trace, ids, launch_relative=True)[0]
    assert result["failure"] == "NONFOOT_INTERRUPTION" and not result["physical_receive_confirmed"]
