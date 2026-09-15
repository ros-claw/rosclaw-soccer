from dataclasses import replace

import numpy as np
import pytest
from test_s368_recurrent_receiver import motor, observation

from rosclaw_soccer.providers.g1.admitted_receiver import (
    AdmittedRecurrentReceiver,
    incoming_receive_admissible,
)


def incoming(frame=415):
    o = observation(frame, frame * 0.02)
    velocity = list(o.qvel)
    velocity[35] = -1.0
    return replace(o, qvel=tuple(velocity), committed_receiver=True)


@pytest.mark.parametrize("fault", ["no-agreement", "receding", "stationary", "far", "fallen"])
def test_admission_requires_actual_incoming_context(fault):
    o = incoming()
    assert incoming_receive_admissible(o)
    q, v = list(o.qpos), list(o.qvel)
    if fault == "no-agreement":
        o = replace(o, committed_receiver=False)
    elif fault == "receding":
        v[35] = 1.0
    elif fault == "stationary":
        v[35] = 0.0
    elif fault == "far":
        q[36] = 2.0
    else:
        q[2] = 0.4
    assert not incoming_receive_admissible(replace(o, qpos=tuple(q), qvel=tuple(v)))


def test_bounded_admission_holds_readiness_then_completes_without_rearm(tmp_path):
    m = AdmittedRecurrentReceiver(motor(tmp_path))
    assert m.propose(replace(incoming(414), committed_receiver=False)) is None
    assert m.readiness(frame=415, time_sec=8.3).ball_action_ready
    for frame in range(415, 515):
        proposal = m.propose(incoming(frame))
        assert proposal is not None
        np.testing.assert_allclose(
            proposal.target_rad, incoming(frame).foundation.target.target_rad
        )
        assert not m.readiness(frame=frame, time_sec=frame * 0.02).ball_action_ready
    assert m.propose(incoming(515)) is None
    assert m.completed and m.readiness(frame=515, time_sec=10.3).ball_action_ready
    assert m.propose(incoming(516)) is None
    assert m.start_frame == 415


def test_foreign_or_skipped_observation_latches_and_blocks_commitment(tmp_path):
    m = AdmittedRecurrentReceiver(motor(tmp_path))
    m.propose(incoming())
    with pytest.raises(ValueError):
        m.propose(incoming(417))
    assert m.faulted
    assert not m.readiness(frame=418, time_sec=8.36).ball_action_ready
    with pytest.raises(ValueError, match="faulted"):
        m.propose(incoming(416))


def test_slow_incoming_protocol_is_explicit_and_does_not_invent_commitment(tmp_path):
    o = incoming()
    v = list(o.qvel)
    v[35] = -0.25
    slow = replace(o, qvel=tuple(v))
    assert not incoming_receive_admissible(slow)
    assert incoming_receive_admissible(slow, minimum_speed_mps=0.1)
    assert not incoming_receive_admissible(
        replace(slow, committed_receiver=False), minimum_speed_mps=0.1
    )
    frozen = motor(tmp_path)
    default = AdmittedRecurrentReceiver(frozen)
    assert (
        default.contract_hash
        == AdmittedRecurrentReceiver(frozen, minimum_speed_mps=0.4).contract_hash
    )
    assert (
        default.contract_hash
        != AdmittedRecurrentReceiver(frozen, minimum_speed_mps=0.1).contract_hash
    )
    for invalid in (True, 0.0, float("nan"), float("inf"), 1.1):
        with pytest.raises(ValueError):
            AdmittedRecurrentReceiver(frozen, minimum_speed_mps=invalid)
        with pytest.raises(ValueError):
            incoming_receive_admissible(slow, minimum_speed_mps=invalid)
