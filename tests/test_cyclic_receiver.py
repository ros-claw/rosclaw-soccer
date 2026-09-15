from dataclasses import replace

import pytest
from test_admitted_receiver import incoming
from test_s368_recurrent_receiver import CONFIG, POLICY, motor

from rosclaw_soccer.providers.g1.cyclic_receiver import CyclicRecurrentReceiver
from rosclaw_soccer.providers.g1.recurrent_receiver import G1RecurrentReceiver
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig
from rosclaw_soccer.skills.team.motor_option import TeamReceiveCommitment
from rosclaw_soccer.skills.team.motor_rearm import TeamMotorRearmContext, validate_motor_successor


def cycle(tmp_path):
    first = motor(tmp_path)
    from rosclaw_soccer.sim.contracts import hash_bytes

    path = tmp_path / "receiver.npz"
    digest = hash_bytes(path.read_bytes())

    def factory():
        return G1RecurrentReceiver(
            path,
            agent_id=first.agent_id,
            expected_actor_hash=digest,
            foundation_hash=POLICY,
            foundation_config_hash=CONFIG,
        )

    return CyclicRecurrentReceiver(factory)


def finish(m):
    for frame in range(415, 516):
        m.propose(incoming(frame))
    assert m.completed


def context(m):
    return TeamMotorRearmContext(
        m.agent_id,
        520,
        10.4,
        515,
        m.contract_hash,
        False,
        TeamReceiveCommitment("blue.finisher", m.agent_id, 2, 518, 10.36, "sha256:" + "c" * 64),
    )


def test_two_bounded_skills_keep_foundation_and_require_new_incoming_ball(tmp_path):
    m = cycle(tmp_path)
    finish(m)
    ctx = context(m)
    following = m.successor(ctx)
    assert following is not None and following.receiver is not m.receiver
    validate_motor_successor(ctx, predecessor=m, successor=following)
    assert following.contract_hash == m.contract_hash
    assert following.propose(replace(incoming(520), committed_receiver=False)) is None
    for frame in range(521, 621):
        target = following.propose(incoming(frame))
        assert target == incoming(frame).foundation.target
    assert following.propose(incoming(621)) is None and following.completed
    assert m.start_frame == 415 and following.start_frame == 521
    with pytest.raises(ValueError, match="already issued"):
        m.successor(ctx)


def test_native_skill_and_old_lease_cannot_be_preempted(tmp_path):
    m = cycle(tmp_path)
    finish(m)
    ctx = context(m)
    assert m.successor(replace(ctx, native_motor_active=True)) is None
    old = replace(ctx.commitment, accepted_frame=515, accepted_time_sec=10.3)
    assert m.successor(replace(ctx, commitment=old)) is None
    assert m.successor(ctx) is not None


def test_fault_and_early_completion_cannot_rearm(tmp_path):
    m = cycle(tmp_path)
    with pytest.raises(ValueError, match="normal recorded"):
        m.successor(context(m))
    m.propose(incoming(415))
    with pytest.raises(ValueError):
        m.propose(incoming(417))
    with pytest.raises(ValueError, match="fault"):
        m.successor(context(m))


def test_world_cycles_are_explicit_and_disabled_hash_is_unchanged():
    config = IndependentTeamWorldConfig()
    assert (
        config.config_hash
        == "sha256:fd442bce83f737c64e2ee59ecc09dc204376100f9476a21361b41b538d8c41d1"
    )
    for change in ({"cyclic_receive_motors": True}, {"cyclic_receive_motors": 1}):
        with pytest.raises(ValueError):
            replace(config, **change)
    assert (
        replace(
            config,
            cyclic_receive_motors=True,
            motor_receive_commitment_context=True,
            retire_completed_motors=True,
            disjoint_motor_backends=True,
        ).config_hash
        != config.config_hash
    )


def test_reused_factory_state_rejected(tmp_path):
    backend = motor(tmp_path)
    m = CyclicRecurrentReceiver(lambda: backend)
    finish(m)
    with pytest.raises(ValueError, match="unused"):
        m.successor(context(m))
