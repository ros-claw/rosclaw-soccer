from dataclasses import replace

import pytest

from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation, TeamReceiveCommitment


def commitment():
    return TeamReceiveCommitment(
        "blue.playmaker", "blue.finisher", 1, 20, 0.4, "sha256:" + "a" * 64
    )


def observation():
    return TeamMotorObservation(
        "blue.finisher",
        21,
        0.42,
        "other",
        False,
        (0.0,) * 43,
        (0.0,) * 41,
        (0.0, 0.0, 0.0),
        committed_receiver=True,
        receive_commitment=commitment(),
    )


def test_commitment_is_optional_read_only_context():
    obs = observation()
    assert obs.receive_commitment == commitment()
    assert replace(obs, receive_commitment=None).receive_commitment is None
    assert replace(obs, frame=22, time_sec=0.44).receive_commitment == commitment()


@pytest.mark.parametrize(
    "change",
    [
        {"committed_receiver": False},
        {"agent_id": "red.finisher"},
        {"frame": 19},
        {"time_sec": 0.39},
        {"receive_commitment": "untyped"},
    ],
)
def test_foreign_future_and_unaccepted_context_rejected(change):
    with pytest.raises(ValueError):
        replace(observation(), **change)


@pytest.mark.parametrize(
    "change",
    [
        {"generation": 0},
        {"generation": True},
        {"accepted_frame": 20.0},
        {"accepted_time_sec": float("nan")},
        {"accepted_time_sec": True},
        {"handshake_hash": "unknown"},
        {"source_agent_id": "blue.finisher"},
    ],
)
def test_malformed_identity_rejected(change):
    with pytest.raises(ValueError):
        replace(commitment(), **change)


def test_tampered_frozen_context_revalidated():
    obs = observation()
    object.__setattr__(obs.receive_commitment, "generation", 0)
    with pytest.raises(ValueError):
        replace(obs, frame=22)


def test_new_generation_is_distinct_even_for_same_pair():
    old = commitment()
    new = replace(old, generation=2, accepted_frame=90, accepted_time_sec=1.8)
    assert old != new


def test_world_opt_in_preserves_legacy_identity():
    world = IndependentTeamWorldConfig()
    assert (
        world.config_hash
        == "sha256:fd442bce83f737c64e2ee59ecc09dc204376100f9476a21361b41b538d8c41d1"
    )
    assert replace(world, motor_receive_commitment_context=True).config_hash != world.config_hash
    with pytest.raises(ValueError):
        replace(world, motor_receive_commitment_context=1)
