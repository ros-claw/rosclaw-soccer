from dataclasses import replace

import pytest

from rosclaw_soccer.growth.pass_handoff import PassHandoff
from rosclaw_soccer.growth.pass_target_commitment import live_pass_flight
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig


def test_real_flight_survives_proposal_change_but_not_expiry_or_interruption():
    promise = PassHandoff("blue.defender", "blue.playmaker", 0.0)
    assert not live_pass_flight(None, time_sec=0.5)
    assert not live_pass_flight(promise, time_sec=0.5)
    launched = promise.observe_contact(
        agent_id=promise.source, foot=True, force_n=10.0, time_sec=0.76
    )
    withdrawn = launched.withdraw_unlaunched(source_still_committed=False)
    assert live_pass_flight(withdrawn, time_sec=0.8)
    assert withdrawn.source_foot_contact_sec == 0.76
    repeated = withdrawn.observe_contact(
        agent_id=promise.source, foot=True, force_n=10.0, time_sec=1.0
    )
    assert repeated == withdrawn
    assert not live_pass_flight(repeated, time_sec=3.76)
    interrupted = repeated.observe_contact(
        agent_id="red.defender", foot=True, force_n=10.0, time_sec=1.1
    )
    assert not live_pass_flight(interrupted, time_sec=1.2)
    assert not live_pass_flight(
        promise.withdraw_unlaunched(source_still_committed=False), time_sec=0.8
    )


def test_flight_preservation_is_explicit_and_default_hash_unchanged():
    old = IndependentTeamWorldConfig()
    assert (
        old.config_hash == "sha256:fd442bce83f737c64e2ee59ecc09dc204376100f9476a21361b41b538d8c41d1"
    )
    for kwargs in (
        {"preserve_launched_handoff": True},
        {"preserve_launched_handoff": 1, "strict_receive_handoff": True},
    ):
        with pytest.raises(ValueError):
            replace(old, **kwargs)
    new = replace(old, strict_receive_handoff=True, preserve_launched_handoff=True)
    assert new.config_hash != replace(new, preserve_launched_handoff=False).config_hash
