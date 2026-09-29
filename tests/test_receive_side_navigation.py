"""A side-conditioned receiver keeps one SIM_ONLY navigation authority."""

import pytest

from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.rsi.team_receive_side_navigation import TeamReceiveSideNavigation
from rosclaw_soccer.skills.team.navigation_option import NavigationSlot


def test_side_navigation_has_one_bound_authority_and_rejects_invalid_weights() -> None:
    mailbox = ReceiveContactMailbox("red.finisher")
    foundation = "sha256:" + "a" * 64
    config = "sha256:" + "b" * 64
    actor = TeamReceiveSideNavigation(
        "red.finisher",
        foundation,
        config,
        mailbox,
        (0.0, 0.0, 0.5, 0.0, 0.0),
        (0.0,) * 5,
    )
    slot = NavigationSlot(actor)
    assert slot.agent_id == "red.finisher"
    assert actor.selected_side is None
    assert actor.post_active_frames == 0
    with pytest.raises(ValueError):
        TeamReceiveSideNavigation(
            "red.finisher",
            foundation,
            config,
            mailbox,
            (0.0, 0.0, float("nan"), 0.0, 0.0),
            (0.0,) * 5,
        )
    with pytest.raises(ValueError):
        TeamReceiveSideNavigation(
            "blue.finisher",
            foundation,
            config,
            mailbox,
            (0.0,) * 5,
            (0.0,) * 5,
        )
