import pytest

from scripts.rsi_airborne_receiving_baseline import posttouch_research_components

HASH = "sha256:" + "a" * 64
OTHER_HASH = "sha256:" + "b" * 64


def test_disabled_has_no_physical_or_navigation_hook():
    assert posttouch_research_components(
        "disabled", agent_id="red.finisher", foundation_hash=HASH, foundation_config_hash=OTHER_HASH
    ) == (None, None)


def test_research_arms_bind_private_completed_contact_and_fixed_weights():
    zero, zero_evidence = posttouch_research_components(
        "zero", agent_id="red.finisher", foundation_hash=HASH, foundation_config_hash=OTHER_HASH
    )
    lateral, lateral_evidence = posttouch_research_components(
        "lateral",
        agent_id="blue.playmaker",
        foundation_hash=HASH,
        foundation_config_hash=OTHER_HASH,
    )
    assert zero.post_weights == (0.0,) * 5
    assert lateral.post_weights == (0.0, 0.5, 0.0, 0.5, -0.25)
    assert zero.mailbox is zero_evidence.mailbox
    assert lateral.mailbox is lateral_evidence.mailbox
    assert zero.agent_id == "red.finisher" and lateral.agent_id == "blue.playmaker"
    assert zero.activation_ceiling == lateral.activation_ceiling == "SIM_ONLY"
    assert zero.contract_hash != lateral.contract_hash


def test_unregistered_arm_rejected():
    with pytest.raises(ValueError):
        posttouch_research_components(
            "best", agent_id="red.finisher", foundation_hash=HASH, foundation_config_hash=OTHER_HASH
        )
