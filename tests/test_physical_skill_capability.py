import pytest

from rosclaw_soccer.growth.tactical_2v1 import TacticalAction
from rosclaw_soccer.growth.tactical_2v1_actor import TwoVsOneTacticalDecision
from rosclaw_soccer.training.physical_skill_capability import (
    CapabilityResearchGate,
    PhysicalSkillTrial,
    choose_tactical_option,
    summarize_exact_capability,
)
from rosclaw_soccer.training.tactical_capability_bridge import constrain_two_vs_one_proposal


def H(char):
    return "sha256:" + char * 64


IDENTITY = ("pass_receive", H("1"), H("2"), H("3"))


def trial(index, outcome="FAILURE", safe=True, group=None):
    return PhysicalSkillTrial(group or f"group-{index}", *IDENTITY, H(f"{index:x}"), outcome, safe)


def test_failure_and_unknown_cannot_become_available_tactical_action():
    rows = tuple(trial(i) for i in range(1, 9))
    report = summarize_exact_capability(rows, CapabilityResearchGate(5, 0.4))
    assert report["successes"] == 0 and report["observed"] == 8
    assert report["empirical_fraction"] == 0 and not report["research_gate_passed"]
    assert report["successor_ready_observed"] == 0 and report["duration_mean_sec"] is None
    assert choose_tactical_option({"pass": 100.0}, {"pass": report}, {"pass": IDENTITY}) is None
    unknown = summarize_exact_capability((trial(1, "UNKNOWN", None),), CapabilityResearchGate(1, 0))
    assert unknown["unknown_outcomes"] == 1 and unknown["empirical_fraction"] is None
    assert not unknown["research_gate_passed"]


def test_exact_identity_and_duplicate_groups_cannot_be_bypassed():
    with pytest.raises(ValueError):
        summarize_exact_capability(
            (trial(1), trial(2, group="group-1")), CapabilityResearchGate(1, 0)
        )
    with pytest.raises(ValueError):
        PhysicalSkillTrial("group", *IDENTITY, H("4"), "SUCCESS", False)
    report = summarize_exact_capability(
        tuple(trial(i, "SUCCESS") for i in range(1, 9)), CapabilityResearchGate(5, 0.4)
    )
    assert report["research_gate_passed"]
    assert (
        choose_tactical_option(
            {"pass": 1.0, "shoot": 3.0}, {"pass": report}, {"pass": IDENTITY, "shoot": IDENTITY}
        )
        == "pass"
    )
    with pytest.raises(ValueError):
        choose_tactical_option({"pass": 1.0}, {"pass": report}, {"pass": ("other", *IDENTITY[1:])})
    report["successes"] = 100
    with pytest.raises(ValueError):
        choose_tactical_option({"pass": 1.0}, {"pass": report}, {"pass": IDENTITY})


def test_unknown_safety_and_mixed_context_fail_closed():
    with pytest.raises(ValueError):
        PhysicalSkillTrial("bad", *IDENTITY, H("5"), "SUCCESS", True, float("nan"))
    with pytest.raises(ValueError):
        PhysicalSkillTrial("bad", *IDENTITY, H("5"), "SUCCESS", True, "not-a-duration")
    row = trial(1, "FAILURE", None)
    report = summarize_exact_capability((row,), CapabilityResearchGate(1, 0))
    assert report["unknown_safety"] == 1 and not report["research_gate_passed"]
    another = PhysicalSkillTrial(
        "another", IDENTITY[0], H("a"), IDENTITY[2], IDENTITY[3], H("6"), "FAILURE", True
    )
    with pytest.raises(ValueError):
        summarize_exact_capability((row, another), CapabilityResearchGate(1, 0))
    assert choose_tactical_option({"unseen": 3.0}, {}, {"unseen": IDENTITY}) is None


def test_tactical_actor_uses_only_evidenced_option_without_claiming_authority():
    decision = TwoVsOneTacticalDecision(
        True, TacticalAction.SHOOT, "TEST", 0.7, 1.0, 3.0, 0.0, H("7")
    )
    good = summarize_exact_capability(
        tuple(trial(i, "SUCCESS") for i in range(1, 9)), CapabilityResearchGate(5, 0.4)
    )
    proposal = constrain_two_vs_one_proposal(
        decision, {"pass": good}, {"pass": IDENTITY, "shoot": IDENTITY}
    )
    assert proposal.action == TacticalAction.PASS
    assert not proposal.hardware_authorized and not proposal.promotion_authorized
    no_evidence = constrain_two_vs_one_proposal(decision, {}, {"pass": IDENTITY, "shoot": IDENTITY})
    assert no_evidence.action is None and no_evidence.route == "NO_PHYSICAL_EVIDENCE_REPLAN"
