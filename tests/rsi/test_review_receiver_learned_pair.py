from scripts.rsi_review_receiver_learned_pair import compare_outcomes


def test_failure_reduction_and_new_failure_are_not_swapped():
    rows = [
        {
            "baseline": {"body_failure": True, "handoff": False},
            "candidate": {"body_failure": False, "handoff": True},
        },
        {
            "baseline": {"body_failure": False, "handoff": True},
            "candidate": {"body_failure": True, "handoff": True},
        },
        {
            "baseline": {"body_failure": True, "handoff": False},
            "candidate": {"body_failure": False, "handoff": False},
        },
    ]
    assert compare_outcomes(rows, "body_failure") == {
        "rescued": 2,
        "lost": 1,
        "baseline": 2,
        "candidate": 1,
    }
    assert compare_outcomes(rows, "handoff") == {
        "rescued": 1,
        "lost": 0,
        "baseline": 1,
        "candidate": 2,
    }
