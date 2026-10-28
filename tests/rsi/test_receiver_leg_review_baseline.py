import pytest

from scripts.rsi_review_receiver_leg_limit import baseline_contract


def raw_baseline(block: int) -> dict:
    return {
        "protocol": "soccer.neural_relay_receiver_leg_target_projection.development.v26",
        "development_context_block": block,
        "receiver_leg_projection_is_unlearned": True,
        "receiver_actor_previous_action_remains_raw": True,
    }


def test_lateral_baseline_is_only_qualified_for_first_block():
    baseline_contract({"development_context_block": 0}, {}, False)
    with pytest.raises(ValueError):
        baseline_contract({"development_context_block": 1}, {}, False)


@pytest.mark.parametrize("block", [0, 1, 2])
def test_explicit_raw_baseline_must_match_block(block):
    baseline_contract({"development_context_block": block}, raw_baseline(block), True)
    with pytest.raises(ValueError):
        baseline_contract({"development_context_block": block}, raw_baseline((block + 1) % 3), True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol", "unrelated"),
        ("receiver_leg_projection_is_unlearned", False),
        ("receiver_actor_previous_action_remains_raw", False),
    ],
)
def test_reject_unqualified_baseline(field, value):
    baseline = raw_baseline(1)
    baseline[field] = value
    with pytest.raises(ValueError):
        baseline_contract({"development_context_block": 1}, baseline, True)


@pytest.mark.parametrize("block", [None, -1, 3, "1", True])
def test_reject_invalid_development_block(block):
    with pytest.raises(ValueError):
        baseline_contract({"development_context_block": block}, raw_baseline(1), True)
