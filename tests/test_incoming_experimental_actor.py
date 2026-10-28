import pytest

from scripts.rsi_probe_motiondecode_incoming import experimental_actor_state


def test_candidate_extracts_only_actor_and_never_baseline():
    checkpoint = {"policy_state_dict": {"actor.a": 1, "baseline.a": 2, "critic.a": 3}}
    assert experimental_actor_state(checkpoint, "candidate") == {"a": 1}
    assert experimental_actor_state({"actor_state_dict": {"a": 4}}, "imitation") == {"a": 4}


@pytest.mark.parametrize(
    "checkpoint,kind",
    [({}, "candidate"), ({"policy_state_dict": {"baseline.a": 2}}, "candidate"), ({}, "imitation")],
)
def test_missing_or_wrong_actor_is_rejected(checkpoint, kind):
    with pytest.raises(ValueError):
        experimental_actor_state(checkpoint, kind)
