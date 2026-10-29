import pytest

from rosclaw_soccer.rsi.relay_receiver_actor_selection import select_relay_receiver_actor


def test_explicit_actor_selection_uses_distinct_candidate():
    whole, parent, candidate = object(), object(), object()

    def select(arm):
        return select_relay_receiver_actor(arm, whole=whole, parent=parent, candidate=candidate)

    assert select({}) is whole
    assert select({"receiver_learned": True, "receiver_child_candidate": False}) is parent
    assert select({"receiver_learned": True, "receiver_child_candidate": True}) is candidate
    with pytest.raises(ValueError):
        select({"receiver_child_candidate": True})
    with pytest.raises(ValueError):
        select_relay_receiver_actor(
            {"receiver_learned": True, "receiver_child_candidate": True},
            whole=whole,
            parent=parent,
            candidate=None,
        )
