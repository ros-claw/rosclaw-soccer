"""One explicit actor choice for SIM preflight and actual receiving worlds."""


def select_relay_receiver_actor(arm, *, whole, parent, candidate):
    selected = arm.get("receiver_child_candidate", False)
    learned = arm.get("receiver_learned", False)
    if type(selected) is not bool or type(learned) is not bool or (selected and not learned):
        raise ValueError("explicit consistent receiver actor role required")
    actor = candidate if selected else parent if learned else whole
    if actor is None:
        raise ValueError("selected receiving actor is unavailable")
    return actor
