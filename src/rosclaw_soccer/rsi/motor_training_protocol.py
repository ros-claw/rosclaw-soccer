"""Fixed scientific protocols, not caller-selected checkpoint counters."""

PROTOCOL_STEPS = {"bc3000": (0, 3000), "dagger4500": (3000, 4500)}


def checkpoint_step_pair(protocol: str) -> tuple[int, int]:
    if type(protocol) is not str or protocol not in PROTOCOL_STEPS:
        raise ValueError("known fixed motor learning protocol required")
    return PROTOCOL_STEPS[protocol]
