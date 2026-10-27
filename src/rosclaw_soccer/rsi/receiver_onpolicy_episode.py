"""Research-only terminal reward and sampled receiver episode validation."""

import numpy as np


def receiver_terminal_reward(
    *, handoff: bool, precise: bool, kick: bool, body_failure: bool
) -> float:
    if any(type(x) is not bool for x in (handoff, precise, kick, body_failure)):
        raise ValueError("explicit measured boolean terminal outcomes required")
    if precise and (not handoff or not kick or body_failure):
        raise ValueError(
            "precise healthy return must include handoff, real kick and no body failure"
        )
    if kick and not handoff:
        raise ValueError("outgoing return kick requires a true handoff")
    return 2.0 * precise + 1.0 * handoff + 0.25 * kick - 2.0 * body_failure


def build_receiver_episode(trace: dict[str, np.ndarray], *, reward: float) -> dict[str, np.ndarray]:
    names = {
        "control",
        "obs",
        "context14",
        "reference_action",
        "action",
        "SAM_executed_action",
        "SAM_logp",
        "SAM_rng_before",
        "SAM_rng_after",
        "SAM_active",
    }
    if not names <= set(trace) or not np.isfinite(reward):
        raise ValueError("full actual causal sampled receiver trace required")
    n = len(trace["control"])
    shapes = {
        "control": (n,),
        "obs": (n, 98),
        "context14": (n, 14),
        "reference_action": (n, 29),
        "action": (n, 29),
        "SAM_executed_action": (n, 29),
        "SAM_logp": (n,),
        "SAM_rng_before": (n, 32),
        "SAM_rng_after": (n, 32),
        "SAM_active": (n,),
    }
    if (
        not 1 <= n <= 300
        or any(trace[k].shape != shape for k, shape in shapes.items())
        or not np.array_equal(trace["control"], np.arange(n))
        or not trace["SAM_active"].all()
        or trace["SAM_rng_before"].dtype != np.uint8
        or trace["SAM_rng_after"].dtype != np.uint8
        or any(not np.isfinite(trace[k]).all() for k in names - {"SAM_rng_before", "SAM_rng_after"})
        or any(
            trace[k].dtype != np.float32
            for k in (
                "obs",
                "context14",
                "reference_action",
                "action",
                "SAM_executed_action",
                "SAM_logp",
            )
        )
    ):
        raise ValueError("finite exact actual 50Hz own sampler history required")
    rewards = np.zeros(n, np.float32)
    rewards[-1] = np.float32(reward)
    return dict(
        inputs=np.concatenate((trace["obs"], trace["context14"]), axis=1),
        reference=trace["reference_action"].copy(),
        old_mean=trace["action"].copy(),
        sampled_action=trace["SAM_executed_action"].copy(),
        old_logp=trace["SAM_logp"].copy(),
        rng_before=trace["SAM_rng_before"].copy(),
        rng_after=trace["SAM_rng_after"].copy(),
        rewards=rewards,
    )
