"""Fixed-width own-action evidence for a simulated return player."""

import numpy as np


def return_player_sampling_trace(adapted):
    """Serialize a committed proposal; zero padding is never an optimizer row."""
    sampled = adapted is not None and "sampling_trace" in adapted
    evidence = adapted["sampling_trace"] if sampled else None
    return {
        "sampling_active": np.array(sampled),
        "sampling_inputs": np.asarray(adapted["inputs"] if sampled else np.zeros(112), np.float32),
        "sampling_mean": np.asarray(
            adapted["sampling_mean"] if sampled else np.zeros(29), np.float32
        ),
        "sampling_reference": np.asarray(
            adapted["reference_action"] if sampled else np.zeros(29), np.float32
        ),
        "sampling_action": np.asarray(adapted["action"] if sampled else np.zeros(29), np.float32),
        "sampling_logp": np.asarray(evidence["logp"] if sampled else 0, np.float32),
        "sampling_rng_before": np.asarray(
            evidence["rng_before"] if sampled else np.zeros(32), np.uint8
        ),
        "sampling_rng_after": np.asarray(
            evidence["rng_after"] if sampled else np.zeros(32), np.uint8
        ),
    }
