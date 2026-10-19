"""Restore an already-verified current SONIC boundary, never a future target.

Callers authenticate complete original source/model/trace proof before using
this helper. Past measured states/actions rebuild history without replaying
all past neural calls. The CURRENT encoder/decoder are then actually called
and checked. A future branch must still infer from its own next body state.
"""

import numpy as np

from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch
from rosclaw_soccer.providers.g1.sonic_vector import BatchedSonicTracker

WIDTHS = dict(
    encoder_features=640,
    latent_token=64,
    decoder_input=994,
    raw_action_isaac=29,
    joint_target_mujoco_rad=29,
)


def restore_current_tracker(
    model: FrozenSonicG1Torch,
    *,
    cached_reference: np.ndarray,
    prefix: dict[str, np.ndarray],
    frame: int,
) -> BatchedSonicTracker:
    if (
        type(model) is not FrozenSonicG1Torch
        or model.variant != "low_latency"
        or type(frame) is not int
        or not 30 <= frame < 298
        or set(prefix)
        != {"canonical_qpos", "canonical_qvel"} | {"foundation_neural_" + k for k in WIDTHS}
    ):
        raise ValueError("qualified model and exact current/past-only SONIC capture required")
    for name, width in {
        "canonical_qpos": 43,
        "canonical_qvel": 41,
        **{"foundation_neural_" + k: v for k, v in WIDTHS.items()},
    }.items():
        a = prefix[name]
        dtype = np.float64 if name.startswith("canonical_") else np.float32
        if (
            not isinstance(a, np.ndarray)
            or a.shape != (frame + 1, 1, width)
            or a.dtype != dtype
            or not np.isfinite(a).all()
        ):
            raise ValueError("finite prefix must end exactly at current control frame")
    tracker = BatchedSonicTracker(
        model,
        cached_reference[None],
        low_latency_legacy_encoder_layout=True,
        capture_neural_observation=True,
    )
    q, v = prefix["canonical_qpos"], prefix["canonical_qvel"]
    tracker.reset(q[0], v[0])
    torch = tracker._torch
    for tick in range(frame + 1):
        if tick:
            tracker.observe(q[tick], v[tick])
        history = torch.cat(
            [
                torch.stack([entry[i] for entry in tracker._history], dim=1).reshape(1, -1)
                for i in range(5)
            ],
            dim=1,
        )
        actual = history.cpu().numpy()
        if not np.array_equal(actual, prefix["foundation_neural_decoder_input"][tick, :, 64:]):
            raise ValueError("causal decoder history not reproduced from past body/actions")
        if tick < frame:
            tracker.action = torch.as_tensor(
                prefix["foundation_neural_raw_action_isaac"][tick], device=model.device
            ).clone()
            tracker._next_frame = tick + 1
            tracker._pending_observation = True
    # Only CURRENT inference. Cached references must come from past planner calls.
    tracker.update(frame, q[frame], v[frame])
    captured = tracker.neural_observation()
    for name in WIDTHS:
        a = captured[name].cpu().numpy()
        b = prefix["foundation_neural_" + name][frame]
        if a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError(
                "current actual frozen-model output differs from qualified capture: " + name
            )
    return tracker
