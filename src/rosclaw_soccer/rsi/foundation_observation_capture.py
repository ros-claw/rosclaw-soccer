"""Explicit SIM-only recording contract for actual frozen full-body NN calls.

This is training-data preparation, not a trained policy or an independent
foundation replay. Recorded targets precede navigation/task-space/motor layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.sonic_runup import (
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB,
    G1SonicRunupController,
    _sonic_control_parameters,
)
from rosclaw_soccer.sim.contracts import hash_bytes

FIELDS = {
    "encoder_features": 640,
    "latent_token": 64,
    "decoder_input": 994,
    "raw_action_isaac": 29,
    "joint_target_mujoco_rad": 29,
}
PREFIX = "foundation_neural_"


def capture_contract() -> dict[str, Any]:
    from rosclaw_soccer.providers.g1 import sonic_vector

    return {
        "schema": "soccer.rsi.foundation_observation_capture.v1",
        "capture_source_hash": hash_bytes(Path(__file__).read_bytes()),
        "tracker_source_hash": hash_bytes(Path(sonic_vector.__file__).read_bytes()),
        "fields": dict(FIELDS),
        "history_frames": 10,
        "raw_action_order": "ISAACLAB",
        "joint_target_order": "MUJOCO_CANONICAL",
        "isaac_to_mujoco": ISAACLAB_TO_MUJOCO.tolist(),
        "mujoco_to_isaac": MUJOCO_TO_ISAACLAB.tolist(),
        "layout": "low_latency_legacy_encoder_layout",
        "target_stage": "foundation_before_navigation_taskspace_and_motor",
        "foundation_recomputed_independently": False,
        "optimizer_updates": 0,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }


def validate_capture(contract: Any, arrays: dict[str, Any], *, frames: int, lanes: int) -> None:
    """Validate owned finite float32 record dimensions and token consistency.

    Neither the encoder nor the decoder is independently recomputed here.
    Source mismatch must be reviewed explicitly, not silently grandfathered.
    """
    if not isinstance(contract, dict) or contract != capture_contract():
        raise ValueError("foundation capture contract/source differs")
    if type(frames) is not int or type(lanes) is not int or not (1 <= frames <= 200000):
        raise ValueError("bounded foundation capture frame/lane count required")
    if not 1 <= lanes <= 4096:
        raise ValueError("bounded foundation capture frame/lane count required")
    recorded = {name for name in arrays if name.startswith(PREFIX)}
    if recorded != {PREFIX + name for name in FIELDS}:
        raise ValueError("foundation capture fields differ")
    for name, width in FIELDS.items():
        value = np.asarray(arrays[PREFIX + name])
        if (
            value.shape != (frames, lanes, width)
            or value.dtype != np.dtype("float32")
            or not np.isfinite(value).all()
        ):
            raise ValueError("foundation capture requires finite float32 bounded tensors")
    if not np.array_equal(
        arrays[PREFIX + "latent_token"], arrays[PREFIX + "decoder_input"][:, :, :64]
    ):
        raise ValueError("foundation decoder token differs from encoder token")


def validate_measured_history(arrays: dict[str, Any]) -> None:
    """Independently reconstruct decoder history from recorded measured body state.

    Reconstructs the ten chronological proprioceptive history entries and the
    action-to-target order/scale. Does not reconstruct planner references or NN
    outputs, so cannot certify the encoder/decoder themselves.
    """
    decoder = arrays[PREFIX + "decoder_input"]
    frames, lanes, _ = decoder.shape
    q = np.asarray(arrays["canonical_qpos"], dtype=np.float32)
    v = np.asarray(arrays["canonical_qvel"], dtype=np.float32)
    if (
        q.shape != (frames, lanes, 43)
        or v.shape != (frames, lanes, 41)
        or not np.isfinite(q).all()
        or not np.isfinite(v).all()
        or not np.allclose(np.linalg.norm(q[:, :, 3:7], axis=2), 1, atol=1e-4, rtol=0)
    ):
        raise ValueError("foundation capture requires finite normalized measured body state")
    default = np.asarray(G1SonicRunupController.default_angles, dtype=np.float32)
    _, _, scale = _sonic_control_parameters(1.0, (1.0,) * 29)
    action = arrays[PREFIX + "raw_action_isaac"]
    previous = np.concatenate((np.zeros_like(action[:1]), action[:-1]), axis=0)
    w, x, y, z = (q[:, :, i] for i in range(3, 7))
    gravity = np.stack((2 * (w * y - x * z), -2 * (w * x + y * z), 2 * (x * x + y * y) - 1), axis=2)
    entries = (
        v[:, :, 3:6],
        (q[:, :, 7:36] - default)[:, :, MUJOCO_TO_ISAACLAB],
        v[:, :, 6:35][:, :, MUJOCO_TO_ISAACLAB],
        previous,
        gravity,
    )
    indices = np.maximum(np.arange(frames)[:, None] + np.arange(-9, 1)[None, :], 0)
    expected = np.concatenate(
        [entry[indices].transpose(0, 2, 1, 3).reshape(frames, lanes, -1) for entry in entries],
        axis=2,
    )
    if not np.allclose(expected, decoder[:, :, 64:], atol=1e-6, rtol=0):
        raise ValueError("foundation decoder history differs from measured body/action history")
    target = default + action[:, :, ISAACLAB_TO_MUJOCO] * np.asarray(scale, dtype=np.float32)
    if not np.allclose(target, arrays[PREFIX + "joint_target_mujoco_rad"], atol=1e-6, rtol=0):
        raise ValueError("foundation action-to-target order/scale differs")
