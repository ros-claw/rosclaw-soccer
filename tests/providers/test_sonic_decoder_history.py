import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_decoder_history import SonicDecoderHistory
from rosclaw_soccer.providers.g1.sonic_runup import (
    ISAACLAB_TO_MUJOCO,
    G1SonicRunupController,
    _sonic_control_parameters,
)
from rosclaw_soccer.rsi.foundation_observation_capture import PREFIX, validate_measured_history


def inputs(lanes=1):
    q = np.zeros((lanes, 43), dtype=np.float32)
    q[:, 3] = q[:, 39] = 1
    return (
        q,
        np.zeros((lanes, 41), dtype=np.float32),
        np.zeros((lanes, 64), dtype=np.float32),
        np.zeros((lanes, 29), dtype=np.float32),
    )


@pytest.mark.parametrize("lanes", [1, 4])
def test_complete_history_matches_existing_independent_measured_history_auditor(lanes):
    history = SonicDecoderHistory(lanes)
    q, v, token, previous = inputs(lanes)
    qs, vs, tokens, actions, decoders, targets = [], [], [], [], [], []
    default = np.asarray(G1SonicRunupController.default_angles, dtype=np.float32)
    _, _, scale = _sonic_control_parameters(1.0, (1.0,) * 29)
    for frame in range(15):
        q[:, 7:36] = frame * 0.002
        v[:, 3:35] = frame * 0.003
        action = np.full((lanes, 29), frame * 0.004, dtype=np.float32)
        decoders.append(history.observe(frame, q, v, token, previous))
        qs.append(q.copy())
        vs.append(v.copy())
        tokens.append(token.copy())
        actions.append(action)
        targets.append(
            default + action[:, ISAACLAB_TO_MUJOCO] * np.asarray(scale, dtype=np.float32)
        )
        previous = action
    validate_measured_history(
        {
            "canonical_qpos": np.stack(qs),
            "canonical_qvel": np.stack(vs),
            PREFIX + "latent_token": np.stack(tokens),
            PREFIX + "decoder_input": np.stack(decoders),
            PREFIX + "raw_action_isaac": np.stack(actions),
            PREFIX + "joint_target_mujoco_rad": np.stack(targets),
        }
    )


def test_owned_inputs_and_fresh_episode_state():
    a, b = SonicDecoderHistory(1), SonicDecoderHistory(1)
    q, v, token, previous = inputs()
    first = a.observe(0, q, v, token, previous)
    q[:, 7:36] = 0.5
    assert not first.flags.writeable
    clean = inputs()
    assert np.array_equal(first, b.observe(0, *clean))
    assert not np.array_equal(a.observe(1, q, v, token, previous), first)


def test_bad_input_does_not_consume_frame_or_reuse_nonzero_action():
    history = SonicDecoderHistory(1)
    q, v, token, action = inputs()
    action[:] = 1
    with pytest.raises(ValueError):
        history.observe(0, q, v, token, action)
    action[:] = 0
    q[:, 3] = 0
    with pytest.raises(ValueError):
        history.observe(0, q, v, token, action)
    q[:, 3] = 1
    history.observe(0, q, v, token, action)
    with pytest.raises(ValueError, match="consecutive"):
        history.observe(0, q, v, token, action)
