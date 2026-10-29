import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.bounded_kick_decoder import decode_bounded, kick_target_bounds


def test_bounded_decoder_uses_canonical_joint_pattern_and_saturates():
    config = dict(
        default_joint_pos=[0.1] * 29,
        action_scale=[0.5] * 29,
        clip_actions=6,
        action_clip={".*_ankle_roll_joint": [-0.6, 0.6]},
    )
    low, high = kick_target_bounds(config)
    np.testing.assert_allclose(low[[5, 11]], -0.2, atol=1e-7)
    np.testing.assert_allclose(high[[5, 11]], 0.4, atol=1e-7)
    logits = torch.tensor([[-1000.0] * 29, [1000.0] * 29], requires_grad=True)
    result = decode_bounded(logits, torch.tensor(low), torch.tensor(high))
    assert np.all(result.detach().numpy()[0] >= low - 1e-6)
    assert np.all(result.detach().numpy()[1] <= high + 1e-6)
    result.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_codec_rejects_overlap_and_nonpositive_scale():
    config = dict(
        default_joint_pos=[0] * 29,
        action_scale=[1] * 29,
        clip_actions=3,
        action_clip={".*_ankle_roll_joint": [-1, 1], "left_ankle_roll_joint": [-1, 1]},
    )
    with pytest.raises(ValueError, match="unique"):
        kick_target_bounds(config)
    config["action_clip"] = {}
    config["action_scale"][0] = 0
    with pytest.raises(ValueError, match="positive"):
        kick_target_bounds(config)
