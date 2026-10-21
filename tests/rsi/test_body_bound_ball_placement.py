"""Run in qualified MJLab env; pure CPU geometry, no simulator allocation."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("mjlab")

from scripts.rsi_mjlab_motion_backend import BodyBoundBallPlacementMixin  # noqa: E402


def fixture():
    obj = BodyBoundBallPlacementMixin()
    full = torch.zeros(1, 2, 3, 3)
    full[0, 0, 1] = torch.tensor([20.0, 4.0, 0.8])
    full[0, 1, 1] = torch.tensor([30.0, 4.0, 0.8])
    full[0, 0, 2] = torch.tensor([1.0, 2.0, 0.8])
    full[0, 1, 2] = torch.tensor([3.0, 2.0, 0.8])
    obj.motion = SimpleNamespace(
        _body_indexes=torch.tensor([0, 2]), body_pos_w=full[:, :, [0, 2]], _body_pos_w=full
    )
    obj.robot = SimpleNamespace(body_names=("pelvis", "right_hip_pitch_link", "torso_link"))
    obj.cfg = SimpleNamespace(fixed_ball_pos=None, anchor_body_name="torso_link")
    obj.motion_anchor_body_index = 1
    obj.device = "cpu"
    obj.motion_idx = torch.zeros(3, dtype=torch.long)
    obj.motion_length = torch.full((3,), 2, dtype=torch.long)
    obj.curve_radius_offset = torch.full((3,), 0.1)
    obj._target_arc_angle = 0.0
    obj.soccer_ball_pos = torch.zeros(3, 3)
    return obj


def test_subset_anchor_uses_named_torso_not_same_number_full_hip():
    obj = fixture()
    obj._compute_soccer_ball_positions(torch.tensor([1]))
    torch.testing.assert_close(obj.soccer_ball_pos[1], torch.tensor([3.1, 2.0, 0.11]))
    assert torch.count_nonzero(obj.soccer_ball_pos[[0, 2]]) == 0


def test_wrong_body_binding_is_rejected_without_writing_ball():
    obj = fixture()
    obj.motion._body_indexes = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="named physical body"):
        obj._compute_soccer_ball_positions(torch.tensor([0]))
    assert torch.count_nonzero(obj.soccer_ball_pos) == 0
