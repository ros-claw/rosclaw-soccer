import copy

import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.paid_successor_sequences import paired_arrays, paired_forward
from scripts.rsi_exam_paid_live_distilled import successor_memory


def records():
    first = dict(
        NN_obs=np.zeros((3, 160), np.float32),
        NN_action=np.ones((3, 29), np.float32),
        initial_time=np.asarray(0.0),
        time_before=np.asarray([0.0, 0.02, 0.04, 0.06]),
    )
    second = copy.deepcopy(first)
    second["NN_obs"][0, 125:154] = 1
    second["initial_time"] = np.asarray(0.2)
    second["PLAN_pose_offset"] = np.ones((3, 6), np.float64)
    return first, second


def test_whole_world_pairs_have_no_gap_padding_or_teacher_state():
    arrays = paired_arrays(*records())
    assert arrays["obs"].shape == (6, 160)
    assert int(arrays["boundary"]) == 3
    assert arrays["active"].tolist() == [False, False, False, True, True, True]
    assert set(arrays) == {"obs", "action", "active", "boundary"}


@pytest.mark.parametrize("fault", ["action", "clock", "nan", "length", "shape", "mask"])
def test_mixed_or_corrupt_pairs_fail_closed(fault):
    first, second = records()
    if fault == "action":
        second["NN_obs"][0, 125] = -1
    elif fault == "clock":
        second["initial_time"] = np.asarray(0.0)
    elif fault == "nan":
        first["NN_obs"][0, 0] = np.nan
    elif fault == "length":
        first["NN_action"] = np.zeros((4, 29), np.float32)
    elif fault == "shape":
        first["NN_obs"] = np.zeros((3, 159), np.float32)
    else:
        second["PLAN_pose_offset"] = np.zeros((4, 29))
    with pytest.raises(ValueError):
        paired_arrays(first, second)


class TinyMemory(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(()))

    def forward(self, obs, state):
        memory = state[0][0, 0, 0] + torch.cumsum(obs[:, 0, 0] * self.gain, dim=0)
        return memory[:, None, None].expand(-1, 1, 29), state


def test_carry_has_actual_cross_skill_gradient_reset_ablation_does_not():
    model = TinyMemory()
    obs = torch.zeros(6, 160)
    obs[:3, 0] = 1
    carry, _ = paired_forward(model, obs, 3, carry=True)
    reset, _ = paired_forward(model, obs, 3, carry=False)
    assert carry[:, 0].tolist() == [1, 2, 3, 3, 3, 3]
    assert reset[:, 0].tolist() == [1, 2, 3, 0, 0, 0]
    carry[3:].sum().backward()
    assert model.gain.grad.item() == 261


@pytest.mark.parametrize("boundary", [0, 6, -1])
def test_invalid_pair_boundary_rejected(boundary):
    with pytest.raises(ValueError):
        paired_forward(TinyMemory(), torch.zeros(6, 160), boundary, carry=True)


def test_inference_handoff_uses_private_actual_first_skill_memory():
    raw = {k: np.ones((3, 2, 1, 128), np.float32) for k in ("NN_h_after", "NN_c_after")}
    state = successor_memory(raw, "carry", same_actor=True)
    assert all(x.shape == (2, 1, 128) for x in state)
    state[0].zero_()
    assert raw["NN_h_after"].all()
    assert successor_memory(raw, "reset", same_actor=False) is None


@pytest.mark.parametrize("fault", ["experts", "mode", "nan", "shape"])
def test_untrained_mixed_or_invalid_memory_handoff_rejected(fault):
    raw = {k: np.ones((3, 2, 1, 128), np.float32) for k in ("NN_h_after", "NN_c_after")}
    if fault == "nan":
        raw["NN_h_after"][-1, 0, 0, 0] = np.nan
    elif fault == "shape":
        raw["NN_c_after"] = np.zeros((3, 128), np.float32)
    with pytest.raises(ValueError):
        successor_memory(
            raw, "unknown" if fault == "mode" else "carry", same_actor=fault != "experts"
        )
