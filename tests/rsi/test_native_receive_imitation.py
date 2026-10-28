import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rosclaw_soccer.rsi.native_receive_imitation import fit_receive_demonstrations


def test_actual_teacher_fit_improves_and_does_not_change_parent():
    torch.set_num_threads(1)
    torch.manual_seed(2)
    actor = nn.Linear(177, 29)
    baseline = copy.deepcopy(actor)
    policy = SimpleNamespace(actor=actor, baseline=baseline)
    observation = torch.randn(120, 177)
    rehearsal = torch.randn(1200, 177)
    with torch.no_grad():
        target = baseline(observation) + 0.05
        parent = {k: v.clone() for k, v in baseline.state_dict().items()}
    result = fit_receive_demonstrations(policy, observation, target, rehearsal)
    assert result["updates"] == 200 and result["teacher_mse_after"] < result["teacher_mse_before"]
    assert all(torch.equal(baseline.state_dict()[k], v) for k, v in parent.items())
    with pytest.raises(ValueError):
        fit_receive_demonstrations(policy, observation[:99], target[:99], rehearsal)
