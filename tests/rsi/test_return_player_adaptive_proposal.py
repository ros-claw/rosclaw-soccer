import numpy as np
import pytest
import torch
from test_anchored_receiving_motor import state
from test_continuous_velocity_stream import fake_policy
from test_velocity_prior_actor import tensors

from rosclaw_soccer.rsi.precontact_receiving_motor import PrecontactReceivingMotor
from rosclaw_soccer.rsi.return_player_adaptive_proposal import return_player_adaptive_proposal
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor


def test_zero_motor_exactly_preserves_same_state_velocity_reference():
    params = fake_policy().params
    actor = (
        PrecontactReceivingMotor(
            VelocityPriorActor(tensors(), reference_features=14),
            torch.as_tensor(params.action_scale, dtype=torch.float32),
        )
        .eval()
        .requires_grad_(False)
    )
    q, v, anchor = state()
    observation, reference = np.zeros(98, np.float32), np.zeros(29, np.float32)
    result = return_player_adaptive_proposal(
        actor,
        params,
        observation,
        reference,
        q,
        v,
        anchor,
        foot_contact_seen=True,
        other_body_contact_seen=False,
    )
    np.testing.assert_array_equal(result["action"], reference)
    np.testing.assert_array_equal(result["target"], result["original_target"])
    assert result["inputs"].shape == (112,) and result["inputs"][-2:].tolist() == [1, 0]
    assert result["physical_execution_proven"] is False
    reference[0] = np.nan
    with pytest.raises(ValueError):
        return_player_adaptive_proposal(
            actor,
            params,
            observation,
            reference,
            q,
            v,
            anchor,
            foot_contact_seen=True,
            other_body_contact_seen=False,
        )
    reference[0] = 0
    actor.plastic.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen exact"):
        return_player_adaptive_proposal(
            actor,
            params,
            observation,
            reference,
            q,
            v,
            anchor,
            foot_contact_seen=True,
            other_body_contact_seen=False,
        )
