"""Pure SIM-only adaptive velocity proposal for a returning team player.

This is not a world, action executor, transport, sampler, or safety gate. The
caller must separately validate its own consecutive observed state and commit
only the action actually executed by the shared simulated world.
"""

import numpy as np
import torch

from rosclaw_soccer.providers.g1.velocity_policy import target29
from rosclaw_soccer.rsi.precontact_receiving_motor import PrecontactReceivingMotor
from rosclaw_soccer.rsi.receiver_body_context import receiver_context14


def return_player_adaptive_proposal(
    actor,
    params,
    observation,
    reference_action,
    qpos,
    qvel,
    teammate_anchor,
    *,
    foot_contact_seen,
    other_body_contact_seen,
):
    """Evaluate a frozen motor using the *same* current reference observation."""
    if (
        type(actor) is not PrecontactReceivingMotor
        or actor.training
        or any(p.requires_grad or p.device.type != "cpu" for p in actor.parameters())
        or not isinstance(observation, np.ndarray)
        or observation.shape != (98,)
        or observation.dtype != np.float32
        or not isinstance(reference_action, np.ndarray)
        or reference_action.shape != (29,)
        or reference_action.dtype != np.float32
        or not np.isfinite(observation).all()
        or not np.isfinite(reference_action).all()
        or not np.array_equal(
            actor.action_scale.numpy(), np.asarray(params.action_scale, np.float32)
        )
    ):
        raise ValueError("frozen exact SIM motor, same-state reference and codec required")
    context = receiver_context14(
        qpos,
        qvel,
        teammate_anchor,
        foot_contact_seen=foot_contact_seen,
        other_body_contact_seen=other_body_contact_seen,
    )
    original_target = target29(reference_action, params)
    inputs = np.concatenate((observation, context))[None]
    with torch.no_grad():
        mean, evidence = actor.forward_with_evidence(
            torch.from_numpy(inputs), torch.from_numpy(reference_action[None].copy())
        )
    action = mean[0].numpy().copy()
    target = target29(action, params)
    return dict(
        inputs=inputs[0].copy(),
        reference_action=reference_action.copy(),
        action=action,
        target=np.asarray(target).copy(),
        original_target=np.asarray(original_target).copy(),
        realized_delta_radians=evidence["realized_delta_radians"][0].numpy().copy(),
        precontact_gate=float(evidence["precontact_gate"][0]),
        physical_execution_proven=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
