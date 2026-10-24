"""Untrained B physical identity admission, never an RL training-data protocol."""

import torch

from rosclaw_soccer.rsi.anchored_receiving_motor import AnchoredReceivingMotor

KIND = "receiving-whole-mlp-initialization-native-v1"


def admit_motor(actor: AnchoredReceivingMotor) -> None:
    if (
        type(actor) is not AnchoredReceivingMotor
        or actor.training
        or any(p.requires_grad or p.device.type != "cpu" for p in actor.parameters())
    ):
        raise ValueError("frozen explicit untrained CPU receiving motor required")
    plastic, baseline = actor.plastic.state_dict(), actor.baseline.state_dict()
    if plastic.keys() != baseline.keys() or any(
        not torch.isfinite(v).all() or not torch.equal(v, baseline[k]) for k, v in plastic.items()
    ):
        raise ValueError("exact whole-MLP zero identity required before physical qualification")


def motor_metadata(controls: int) -> dict[str, str | int | bool]:
    if type(controls) is not int or not 1 <= controls <= 400:
        raise ValueError("bounded actual controls required")
    return dict(
        controller_kind=KIND,
        receiver_motor_initialization_only=True,
        receiver_motor_optimizer_updates=0,
        receiver_reference_neural_calls=controls,
        receiver_whole_mlp_neural_calls=2 * controls,
        receiver_actual_executed_feedback=True,
        receiving_PPO_actor_rows=0,
    )
