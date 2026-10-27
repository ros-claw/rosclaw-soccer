"""Construct a zero-difference frozen full receiver for SIM identity research."""


def zero_receiver_actor(setup):
    import torch

    from rosclaw_soccer.rsi.anchored_receiving_motor import AnchoredReceivingMotor
    from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor

    prior = VelocityPriorActor.from_onnx(
        setup.velocity_path,
        expected_hash=setup.pins[str(setup.velocity_path)],
        reference_features=14,
    )
    scale = torch.tensor(setup.velocity_parameters.action_scale, dtype=torch.float32)
    actor = AnchoredReceivingMotor(prior, scale).eval().requires_grad_(False)
    left, right = actor.plastic.state_dict(), actor.baseline.state_dict()
    if left.keys() != right.keys() or any(not torch.equal(left[k], right[k]) for k in left):
        raise ValueError("exact frozen identical plastic and anchor networks required")
    return actor
