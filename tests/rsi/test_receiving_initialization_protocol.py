import pytest
import torch
from test_velocity_prior_actor import tensors

from rosclaw_soccer.rsi.anchored_receiving_motor import AnchoredReceivingMotor
from rosclaw_soccer.rsi.receiving_initialization_protocol import admit_motor, motor_metadata
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor


def motor():
    return (
        AnchoredReceivingMotor(
            VelocityPriorActor(tensors(), reference_features=14), torch.full((29,), 0.25)
        )
        .eval()
        .requires_grad_(False)
    )


def test_initialization_only_and_four_network_total_accounting():
    admit_motor(motor())
    meta = motor_metadata(389)
    assert meta["receiver_reference_neural_calls"] == 389
    assert meta["receiver_whole_mlp_neural_calls"] == 778
    assert meta["receiving_PPO_actor_rows"] == 0


@pytest.mark.parametrize("fault", ["gradient", "changed", "nonfinite", "training"])
def test_refuse_trained_or_unfrozen_model(fault):
    actor = motor()
    if fault == "gradient":
        actor.plastic.requires_grad_(True)
    elif fault == "training":
        actor.train()
    else:
        with torch.no_grad():
            next(actor.plastic.parameters()).flatten()[0] += (
                float("nan") if fault == "nonfinite" else 0.1
            )
    with pytest.raises(ValueError):
        admit_motor(actor)


@pytest.mark.parametrize("controls", [0, -1, 401, True, 1.0])
def test_no_unbounded_or_fake_controls(controls):
    with pytest.raises(ValueError):
        motor_metadata(controls)


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("fault", ["no_sampling", "role_ablation", "teacher", "plane"])
def test_stacked_or_undeclared_protocol_rejected_before_world(replay, fault):
    from scripts.rsi_pass_transfer_replay_trial import replay_trial
    from scripts.rsi_pass_transfer_trial import collect_trial

    kwargs = dict(
        receiver_motor=motor(),
        sampling_seed=123,
        sampling_include_recovery=True,
        output=None,
        index=0,
        counts=None,
    )
    if fault == "no_sampling":
        kwargs["sampling_seed"] = None
    elif fault == "role_ablation":
        kwargs["receiver_role_ablation"] = True
    elif fault == "teacher":
        kwargs["feedback_variant"] = "unexpected"
    else:
        kwargs["receiver_intent_mode"] = "anchor-plane-v2"
    # None setup proves no simulator/model/world allocation is reached.
    with pytest.raises(ValueError, match="separate untrained receiving"):
        if replay:
            replay_trial(
                None,
                None,
                dict(assisted=False),
                dict(receiver_motor_initialization_only=True),
                raw_root=None,
                **kwargs,
            )
        else:
            collect_trial(None, None, dict(assisted=False), **kwargs)
