"""Strict CPU task-command continuation readback, not policy activation."""

from typing import Any

import torch

from rosclaw_soccer.rsi.receiving_command_actor import ReceivingCommandActor
from rosclaw_soccer.rsi.receiving_task_command_actor import ReceivingTaskCommandActor


def load_task_command(
    saved: dict[str, Any],
    report: dict[str, Any],
    candidate: dict[str, Any],
    parent: dict[str, Any],
) -> ReceivingTaskCommandActor:
    continued = saved.get("schema") == "soccer.receiving_task_command_dagger.v3"
    prior_steps = 1200 if continued else 800
    task_steps = 800 if continued else 400
    if (
        saved["schema"]
        not in (
            "soccer.receiving_task_command_dagger.v2",
            "soccer.receiving_task_command_dagger.v3",
        )
        or saved["policy_domain"] != ReceivingTaskCommandActor.policy_domain
        or saved["seed"] != candidate["seed"]
        or saved["seed"] != parent["seed"]
        or saved["optimizer_updates"] != 400
        or saved["cumulative_prior_steps"] != prior_steps
        or saved["bank_hash"] != report["bank_hash"]
        or saved["parent_bc_hash"] != report["parent_bc_hash"]
        or saved["parent_checkpoint_hash"] != candidate["parent_checkpoint_hash"]
        or parent["optimizer_updates"] != 400
        or (
            continued
            and (
                parent.get("schema") != "soccer.receiving_task_command_dagger.v2"
                or parent.get("cumulative_prior_steps") != 800
                or not isinstance(report.get("parent_training_hash"), str)
                or saved.get("parent_training_hash") != report.get("parent_training_hash")
                or saved["parent_bc_hash"] != parent.get("parent_bc_hash")
            )
        )
        or len(saved["optimizer"]["param_groups"]) != 1
        or saved["optimizer"]["param_groups"][0]["params"] != list(range(7))
        or set(saved["optimizer"]["state"]) != set(range(7))
        or [float(saved["optimizer"]["state"][i]["step"]) for i in range(7)]
        != [prior_steps] * 6 + [task_steps]
    ):
        raise ValueError("exact bounded task continuation weights/parent/Adam evidence required")
    weights = (
        {k.removeprefix("prior."): v for k, v in parent["actor"].items() if k.startswith("prior.")}
        if continued
        else parent["actor"]
    )
    base = ReceivingCommandActor(weights["observation_mean"], weights["observation_scale"])
    if any(
        not torch.equal(weights[k], base.state_dict()[k])
        for k in ("command_center", "command_radius")
    ):
        raise ValueError("original velocity envelope cannot change")
    base.load_state_dict(weights, strict=True)
    actor = ReceivingTaskCommandActor(
        base, saved["actor"]["task_mean"], saved["actor"]["task_scale"]
    )
    for key in ("observation_mean", "observation_scale", "command_center", "command_radius"):
        if not torch.equal(saved["actor"]["prior." + key], weights[key]):
            raise ValueError("parent normalization/envelope cannot be silently reinterpreted")
    if continued and any(
        not torch.equal(saved["actor"][k], parent["actor"][k]) for k in ("task_mean", "task_scale")
    ):
        raise ValueError("second round must preserve the original task normalizer")
    actor.load_state_dict(saved["actor"], strict=True)
    if any(not torch.isfinite(p).all() for p in actor.parameters()):
        raise ValueError("finite task-conditioned parameters required")
    return actor.eval().requires_grad_(False)
