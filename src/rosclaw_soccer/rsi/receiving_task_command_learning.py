"""Continue BC400 with task conditioning and OWN-state DAgger advice.

No data admission, teacher execution, new physics or promotion. Original six
Adam states and private batch RNG continue, the new task projection starts at
zero. Half each minibatch retains old training, half uses own-state advice.
"""

import copy
from collections.abc import Callable
from typing import Any

import torch

from rosclaw_soccer.rsi.receiving_command_actor import ReceivingCommandActor
from rosclaw_soccer.rsi.receiving_task_command_actor import ReceivingTaskCommandActor


def continue_task_command(
    parent_state: dict[str, Any],
    old_x: torch.Tensor,
    old_y: torch.Tensor,
    old_training: torch.Tensor,
    old_validation: torch.Tensor,
    own_x: torch.Tensor,
    own_advice: torch.Tensor,
    own_critical: torch.Tensor,
    *,
    steps: int = 400,
    continuation_round: int = 1,
    on_step: Callable[[int, float, float], None] | None = None,
) -> dict[str, Any]:
    for x, y in ((old_x, old_y), (own_x, own_advice)):
        if (
            x.ndim != 2
            or x.shape[1] != 118
            or not 1 <= len(x) <= 38400
            or y.shape != (len(x), 3)
            or x.dtype != torch.float32
            or y.dtype != x.dtype
            or x.device.type != "cpu"
            or y.device != x.device
            or x.requires_grad
            or y.requires_grad
            or not torch.isfinite(x).all()
            or not torch.isfinite(y).all()
            or ((y < torch.tensor([-0.5, -0.5, -1])) | (y > torch.tensor([1.0, 0.5, 1.0]))).any()
        ):
            raise ValueError("finite own118 states and original bounded3 command labels required")
    if (
        type(continuation_round) is not int
        or continuation_round not in (1, 2)
        or parent_state["optimizer_updates"] != 400
        or any(
            m.shape != (len(old_x),) or m.dtype != torch.bool or m.device.type != "cpu"
            for m in (old_training, old_validation)
        )
        or (old_training & old_validation).any()
        or not old_training.any()
        or not old_validation.any()
        or own_critical.shape != (len(own_x),)
        or own_critical.dtype != torch.bool
        or own_critical.device.type != "cpu"
        or not own_critical.any()
        or type(steps) is not int
        or not 1 <= steps <= 400
    ):
        raise ValueError(
            "BC400 parent, disjoint old TRAIN/validation and bounded continuation required"
        )
    train_ids = torch.nonzero(old_training, as_tuple=True)[0]
    val_ids = torch.nonzero(old_validation, as_tuple=True)[0]
    critical_ids = torch.nonzero(own_critical, as_tuple=True)[0]
    weights = parent_state["actor"]
    prior: ReceivingCommandActor | ReceivingTaskCommandActor
    with torch.random.fork_rng(devices=[]):
        if continuation_round == 1:
            task_training = torch.cat((old_x[train_ids, 112:], own_x[:, 112:]))
            task_mean = task_training.mean(0)
            task_scale = torch.maximum(
                task_training.std(0, unbiased=False), torch.tensor([0.25] * 3 + [0.2] * 3)
            )
            prior = ReceivingCommandActor(weights["observation_mean"], weights["observation_scale"])
            prior.load_state_dict(weights, strict=True)
            prior.eval().requires_grad_(False)
            actor = ReceivingTaskCommandActor(prior, task_mean, task_scale)
        else:
            if (
                parent_state.get("schema") != "soccer.receiving_task_command_dagger.v2"
                or parent_state.get("cumulative_prior_steps") != 800
                or parent_state.get("policy_domain") != ReceivingTaskCommandActor.policy_domain
            ):
                raise ValueError("bounded second DAgger round needs exact actual first task parent")
            base = ReceivingCommandActor(
                weights["prior.observation_mean"], weights["prior.observation_scale"]
            )
            actor = ReceivingTaskCommandActor(base, weights["task_mean"], weights["task_scale"])
            actor.load_state_dict(weights, strict=True)
            prior = copy.deepcopy(actor).eval().requires_grad_(False)
    # No optimizer update until EVERY consumed old/own state proves identity.
    with torch.no_grad():
        for x in (old_x, own_x):
            for start in range(0, len(x), 2048):
                part = x[start : start + 2048]
                reference = prior(part[:, :112]) if continuation_round == 1 else prior(part)
                if not torch.equal(actor(part), reference):
                    raise ValueError("continuation must preserve parent outputs exactly before updates")
    original = {k: v.clone() for k, v in actor.named_parameters()}
    inherited = copy.deepcopy(parent_state["optimizer"])
    previous_steps = [400] * 6 if continuation_round == 1 else [800] * 6 + [400]
    previous_tensors = len(previous_steps)
    if (
        len(inherited["param_groups"]) != 1
        or inherited["param_groups"][0]["params"] != list(range(previous_tensors))
        or set(inherited["state"]) != set(range(previous_tensors))
        or [float(inherited["state"][i]["step"]) for i in range(previous_tensors)] != previous_steps
    ):
        raise ValueError("all exact parent Adam histories required; no optimizer reset")
    inherited["param_groups"][0]["params"] = list(range(7))
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.0003)
    optimizer.load_state_dict(inherited)
    generator = torch.Generator(device="cpu")
    generator.set_state(parent_state["batch_rng"].clone())
    rng_initial = generator.get_state().clone()
    radius = torch.tensor([0.75, 0.5, 1.0], dtype=torch.float32)

    def mse(x: torch.Tensor, y: torch.Tensor) -> float:
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(x), 2048):
                total += float(
                    ((actor(x[start : start + 2048]) - y[start : start + 2048]) / radius)
                    .square()
                    .sum()
                )
        return total / (len(x) * 3)

    def metrics() -> dict[str, float]:
        return dict(
            old_train_mse=mse(old_x[train_ids], old_y[train_ids]),
            old_validation_mse=mse(old_x[val_ids], old_y[val_ids]),
            own_advice_mse=mse(own_x, own_advice),
        )

    before = metrics()
    actor.train()
    for step in range(steps):
        old_ids = train_ids[torch.randint(len(train_ids), (128,), generator=generator)]
        own_ids = torch.cat(
            (
                critical_ids[torch.randint(len(critical_ids), (64,), generator=generator)],
                torch.randint(len(own_x), (64,), generator=generator),
            )
        )
        x = torch.cat((old_x[old_ids], own_x[own_ids]))
        y = torch.cat((old_y[old_ids], own_advice[own_ids]))
        optimizer.zero_grad(set_to_none=True)
        loss = ((actor(x) - y) / radius).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite DAgger continuation loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if on_step is not None:
            on_step(step + 1, float(loss.detach()), float(norm))
    actor.eval().requires_grad_(False)
    changed = [k for k, v in actor.named_parameters() if not torch.equal(v, original[k])]
    final_optimizer = optimizer.state_dict()
    observed_steps = [float(final_optimizer["state"][i]["step"]) for i in range(7)]
    if (
        observed_steps
        != [400 * continuation_round + steps] * 6 + [400 * (continuation_round - 1) + steps]
        or len(changed) != 7
        or any(not torch.isfinite(v).all() for v in actor.parameters())
    ):
        raise ValueError(
            "all7 tensors must update with the declared inherited Adam counters"
        )
    return dict(
        actor=actor,
        optimizer=final_optimizer,
        batch_rng=generator.get_state(),
        initial_batch_rng=rng_initial,
        optimizer_updates=steps,
        cumulative_prior_steps=400 * continuation_round + steps,
        optimizer_parameter_steps=observed_steps,
        changed_parameter_tensors=changed,
        exact_parent_function_before_updates=True,
        own_critical_rows=len(critical_ids),
        minibatch_old_rows=128,
        minibatch_own_critical_rows=64,
        minibatch_own_all_rows=64,
        before=before,
        after=metrics(),
    )
