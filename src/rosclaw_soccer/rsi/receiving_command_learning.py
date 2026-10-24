"""Finite supervised command fitting. Admission/physics are outside this module."""

from collections.abc import Callable
from typing import Any

import torch

from rosclaw_soccer.rsi.receiving_command_actor import ReceivingCommandActor

NORMALIZATION_FLOOR = (
    (0.2,) * 3
    + (0.1,) * 3
    + (0.2,) * 3
    + (0.2,) * 2
    + (0.1,) * 29
    + (0.5,) * 29
    + (0.2,) * 29
    + (0.25,) * 3
    + (0.5,) * 3
    + (0.25,) * 3
    + (0.2,) * 3
    + (0.5,) * 2
)


def initial_command_reference(
    mean: torch.Tensor,
    scale: torch.Tensor,
    *,
    seed: int,
    x: torch.Tensor,
    y: torch.Tensor,
    training_ids: torch.Tensor,
    expected_initial_mse: float,
) -> ReceivingCommandActor:
    """Same architecture/normalizer/seed BEFORE optimizer changes, no fit."""
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("exact original BC initialization seed required")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = ReceivingCommandActor(mean, scale)
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(training_ids), 2048):
            part = training_ids[start : start + 2048]
            total += float(
                ((actor(x[part]) - y[part]) / torch.tensor([0.75, 0.5, 1.0], dtype=torch.float32))
                .square()
                .sum()
            )
    if not len(training_ids) or total / (len(training_ids) * 3) != expected_initial_mse:
        raise ValueError("seed reconstruction must reproduce recorded pre-optimizer MSE exactly")
    return actor.eval().requires_grad_(False)


def fit_command_bc(
    x: torch.Tensor,
    y: torch.Tensor,
    training: torch.Tensor,
    validation: torch.Tensor,
    *,
    seed: int,
    steps: int = 400,
    on_step: Callable[[int, float, float], None] | None = None,
) -> dict[str, Any]:
    if (
        x.ndim != 2
        or x.shape[1] != 112
        or not 2 <= len(x) <= 38400
        or y.shape != (len(x), 3)
        or x.dtype != torch.float32
        or y.dtype != x.dtype
        or x.device.type != "cpu"
        or y.device != x.device
        or x.requires_grad
        or y.requires_grad
        or not torch.isfinite(x).all()
        or not torch.isfinite(y).all()
        or any(
            m.shape != (len(x),) or m.dtype != torch.bool or m.device != x.device
            for m in (training, validation)
        )
        or not training.any()
        or not validation.any()
        or (training & validation).any()
        or ((y < torch.tensor([-0.5, -0.5, -1])) | (y > torch.tensor([1.0, 0.5, 1.0]))).any()
        or type(seed) is not int
        or not 0 <= seed < 2**31
        or type(steps) is not int
        or not 1 <= steps <= 400
    ):
        raise ValueError(
            "finite CPU labels, disjoint nonempty TRAIN/validation masks and bounded fit required"
        )
    train_ids = torch.nonzero(training, as_tuple=True)[0]
    val_ids = torch.nonzero(validation, as_tuple=True)[0]
    mean = x[train_ids].mean(0)
    # Do not amplify near-constant teacher joint/gravity channels by 1000x
    # when a student first leaves the teacher manifold. Fixed physical-unit
    # floors are declared before fitting; no validation data chooses them.
    floor = torch.tensor(NORMALIZATION_FLOOR, dtype=torch.float32)
    scale = torch.maximum(x[train_ids].std(0, unbiased=False), floor)
    radius = torch.tensor([0.75, 0.5, 1.0], dtype=torch.float32)

    def mse(actor: ReceivingCommandActor, ids: torch.Tensor) -> float:
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(ids), 2048):
                part = ids[start : start + 2048]
                total += float(((actor(x[part]) - y[part]) / radius).square().sum())
        return total / (len(ids) * 3)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = ReceivingCommandActor(mean, scale)
    original = {k: v.clone() for k, v in actor.named_parameters()}
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.0003)
    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    before = dict(train_mse=mse(actor, train_ids), validation_mse=mse(actor, val_ids))
    actor.train()
    metrics = []
    for step in range(steps):
        chosen = train_ids[torch.randint(len(train_ids), (256,), generator=generator)]
        optimizer.zero_grad(set_to_none=True)
        loss = ((actor(x[chosen]) - y[chosen]) / radius).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite BC loss; never admit candidate")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if on_step is not None:
            on_step(step + 1, float(loss.detach()), float(norm))
        if (step + 1) % 50 == 0 or step + 1 == steps:
            metrics.append(
                dict(step=step + 1, loss=float(loss.detach()), gradient_norm=float(norm))
            )
    actor.eval().requires_grad_(False)
    changed = [k for k, v in actor.named_parameters() if not torch.equal(v, original[k])]
    if len(changed) != 6 or any(not torch.isfinite(v).all() for v in actor.parameters()):
        raise ValueError("all six command-policy tensors must actually update finitely")
    return dict(
        actor=actor,
        optimizer=optimizer.state_dict(),
        batch_rng=generator.get_state(),
        optimizer_updates=steps,
        changed_parameter_tensors=changed,
        before=before,
        after=dict(train_mse=mse(actor, train_ids), validation_mse=mse(actor, val_ids)),
        training_rows=len(train_ids),
        validation_rows=len(val_ids),
        metrics=metrics,
    )
