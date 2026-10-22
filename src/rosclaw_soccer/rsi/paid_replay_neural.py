"""Experimental smooth own288 motor head with soft rehearsal, not narrow cones.

This deliberately trades an exact anchor theorem for measured retention loss.
It must not enter physics without a separate typed loader and native evaluation.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn

FIT_STATUS = "PAID_SOFT_REPLAY_NEURAL_OFFLINE_FIT_NOT_RUNTIME_ADMITTED"


def replay_config(index: int) -> dict[str, Any]:
    if type(index) is not int or index not in (0, 1, 2):
        raise ValueError("three fixed neural replication seeds required")
    return dict(
        protocol="paid.own288.soft_replay_neural.v1",
        seed_index=index,
        seed=2026102209 + index,
        optimizer="Adam",
        updates=6144,
        lr=0.0003,
        final_lr=0.00003,
        active_batch=512,
        rehearsal_batch=512,
        rehearsal_weight=2.0,
        gradient_norm_clip=5.0,
        architecture=[288, 256, 128, 64, 29],
        activation="SiLU",
        train_only_feature_std_floor=0.1,
        relative_bound_rad=0.4,
        final_R2_bound_rad=0.2,
        geometric_support_gate=False,
        parent_frozen=True,
        R2_frozen=True,
        retention="soft_rehearsal_MSE_not_exact_anchor_guarantee",
        runtime_case_lookup=False,
        runtime_frame_lookup=False,
        Fresh_fit_rows=0,
        original_DEV16_fit_rows=0,
        prior_unseen32_is_now_observed_development=True,
        primary_seed_index=0,
        best_seed_selection_authorized=False,
    )


class ReplayMotorHead(nn.Module):
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        if (
            any(
                v.shape != (288,) or v.dtype != torch.float32 or not torch.isfinite(v).all()
                for v in (mean, scale)
            )
            or (scale < 0.1).any()
        ):
            raise ValueError("finite TRAIN-only normalization required")
        self.register_buffer("feature_mean", mean.detach().clone())
        self.register_buffer("feature_scale", scale.detach().clone())
        self.net = nn.Sequential(
            nn.Linear(288, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 29),
        )
        output = self.net[-1]
        if not isinstance(output, nn.Linear):
            raise ValueError("explicit 29-action linear output required")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Dataset validation and the future actor boundary check finite own288;
        # leave this pure tensor path synchronisation-free during GPU fitting.
        return 0.4 * torch.tanh(self.net((features - self.feature_mean) / self.feature_scale))

    def combine(self, features: torch.Tensor, parent_delta: torch.Tensor) -> torch.Tensor:
        return (parent_delta + self.forward(features)).clamp(-0.2, 0.2)


def training_arrays(z: dict[str, np.ndarray]) -> tuple[torch.Tensor, ...]:
    n = len(z["features"])
    if (
        not 1 <= n <= 50000
        or any(
            z[k].shape != (n, d) or z[k].dtype != np.float32 or not np.isfinite(z[k]).all()
            for k, d in (("features", 288), ("target", 29), ("parent_delta", 29))
        )
        or z["active"].shape != (n,)
        or z["active"].dtype != np.bool_
    ):
        raise ValueError("bounded finite causal TRAIN arrays required")
    y, parent, active = z["target"], z["parent_delta"], z["active"]
    if (
        not active.any()
        or active.all()
        or np.abs(y).max() > 0.40002
        or np.abs(parent).max() > 0.2
        or np.abs(parent + y).max() > 0.20003
        or not np.array_equal(active, np.abs(y).max(axis=1) > 1e-6)
    ):
        raise ValueError("active motor differences and genuine rehearsal required")
    return tuple(
        torch.from_numpy(z[k].copy()) for k in ("features", "target", "parent_delta", "active")
    )


def fit_replay_head(
    z: dict[str, np.ndarray],
    index: int,
    device: str,
    progress: Callable[[dict[str, Any]], None],
) -> tuple[ReplayMotorHead, dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    cfg = replay_config(index)
    x, y, parent, active = training_arrays(z)
    # All means and scales derive solely from the reconstructed TRAIN data.
    mean = x.double().mean(0).float()
    scale = x.double().std(0, unbiased=False).float().clamp_min(0.1)
    torch.manual_seed(cfg["seed"])
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg["seed"])
    head = ReplayMotorHead(mean, scale).to(device)
    initial = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    x, y, parent, active = [t.to(device) for t in (x, y, parent, active)]
    positives, negatives = active.nonzero().flatten(), (~active).nonzero().flatten()
    rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 100)
    optimizer = torch.optim.Adam(head.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["updates"], eta_min=cfg["final_lr"]
    )
    nonzero = 0
    for step in range(cfg["updates"]):
        pi = positives[
            torch.randint(len(positives), (cfg["active_batch"],), generator=rng, device=device)
        ]
        ni = negatives[
            torch.randint(len(negatives), (cfg["rehearsal_batch"],), generator=rng, device=device)
        ]
        correction = head(x[torch.cat((pi, ni))])
        lp = (correction[: len(pi)] - y[pi]).square().mean()
        ln = correction[len(pi) :].square().mean()
        loss = lp + cfg["rehearsal_weight"] * ln
        if not torch.isfinite(loss):
            raise ValueError("nonfinite neural objective; preserve partial fit")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            head.parameters(), cfg["gradient_norm_clip"], error_if_nonfinite=True
        )
        nonzero += int(float(norm) > 0)
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 256 == 0:
            progress(
                dict(
                    completed_updates=step + 1,
                    active_loss=float(lp),
                    rehearsal_loss=float(ln),
                    gradient_norm=float(norm),
                    learning_rate=optimizer.param_groups[0]["lr"],
                )
            )
    head = head.cpu().eval()
    changed = [k for k, v in head.state_dict().items() if not torch.equal(v.cpu(), initial[k])]
    if nonzero != cfg["updates"] or not changed or any(k.startswith("feature_") for k in changed):
        raise ValueError("effective network updates and frozen normalization required")
    return (
        head,
        initial,
        dict(
            completed_updates=cfg["updates"],
            nonzero_gradient_updates=nonzero,
            active_forward_rows=cfg["updates"] * cfg["active_batch"],
            rehearsal_forward_rows=cfg["updates"] * cfg["rehearsal_batch"],
            changed_network_tensors=changed,
        ),
        optimizer.state_dict(),
    )


def replay_certificate(head: ReplayMotorHead, z: dict[str, np.ndarray]) -> dict[str, Any]:
    x, y, prior, active = training_arrays(z)
    predicted = []
    with torch.no_grad():
        # The exact deployment batch size, not a large-batch fit-only estimate.
        for row in x:
            predicted.append(head(row.reshape(1, 288))[0])
    correction = torch.stack(predicted)
    if not torch.isfinite(correction).all() or (correction.abs() > 0.4).any():
        raise ValueError("finite bounded CPU batch1 correction required")
    final = (prior + correction).clamp(-0.2, 0.2)
    desired = (prior + y).clamp(-0.2, 0.2)
    return dict(
        CPU_batch1_rows=len(x),
        active_rows=int(active.sum()),
        rehearsal_rows=int((~active).sum()),
        active_correction_MSE=float((correction[active] - y[active]).square().mean()),
        active_final_motor_MSE=float((final[active] - desired[active]).square().mean()),
        rehearsal_correction_MSE=float(correction[~active].square().mean()),
        maximum_rehearsal_motor_change_rad=float((final[~active] - prior[~active]).abs().max()),
        maximum_active_correction_error_rad=float((correction[active] - y[active]).abs().max()),
        maximum_final_R2_residual_rad=float(final.abs().max()),
        exact_retention_claim=False,
        physical_qualification=False,
    )
