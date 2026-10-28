"""Fit an unpromoted 140-to-29 torque-residual model on assisted SIM data."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


class ResidualActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(140, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 29)
        )

    def forward(self, x):
        return 20.0 * torch.tanh(self.net(x))


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    active = target.abs().max(dim=1).values > 1e-6
    error = (prediction - target).square().mean(dim=1)
    return dict(
        samples=len(target),
        active_samples=int(active.sum()),
        mse_nm2=float(error.mean()),
        zero_baseline_mse_nm2=float(target.square().mean()),
        active_mse_nm2=float(error[active].mean()) if active.any() else None,
        active_zero_baseline_mse_nm2=float(target[active].square().mean())
        if active.any()
        else None,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--dataset-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    receipt = a.dataset / "complete.json"
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or receipt.is_symlink()
        or digest(receipt) != a.dataset_hash
    ):
        raise ValueError("exact immutable data and new external output required")
    data = json.loads(receipt.read_text())
    if (
        data.get("status") != "RECEIVER_ASSISTED_CAUSAL_TORQUE_DATA_NOT_TRAINED"
        or data.get("feature_dim") != 140
        or data.get("train_contexts") != [2, 3, 5]
        or data.get("validation_contexts") != [7]
        or data.get("actor_kinematics_are_before_step") is not True
        or any(
            data.get(k) is not False
            for k in (
                "future_actions_used",
                "teacher_qualified",
                "training_performed",
                "independent_fresh",
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
            )
        )
        or len(data.get("records", [])) != 24
    ):
        raise ValueError("complete causal unqualified development teacher data required")
    sets = {"train": [], "validation": []}
    identities, pins = set(), []
    for row in data["records"]:
        path = a.dataset / row["path"]
        if (
            row["index"] in identities
            or row["split"] not in sets
            or row["context"] not in (2, 3, 5, 7)
            or (row["context"] == 7) != (row["split"] == "validation")
            or path.is_symlink()
            or path.resolve().parent != a.dataset.resolve()
            or digest(path) != row["hash"]
        ):
            raise ValueError("unique matching grouped source sample required")
        identities.add(row["index"])
        pins.append((path, row["hash"]))
        with np.load(path, allow_pickle=False) as z:
            x, y = z["observations"].copy(), z["applied_residual_nm"].copy()
        if (
            x.shape != (row["samples"], 140)
            or y.shape != (len(x), 29)
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or np.max(np.abs(y)) > 20.0001
        ):
            raise ValueError("finite causal samples and bounded actual torque labels required")
        sets[row["split"]].append((x, y))
    if len(sets["train"]) != 18 or len(sets["validation"]) != 6:
        raise ValueError("all original 18/6 grouped cases required")
    torch.manual_seed(926029)
    torch.set_num_threads(2)
    tensors = {
        s: (
            torch.from_numpy(np.concatenate([x for x, _ in rows])),
            torch.from_numpy(np.concatenate([y for _, y in rows])),
        )
        for s, rows in sets.items()
    }
    train_x, train_y = tensors["train"]
    mean, std = train_x.mean(0), train_x.std(0, unbiased=False).clamp(min=0.01)
    model = ResidualActor()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(926029)
    updates = 0
    for _ in range(12):
        for batch in torch.randperm(len(train_x), generator=generator).split(512):
            x, y = train_x[batch], train_y[batch]
            output = model((x - mean) / std)
            weight = 1.0 + 4.0 * (y.abs().max(dim=1).values > 1e-6).float()
            loss = ((output - y).square().mean(dim=1) * weight).mean()
            if not torch.isfinite(loss):
                raise ValueError("finite teacher BC loss required")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
    model.eval()
    with torch.no_grad():
        scores = {s: metrics(model((x - mean) / std), y) for s, (x, y) in tensors.items()}
    if digest(receipt) != a.dataset_hash or any(digest(path) != h for path, h in pins):
        raise ValueError("source data changed during training")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    checkpoint = a.output / "actor.pt"
    torch.save(
        dict(
            actor=model.state_dict(),
            observation_mean=mean,
            observation_std=std,
            dataset_hash=a.dataset_hash,
            architecture="tanh-256-256-tanh20",
            teacher_qualified=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
        checkpoint,
    )
    write_once(
        a.output / "complete.json",
        dict(
            status="ASSISTED_RECEIVER_TORQUE_BC_NOT_PHYSICALLY_VALIDATED",
            source_head=a.expected_head,
            dataset_hash=a.dataset_hash,
            checkpoint_hash=digest(checkpoint),
            train_contexts=[2, 3, 5],
            validation_contexts=[7],
            seed=926029,
            epochs=12,
            optimizer_updates=updates,
            active_weight=5.0,
            scores=scores,
            validation_is_development=True,
            teacher_qualified=False,
            online_rl=False,
            actual_new_physics=False,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
