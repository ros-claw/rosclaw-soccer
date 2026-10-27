"""Group-held-out 100ms contact forecasting; never a motor controller."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def split_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=3)
    if y.ndim != 1 or len(counts) != 3 or (counts == 0).any():
        raise ValueError("all three observed classes required in training fold")
    weights = np.sqrt(len(y) / (3 * counts)).astype(np.float32)
    return weights / weights.mean()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("features", "audit", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("features-hash", "audit-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    fp, dp, ap = (
        a.features / "complete.json",
        a.features / "continuous.npz",
        a.audit / "complete.json",
    )
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() for p in (fp, dp, ap))
        or digest(fp) != a.features_hash
        or digest(ap) != a.audit_hash
    ):
        raise ValueError("exact audited continuous data and unused external output required")
    receipt, audit = json.loads(fp.read_text()), json.loads(ap.read_text())
    if (
        receipt.get("status") != "RELAY_CONTINUOUS_CONTACT_DATA_NOT_TRAINED"
        or audit.get("status") != "RELAY_CONTINUOUS_CONTACT_FINITE_DIFFERENCE_AUDIT_NOT_CONTROL"
        or audit.get("features_hash") != a.features_hash
        or audit.get("all_labels_independently_recomputed") is not True
        or audit.get("all_features_independently_finite_differenced") is not True
        or digest(dp) != receipt["data_hash"]
        or audit.get("samples") != receipt["samples"]
    ):
        raise ValueError("complete independent causal continuous data audit required")
    with np.load(dp, allow_pickle=False) as f:
        x = f["features"].astype(np.float32)
        y, group, condition, control = (
            f[k].copy() for k in ("labels", "context", "condition", "control")
        )
    n = receipt["samples"]
    if (
        x.shape != (n, 12)
        or any(v.shape != (n,) for v in (y, group, condition, control))
        or not np.isfinite(x).all()
        or not np.isin(y, (0, 1, 2)).all()
        or np.bincount(y, minlength=3).tolist() != receipt["class_counts"]
    ):
        raise ValueError("finite audited features and three-class labels required")
    contexts = sorted(int(c) for c in set(group.tolist()))
    if len(contexts) < 2:
        raise ValueError("multiple original source groups required")
    for c in contexts:
        split_weights(y[group != c])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            features_hash=a.features_hash,
            audit_hash=a.audit_hash,
            contexts=contexts,
            seeds=[17, 43, 71],
            horizon_controls=5,
            optimizer_steps=200,
            model="12-32-tanh-3",
            weighting="train-only normalized sqrt inverse frequency",
            split="entire original passer context held out including all sides and time windows",
            decision="argmax",
            independent_conditions=receipt["conditions"],
            samples=n,
            online_rl=False,
            physical_control_authorized=False,
            hardware_authorized=False,
        ),
    )
    rows = []
    for c in contexts:
        train, test = group != c, group == c
        center = x[train].mean(axis=0)
        scale = np.maximum(x[train].std(axis=0), 0.05)
        z = np.clip((x - center) / scale, -8, 8).astype(np.float32)
        weights = split_weights(y[train])
        xt, yt = torch.from_numpy(z[train]), torch.from_numpy(y[train].astype(np.int64))
        for seed in (17, 43, 71):
            torch.manual_seed(seed)
            model = torch.nn.Sequential(
                torch.nn.Linear(12, 32), torch.nn.Tanh(), torch.nn.Linear(32, 3)
            )
            optim = torch.optim.Adam(model.parameters(), lr=0.003)
            for _ in range(200):
                loss = torch.nn.functional.cross_entropy(
                    model(xt), yt, weight=torch.from_numpy(weights)
                )
                optim.zero_grad()
                loss.backward()
                optim.step()
            with torch.no_grad():
                logits = model(torch.from_numpy(z[test])).numpy()
            if not np.isfinite(logits).all() or not bool(torch.isfinite(loss)):
                raise ValueError("finite trained prediction required")
            cp = a.output / f"context-{c}-seed-{seed}.npz"
            np.savez_compressed(
                cp,
                center=center,
                scale=scale,
                class_weights=weights,
                weight1=model[0].weight.detach().numpy(),
                bias1=model[0].bias.detach().numpy(),
                weight2=model[2].weight.detach().numpy(),
                bias2=model[2].bias.detach().numpy(),
                heldout_indices=np.flatnonzero(test),
                heldout_logits=logits,
            )
            pred = logits.argmax(axis=1)
            truth = y[test]
            cm = np.zeros((3, 3), dtype=np.int64)
            np.add.at(cm, (truth, pred), 1)
            rows.append(
                dict(
                    context=c,
                    seed=seed,
                    training_samples=int(train.sum()),
                    heldout_samples=int(test.sum()),
                    confusion=cm.tolist(),
                    checkpoint=cp.name,
                    checkpoint_hash=digest(cp),
                )
            )
    if (
        digest(fp) != a.features_hash
        or digest(ap) != a.audit_hash
        or digest(dp) != receipt["data_hash"]
    ):
        raise ValueError("data changed during training")
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="RELAY_CONTINUOUS_CONTACT_FORECAST_NOT_CONTROL",
            source_head=a.expected_head,
            features_hash=a.features_hash,
            audit_hash=a.audit_hash,
            contexts=contexts,
            rows=rows,
            optimizer_updates=len(rows) * 200,
            physical_control_authorized=False,
            online_rl=False,
            independent_physics_evaluation=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
