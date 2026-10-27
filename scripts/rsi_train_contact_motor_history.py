"""Train causal motor-history contact forecast, paired with frozen foot-only folds."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_train_continuous_contact_forecast import split_weights


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("history", "history-audit", "continuous", "parent", "parent-audit", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "history-hash",
        "history-audit-hash",
        "continuous-hash",
        "parent-hash",
        "parent-audit-hash",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    hp, ha, cp, pp, pa = (
        a.history / "complete.json",
        a.history_audit / "complete.json",
        a.continuous / "complete.json",
        a.parent / "complete.json",
        a.parent_audit / "complete.json",
    )
    checks = (
        (hp, a.history_hash),
        (ha, a.history_audit_hash),
        (cp, a.continuous_hash),
        (pp, a.parent_hash),
        (pa, a.parent_audit_hash),
    )
    hdata, cdata = a.history / "motor-history.npz", a.continuous / "continuous.npz"
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(path.is_symlink() or digest(path) != expected for path, expected in checks)
        or hdata.is_symlink()
        or cdata.is_symlink()
    ):
        raise ValueError("exact audited history, parent and unused output required")
    h, audit, base, parent, parent_audit = (json.loads(path.read_text()) for path, _ in checks)
    if (
        h.get("status") != "RELAY_CONTACT_HISTORY_DATA_NOT_TRAINED"
        or audit.get("status") != "RELAY_CONTACT_HISTORY_READBACK_NOT_CONTROL"
        or audit.get("features_hash") != a.history_hash
        or audit.get("independently_rebuilt_causal_history") is not True
        or h.get("continuous_hash") != a.continuous_hash
        or digest(hdata) != h["feature_hash"]
        or digest(cdata) != base["data_hash"]
        or parent.get("status") != "RELAY_CONTINUOUS_CONTACT_FORECAST_NOT_CONTROL"
        or parent.get("features_hash") != a.continuous_hash
        or parent_audit.get("status") != "RELAY_CONTINUOUS_CONTACT_NUMPY_AUDITED_NOT_CONTROL"
        or parent_audit.get("source_hash") != a.parent_hash
    ):
        raise ValueError("fully audited aligned history and frozen foot-only parent required")
    with np.load(hdata, allow_pickle=False) as f:
        x = f["features"].copy()
    with np.load(cdata, allow_pickle=False) as f:
        y, groups = f["labels"].copy(), f["context"].copy()
    n = base["samples"]
    if (
        x.shape != (n, 308)
        or x.dtype != np.float32
        or y.shape != (n,)
        or groups.shape != (n,)
        or not np.isfinite(x).all()
        or not np.isin(y, (0, 1, 2)).all()
    ):
        raise ValueError("complete finite matched data required")
    contexts = sorted(int(c) for c in set(groups.tolist()))
    parent_rows = {(r["context"], r["seed"]): r for r in parent["rows"]}
    if (
        parent["contexts"] != contexts
        or len(parent_rows) != len(contexts) * 3
        or set(parent_rows) != {(c, s) for c in contexts for s in (17, 43, 71)}
    ):
        raise ValueError("all matched parent folds and seeds required")
    for c in contexts:
        split_weights(y[groups != c])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            history_hash=a.history_hash,
            history_audit_hash=a.history_audit_hash,
            continuous_hash=a.continuous_hash,
            parent_hash=a.parent_hash,
            parent_audit_hash=a.parent_audit_hash,
            contexts=contexts,
            seeds=[17, 43, 71],
            model="308-32-tanh-3",
            optimizer_steps_per_fold=200,
            learning_rate=0.003,
            split="entire original passer context including all sides/windows",
            no_future_inputs=True,
            online_rl=False,
            physical_control_authorized=False,
            hardware_authorized=False,
        ),
    )
    rows = []
    for context in contexts:
        train, test = groups != context, groups == context
        center, scale = x[train].mean(axis=0), np.maximum(x[train].std(axis=0), 0.05)
        z = np.clip((x - center) / scale, -8, 8).astype(np.float32)
        weights = split_weights(y[train])
        xt, yt = torch.from_numpy(z[train]), torch.from_numpy(y[train].astype(np.int64))
        for seed in (17, 43, 71):
            old = parent_rows[(context, seed)]
            oldcp = a.parent / old["checkpoint"]
            if (
                oldcp.is_symlink()
                or oldcp.resolve().parent != a.parent.resolve()
                or digest(oldcp) != old["checkpoint_hash"]
            ):
                raise ValueError("exact parent fold required")
            with np.load(oldcp, allow_pickle=False) as f:
                if not np.array_equal(f["heldout_indices"], np.flatnonzero(test)):
                    raise ValueError("same original parent held-out windows required")
            torch.manual_seed(seed)
            model = torch.nn.Sequential(
                torch.nn.Linear(308, 32), torch.nn.Tanh(), torch.nn.Linear(32, 3)
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
                raise ValueError("finite trained own history predictions required")
            target = a.output / f"context-{context}-seed-{seed}.npz"
            np.savez_compressed(
                target,
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
            cm = np.zeros((3, 3), np.int64)
            np.add.at(cm, (y[test], logits.argmax(axis=1)), 1)
            rows.append(
                dict(
                    context=context,
                    seed=seed,
                    training_samples=int(train.sum()),
                    heldout_samples=int(test.sum()),
                    confusion=cm.tolist(),
                    parent_checkpoint_hash=old["checkpoint_hash"],
                    checkpoint=target.name,
                    checkpoint_hash=digest(target),
                )
            )
    if (
        any(digest(path) != expected for path, expected in checks)
        or digest(hdata) != h["feature_hash"]
        or digest(cdata) != base["data_hash"]
    ):
        raise ValueError("training inputs changed during execution")
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="RELAY_CONTACT_HISTORY_FORECAST_NOT_CONTROL",
            source_head=a.expected_head,
            history_hash=a.history_hash,
            history_audit_hash=a.history_audit_hash,
            continuous_hash=a.continuous_hash,
            parent_hash=a.parent_hash,
            parent_audit_hash=a.parent_audit_hash,
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
