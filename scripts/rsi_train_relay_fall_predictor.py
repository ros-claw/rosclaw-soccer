"""Train grouped offline fall predictors from audited SIM data; no motor authority."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("dataset-hash", "audit-hash", "expected-head"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    rp, ap, dp = (
        args.dataset / "complete.json",
        args.audit / "complete.json",
        args.dataset / "dataset.npz",
    )
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() for p in (rp, ap, dp))
        or digest(rp) != args.dataset_hash
        or digest(ap) != args.audit_hash
    ):
        raise ValueError("exact audited data and unused external output required")
    receipt, audit = (json.loads(p.read_text()) for p in (rp, ap))
    if (
        receipt["status"] != "RELAY_CAUSAL_FALL_SAMPLES_EXPORTED_NOT_TRAINED"
        or audit["status"] != "RELAY_FALL_SAMPLES_READBACK_NOT_REINTEGRATED"
        or audit["dataset_hash"] != args.dataset_hash
        or digest(dp) != receipt["dataset_hash"]
    ):
        raise ValueError("causal data readback required")
    with np.load(dp, allow_pickle=False) as data:
        x, y = data["features"].copy(), data["label"].copy()
        contexts, episodes = data["group"].copy() // 2, data["episode"].copy()
    if (
        x.shape != (len(y), 104)
        or y.dtype != np.bool_
        or not np.isfinite(x).all()
        or len(y) != receipt["samples"]
        or not 0 < y.sum() < len(y)
        or len(set(contexts.tolist())) != 12
    ):
        raise ValueError("finite 104-dimensional measured states and 12 context groups required")
    all_contexts = sorted(int(c) for c in set(contexts))
    selected = all_contexts[args.shard * 4 : args.shard * 4 + 4]
    args.output.mkdir(parents=True)
    write_once(
        args.output / "declaration.json",
        dict(
            dataset_hash=args.dataset_hash,
            audit_hash=args.audit_hash,
            source_head=args.expected_head,
            contexts=selected,
            all_contexts=all_contexts,
            seeds=[17, 43, 71],
            network="104-64-tanh-1",
            optimizer="Adam 0.003, 200 full-batch updates per model",
            split=(
                "leave one initial passer context out; both sides and all four "
                "relay arms held out together"
            ),
            target="B body failure within next 25 controls; right-censored tails excluded",
            input="prior proprioception, relative ball geometry and actual previous target",
            threshold=0.5,
            preprocessor="train-only mean/std, std floor 0.01, clip 10",
            class_weight="negative/positive capped at 20 on train only",
            optimizer_updates_planned=4 * 3 * 200,
            offline=True,
            control_authority=False,
            hardware_authorized=False,
        ),
    )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    for holdout in selected:
        train, test = contexts != holdout, contexts == holdout
        center = x[train].mean(axis=0)
        scale = np.maximum(x[train].std(axis=0), 0.01)
        z = np.clip((x - center) / scale, -10, 10).astype(np.float32)
        xt = torch.from_numpy(z[train])
        yt = torch.from_numpy(y[train, None].astype(np.float32))
        xv, truth = torch.from_numpy(z[test]), y[test]
        n_pos, n_neg = int(yt.sum()), len(yt) - int(yt.sum())
        if not n_pos or not n_neg or not len(truth):
            raise ValueError("both train classes and nonempty held-out group required")
        positive_weight = torch.tensor([min(20.0, n_neg / n_pos)])
        for seed in (17, 43, 71):
            torch.manual_seed(seed)
            model = torch.nn.Sequential(
                torch.nn.Linear(104, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1)
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            for _ in range(200):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(xt), yt, pos_weight=positive_weight
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                probability = torch.sigmoid(model(xv)).numpy().ravel()
            if not np.isfinite(probability).all() or not bool(torch.isfinite(loss)):
                raise ValueError("finite trained loss and held-out predictions required")
            predict = probability >= 0.5
            checkpoint = args.output / f"context-{holdout}-seed-{seed}.npz"
            np.savez_compressed(
                checkpoint,
                center=center,
                scale=scale,
                weight1=model[0].weight.detach().numpy(),
                bias1=model[0].bias.detach().numpy(),
                weight2=model[2].weight.detach().numpy(),
                bias2=model[2].bias.detach().numpy(),
            )
            row = dict(
                heldout_context=holdout,
                seed=seed,
                heldout_samples=int(test.sum()),
                heldout_positive=int(truth.sum()),
                heldout_episodes=int(len(set(episodes[test].tolist()))),
                tp=int((predict & truth).sum()),
                fn=int((~predict & truth).sum()),
                fp=int((predict & ~truth).sum()),
                tn=int((~predict & ~truth).sum()),
                checkpoint=checkpoint.name,
                checkpoint_hash=digest(checkpoint),
            )
            rows.append(row)
            write_once(args.output / f"context-{holdout}-seed-{seed}.json", row)
    if (
        digest(rp) != args.dataset_hash
        or digest(ap) != args.audit_hash
        or digest(dp) != receipt["dataset_hash"]
    ):
        raise ValueError("data changed during training")
    clean(args.expected_head)
    write_once(
        args.output / "complete.json",
        dict(
            status="RELAY_FALL_SUPERVISED_CONTEXT_HOLDOUT_NOT_CONTROL",
            source_head=args.expected_head,
            dataset_hash=args.dataset_hash,
            audit_hash=args.audit_hash,
            shard=args.shard,
            rows=rows,
            optimizer_updates=len(rows) * 200,
            offline=True,
            online_rl=False,
            controller_interface=False,
            safety_certificate=False,
            physics_evaluated_as_controller=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
