"""Independently replay all offline grouped selectors in NumPy; no physics."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.relay_balance_selector import select_with_parent_fallback
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "dataset", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-hash", "dataset-hash", "expected-head"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    sp, rp, dp = (
        args.source / "complete.json",
        args.dataset / "complete.json",
        args.dataset / "dataset.npz",
    )
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() for p in (sp, rp, dp))
        or digest(sp) != args.source_hash
        or digest(rp) != args.dataset_hash
    ):
        raise ValueError("exact trained model, data and unused output required")
    source, receipt = (json.loads(p.read_text()) for p in (sp, rp))
    if (
        source.get("status") != "RELAY_BALANCE_OFFLINE_SELECTOR_HOLDOUT_NOT_PHYSICS"
        or source.get("dataset_hash") != args.dataset_hash
        or digest(dp) != receipt["dataset_hash"]
        or source.get("optimizer_updates") != len(source.get("rows", [])) * 500
    ):
        raise ValueError("complete grouped offline training required")
    with np.load(dp, allow_pickle=False) as f:
        x, groups = f["features"].copy(), f["context"].copy()
        success, failure, kick = f["success"].copy(), f["failure"].copy(), f["kick"].copy()
    contexts = sorted(int(c) for c in set(groups))
    if (
        source["contexts"] != contexts
        or {(r["context"], r["seed"]) for r in source["rows"]}
        != {(c, s) for c in contexts for s in (17, 43, 71)}
        or len(source["rows"]) != len(contexts) * 3
    ):
        raise ValueError("every held-out context and all three seeds required")
    results = []
    for row in source["rows"]:
        context = row["context"]
        train, test = groups != context, groups == context
        path = args.source / row["checkpoint"]
        if (
            path.is_symlink()
            or path.resolve().parent != args.source.resolve()
            or digest(path) != row["checkpoint_hash"]
        ):
            raise ValueError("exact confined model checkpoint required")
        with np.load(path, allow_pickle=False) as f:
            center, scale = f["center"], f["scale"]
            support = f["support"]
            w1, b1, w2, b2 = f["weight1"], f["bias1"], f["weight2"], f["bias2"]
            if (
                center.shape != (17,)
                or scale.shape != (17,)
                or w1.shape != (32, 17)
                or b1.shape != (32,)
                or w2.shape != (4, 32)
                or b2.shape != (4,)
                or not all(np.isfinite(v).all() for v in (center, scale, support, w1, b1, w2, b2))
                or not np.array_equal(center, x[train].mean(axis=0))
                or not np.array_equal(scale, np.maximum(x[train].std(axis=0), 0.05))
            ):
                raise ValueError("exact train-only normalization and finite model required")
            z = np.clip((x - center) / scale, -8, 8).astype(np.float32)
            if not np.array_equal(support, z[train]):
                raise ValueError("exact train-only support required")
            values = np.tanh(z[test] @ w1.T + b1) @ w2.T + b2
        indices = np.flatnonzero(test)
        if len(indices) != len(row["choices"]) or row["training_decisions"] != int(train.sum()):
            raise ValueError("exact grouped holdout size required")
        for local, (index, saved) in enumerate(zip(indices, row["choices"], strict=True)):
            distance = float(np.sqrt(np.mean((support - z[index]) ** 2, axis=1)).min())
            action, reason = select_with_parent_fallback(values[local], distance)
            if (
                saved["sample"] != int(index)
                or saved["action"] != action
                or saved["reason"] != reason
                or abs(saved["support_distance"] - distance) > 1e-6
                or saved["parent_success"] != bool(success[index, 0])
                or saved["chosen_success"] != bool(success[index, action])
                or saved["parent_failure"] != bool(failure[index, 0])
                or saved["chosen_failure"] != bool(failure[index, action])
                or saved["parent_kick"] != bool(kick[index, 0])
                or saved["chosen_kick"] != bool(kick[index, action])
            ):
                raise ValueError("independent grouped choice or outcome differs")
        results.append(
            dict(
                context=context,
                seed=row["seed"],
                heldout_decisions=len(indices),
                choices_replayed=True,
            )
        )
    if (
        digest(sp) != args.source_hash
        or digest(rp) != args.dataset_hash
        or digest(dp) != receipt["dataset_hash"]
    ):
        raise ValueError("sources changed during inference replay")
    clean(args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        dict(
            status="RELAY_BALANCE_SELECTOR_NUMPY_REPLAY_NOT_CONTROL",
            source_head=args.expected_head,
            source_hash=args.source_hash,
            dataset_hash=args.dataset_hash,
            rows=results,
            independent_numpy_inference=True,
            physics_evaluated_as_controller=False,
            safety_certificate=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
