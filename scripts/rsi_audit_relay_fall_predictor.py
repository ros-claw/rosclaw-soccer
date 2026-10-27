"""Independently read trained weights and replay grouped fall predictions in NumPy."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def predict(checkpoint: Path, x: np.ndarray) -> np.ndarray:
    with np.load(checkpoint, allow_pickle=False) as f:
        center, scale = f["center"], f["scale"]
        w1, b1, w2, b2 = f["weight1"], f["bias1"], f["weight2"], f["bias2"]
    if (
        x.ndim != 2
        or x.shape[1] != 104
        or center.shape != (104,)
        or scale.shape != (104,)
        or w1.shape != (64, 104)
        or b1.shape != (64,)
        or w2.shape != (1, 64)
        or b2.shape != (1,)
        or not all(np.isfinite(a).all() for a in (x, center, scale, w1, b1, w2, b2))
        or (scale < 0.01).any()
    ):
        raise ValueError("finite exact fall classifier dimensions required")
    z = np.clip((x - center) / scale, -10, 10).astype(np.float32)
    logits = (np.tanh(z @ w1.T + b1) @ w2.T + b2).ravel()
    return np.asarray(logits >= 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "dataset", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-hash", "dataset-hash", "expected-head"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    sp, dp, data_file = (
        args.source / "complete.json",
        args.dataset / "complete.json",
        args.dataset / "dataset.npz",
    )
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() for p in (sp, dp, data_file))
        or digest(sp) != args.source_hash
        or digest(dp) != args.dataset_hash
    ):
        raise ValueError("exact source, data and unused external output required")
    source, dataset = (json.loads(p.read_text()) for p in (sp, dp))
    if (
        source["status"] != "RELAY_FALL_SUPERVISED_CONTEXT_HOLDOUT_NOT_CONTROL"
        or source["dataset_hash"] != args.dataset_hash
        or digest(data_file) != dataset["dataset_hash"]
        or len(source["rows"]) != 12
        or source["optimizer_updates"] != 2400
    ):
        raise ValueError("exact complete independently grouped model shard required")
    with np.load(data_file, allow_pickle=False) as f:
        x, y, contexts = f["features"].copy(), f["label"].copy(), f["group"].copy() // 2
    all_contexts = sorted(int(c) for c in set(contexts))
    shard = source["shard"]
    if type(shard) is not int or shard not in (0, 1, 2) or len(all_contexts) != 12:
        raise ValueError("all twelve context groups and explicit shard required")
    expected = {(c, seed) for c in all_contexts[shard * 4 : shard * 4 + 4] for seed in (17, 43, 71)}
    if {(r["heldout_context"], r["seed"]) for r in source["rows"]} != expected:
        raise ValueError("exact heldout contexts and three seeds required")
    results = []
    for row in source["rows"]:
        path = args.source / row["checkpoint"]
        if (
            path.is_symlink()
            or path.resolve().parent != args.source.resolve()
            or digest(path) != row["checkpoint_hash"]
        ):
            raise ValueError("exact confined trained weight file required")
        mask = contexts == row["heldout_context"]
        result = predict(path, x[mask])
        truth = y[mask]
        counts = dict(
            tp=int((result & truth).sum()),
            fn=int((~result & truth).sum()),
            fp=int((result & ~truth).sum()),
            tn=int((~result & ~truth).sum()),
        )
        if any(row[key] != value for key, value in counts.items()):
            raise ValueError("own NumPy prediction counts differ")
        results.append(dict(context=row["heldout_context"], seed=row["seed"], **counts))
    if (
        digest(sp) != args.source_hash
        or digest(dp) != args.dataset_hash
        or digest(data_file) != dataset["dataset_hash"]
    ):
        raise ValueError("input changed during prediction replay")
    clean(args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        dict(
            status="RELAY_FALL_PREDICTIONS_REPLAYED_NOT_CERTIFIED",
            source_head=args.expected_head,
            source_hash=args.source_hash,
            dataset_hash=args.dataset_hash,
            rows=results,
            independent_numpy_inference=True,
            independent_physics_reintegrated=False,
            controller_interface=False,
            safety_certificate=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
