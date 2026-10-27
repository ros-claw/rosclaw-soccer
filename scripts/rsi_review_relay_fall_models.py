"""Summarize all held-out models; exploratory thresholds do not authorize control."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def confusion(prediction: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    return dict(
        tp=int((prediction & truth).sum()),
        fn=int((~prediction & truth).sum()),
        fp=int((prediction & ~truth).sum()),
        tn=int((~prediction & ~truth).sum()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-hash", required=True)
    parser.add_argument(
        "--pair", nargs=3, action="append", required=True, metavar=("TRAIN", "AUDIT", "AUDIT_HASH")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    dp, data_file = args.dataset / "complete.json", args.dataset / "dataset.npz"
    if (
        len(args.pair) != 3
        or args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or dp.is_symlink()
        or data_file.is_symlink()
        or digest(dp) != args.dataset_hash
    ):
        raise ValueError("three audited model shards and exact dataset required")
    dataset = json.loads(dp.read_text())
    if digest(data_file) != dataset["dataset_hash"]:
        raise ValueError("exact dataset hash required")
    with np.load(data_file, allow_pickle=False) as f:
        x, y, groups, episodes = (
            f["features"].copy(),
            f["label"].copy(),
            f["group"].copy() // 2,
            f["episode"].copy(),
        )
    all_contexts = sorted(int(v) for v in set(groups))
    if len(all_contexts) != 12 or x.shape != (len(y), 104) or y.dtype != np.bool_:
        raise ValueError("all twelve held-out source contexts required")
    rows, pins = [], []
    seen = set()
    for training_name, audit_name, expected_audit_hash in args.pair:
        train_root, audit_file = Path(training_name), Path(audit_name)
        source_file = train_root / "complete.json"
        if (
            source_file.is_symlink()
            or audit_file.is_symlink()
            or digest(audit_file) != expected_audit_hash
        ):
            raise ValueError("exact independent audit hash required")
        source_hash = digest(source_file)
        source, audit = (json.loads(p.read_text()) for p in (source_file, audit_file))
        shard = source.get("shard")
        if (
            type(shard) is not int
            or shard not in (0, 1, 2)
            or shard in seen
            or source.get("status") != "RELAY_FALL_SUPERVISED_CONTEXT_HOLDOUT_NOT_CONTROL"
            or source.get("dataset_hash") != args.dataset_hash
            or audit.get("status") != "RELAY_FALL_PREDICTIONS_REPLAYED_NOT_CERTIFIED"
            or audit.get("source_hash") != source_hash
            or audit.get("dataset_hash") != args.dataset_hash
        ):
            raise ValueError("all distinct matching trained and replayed shards required")
        seen.add(shard)
        pins.append(dict(source_hash=source_hash, audit_hash=expected_audit_hash, shard=shard))
        for row in source["rows"]:
            context, seed = row["heldout_context"], row["seed"]
            mask = groups == context
            checkpoint = train_root / row["checkpoint"]
            if (
                checkpoint.is_symlink()
                or checkpoint.resolve().parent != train_root.resolve()
                or digest(checkpoint) != row["checkpoint_hash"]
            ):
                raise ValueError("exact own checkpoint required")
            with np.load(checkpoint, allow_pickle=False) as f:
                z = np.clip((x[mask] - f["center"]) / f["scale"], -10, 10).astype(np.float32)
                logits = (
                    np.tanh(z @ f["weight1"].T + f["bias1"]) @ f["weight2"].T + f["bias2"]
                ).ravel()
            if not np.isfinite(logits).all():
                raise ValueError("finite replayed prediction required")
            probability = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
            actual = y[mask]
            standard = confusion(probability >= 0.5, actual)
            if any(row[k] != v for k, v in standard.items()):
                raise ValueError("fixed half-threshold results must reproduce")
            rows.append(
                dict(
                    context=context,
                    seed=seed,
                    samples=int(mask.sum()),
                    positive_samples=int(actual.sum()),
                    episodes=len(set(episodes[mask].tolist())),
                    threshold_050=standard,
                    threshold_020_exploratory=confusion(probability >= 0.2, actual),
                    threshold_010_exploratory=confusion(probability >= 0.1, actual),
                )
            )
    if seen != {0, 1, 2} or {(r["context"], r["seed"]) for r in rows} != {
        (c, s) for c in all_contexts for s in (17, 43, 71)
    }:
        raise ValueError("whole context and seed evaluation required")
    height = x[:, 3]
    if digest(data_file) != dataset["dataset_hash"]:
        raise ValueError("unchanged data required")
    clean(args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        dict(
            status="RELAY_FALL_HELDOUT_REVIEW_NOT_CONTROL",
            source_head=args.expected_head,
            dataset_hash=args.dataset_hash,
            sources=pins,
            rows=rows,
            exploratory_height_baseline_070=confusion(height < 0.70, y),
            exploratory_height_baseline_075=confusion(height < 0.75, y),
            correlated_control_samples_not_independent_games=True,
            exploratory_thresholds_not_selection_or_authorization=True,
            physics_evaluated_as_controller=False,
            safety_certificate=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
