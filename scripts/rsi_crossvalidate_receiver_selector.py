"""Run grouped offline validation; never train on the held-out context."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.receiver_contextual_selector import group_holdout_validation
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "review", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("training-hash", "review-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.training.is_symlink()
        or digest(a.training) != a.training_hash
        or a.review.is_symlink()
        or digest(a.review) != a.review_hash
    ):
        raise ValueError("exact training and review receipts and new external output required")
    training = json.loads(a.training.read_text())
    review = json.loads(a.review.read_text())
    data_path = a.training.parent / "development-data.npz"
    if (
        training.get("status")
        != "RECEIVER_CONTEXT_SELECTOR_TRAINED_DEV_ONLY_NOT_PHYSICALLY_VALIDATED"
        or training.get("inputs", {}).get(str(a.review)) != a.review_hash
        or review.get("status") != "MATCHED_RECEIVER_PHYSICAL_PROBE_NOT_CATCH_CERTIFICATE"
        or data_path.is_symlink()
        or digest(data_path) != training.get("dataset_hash")
        or any(
            r.get(k) is not False
            for r in (training, review)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("bound unpromoted development training data required")
    rows = {r["index"]: r for r in review["rows"]}
    with np.load(data_path, allow_pickle=False) as data:
        x, labels, indices = (data[k].copy() for k in ("observations", "labels", "indices"))
    if (
        len(rows) != 8
        or indices.dtype != np.int64
        or indices.shape != (8,)
        or len(set(indices.tolist())) != 8
        or set(indices.tolist()) != set(rows)
    ):
        raise ValueError("one unique group identity per development sample required")
    groups = np.asarray([rows[int(i)]["context"] for i in indices], dtype=np.int64)
    result = group_holdout_validation(x, labels, groups, steps=200)
    if (
        digest(data_path) != training["dataset_hash"]
        or digest(a.training) != a.training_hash
        or digest(a.review) != a.review_hash
    ):
        raise ValueError("source evidence changed during grouped validation")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            **result,
            source_head=a.expected_head,
            training_hash=a.training_hash,
            review_hash=a.review_hash,
            dataset_hash=training["dataset_hash"],
            trained_selector_weights_used_in_folds=False,
        ),
    )


if __name__ == "__main__":
    main()
