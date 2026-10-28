"""Fit a three-seed, grouped-split first-touch outcome model from actual trials."""

import argparse
import json
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-audit", type=Path, required=True)
    p.add_argument("--source-audit-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if (
        a.output.exists()
        or a.source_audit.is_symlink()
        or digest(a.source_audit) != a.source_audit_hash
    ):
        raise ValueError("unused output and exact source audit required")
    source = json.loads(a.source_audit.read_text())
    if (
        source.get("status")
        != "PUBLIC40_CEM_ALL760_RAW_LABELS_AND_REEXECUTION_AUTHENTICATED_NOT_PROMOTED"
        or [r["course_index"] for r in source["results"]] != list(range(40))
        or source.get("promotion_authorized") is not False
    ):
        raise ValueError("exact unpromoted public40 physical audit required")
    pins = dict(source["input_file_hashes"])
    pins[str(a.source_audit)] = a.source_audit_hash
    source_pins(pins)
    check_pins(pins)
    indexed = {}
    pattern = re.compile(r"/course-(\d\d)/candidate-(\d\d)\.json$")
    for name in pins:
        match = pattern.search(name)
        if match:
            key = tuple(map(int, match.groups()))
            if key in indexed:
                raise ValueError("ambiguous physical candidate source")
            indexed[key] = Path(name)

    # Public evaluation indices and grouped model-validation cases stay out of fitting.
    public_test = set(range(0, 40, 5))
    validation = {1, 6, 11, 16, 21, 26, 31, 36}
    if public_test & validation:
        raise ValueError("disjoint grouped splits required")
    train = set(range(40)) - public_test - validation
    features, targets, groups = [], [], []
    for i in sorted(train | validation):
        for j in range(16):
            path = indexed[i, j]
            if path.is_symlink() or digest(path) != pins[str(path)]:
                raise ValueError("exact physical candidate record required")
            row = json.loads(path.read_text())
            if row["course_index"] != i or row["course"] != source["results"][i]["course"]:
                raise ValueError("course identity mismatch")
            if row["experimental_fast_approach"] is not True:
                raise ValueError("candidate must use same measured experimental velocity codec")
            params, outcome = row["parameters"], row["outcome"]
            artifact = path.parent / row["artifact"]["path"]
            if (
                artifact.parent != path.parent
                or artifact.is_symlink()
                or digest(artifact) != row["artifact"]["hash"]
            ):
                raise ValueError("exact confined physical trajectory required")
            features.append(
                [
                    *row["course"],
                    params["speed_mps"],
                    params["entry_gap_m"],
                    params["reference_frame"],
                    params["root_lateral_offset_m"],
                ]
            )
            targets.append(
                [bool(outcome["task_pass"]), bool(outcome["safe"] and row["healthy_full_horizon"])]
            )
            groups.append(i)

    import numpy as np
    import torch
    from torch import nn

    from rosclaw_soccer.rsi.first_touch_outcome_model import FirstTouchOutcomeModel

    torch.set_num_threads(4)
    features = np.asarray(features, np.float32)
    targets = np.asarray(targets, np.float32)
    groups = np.asarray(groups, np.int64)
    if features.shape != (512, 7) or targets.shape != (512, 2) or not np.isfinite(features).all():
        raise ValueError("all 512 finite physical samples required")
    tr = np.isin(groups, list(train))
    va = np.isin(groups, list(validation))
    if tr.sum() != 384 or va.sum() != 128 or (tr & va).any():
        raise ValueError("complete 24/8 state-level split required")
    x, y = torch.from_numpy(features), torch.from_numpy(targets)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            source_audit_hash=a.source_audit_hash,
            training_groups=sorted(train),
            validation_groups=sorted(validation),
            excluded_public_test_groups=sorted(public_test),
            models=3,
            seeds=[981, 982, 983],
            steps_per_model=400,
            method="full-batch Adam binary logits on task pass and strict safety",
            public_development_only=True,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    np.savez_compressed(
        a.output / "data.npz",
        features=features,
        targets=targets,
        groups=groups,
        training=tr,
        validation=va,
    )
    records = []
    for seed in (981, 982, 983):
        torch.manual_seed(seed)
        model = FirstTouchOutcomeModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(400):
            loss = nn.functional.binary_cross_entropy_with_logits(model(x[tr]), y[tr])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pred_train, pred_val = model(x[tr]).sigmoid(), model(x[va]).sigmoid()
            metrics = dict(
                train_brier=((pred_train - y[tr]) ** 2).mean(dim=0).tolist(),
                validation_brier=((pred_val - y[va]) ** 2).mean(dim=0).tolist(),
                validation_accuracy=((pred_val >= 0.5) == y[va].bool())
                .float()
                .mean(dim=0)
                .tolist(),
                validation_positive_rates=y[va].mean(dim=0).tolist(),
            )
        path = a.output / f"model-{seed}.pt"
        torch.save(dict(state_dict=model.state_dict(), seed=seed), path)
        records.append(
            dict(
                seed=seed,
                checkpoint_hash=digest(path),
                final_train_loss=float(loss),
                metrics=metrics,
            )
        )
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="GROUPED_OUTCOME_MODEL_NEEDS_PHYSICS_NOT_SKILL",
            source_head=a.expected_head,
            source_audit_hash=a.source_audit_hash,
            dataset_hash=digest(a.output / "data.npz"),
            records=records,
            optimizer_steps=1200,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
