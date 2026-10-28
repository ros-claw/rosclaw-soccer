"""Group-held-out calibrated nonfoot-contact forecasting; not motor control."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.rsi.condition_hazard_calibration import (
    calibrate_negative_maxima,
    condition_hazard_summary,
)
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_train_continuous_contact_forecast import split_weights


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("features", "audit", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("features-hash", "audit-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    manifest, npz, audit = (
        a.features / "complete.json",
        a.features / "continuous.npz",
        a.audit / "complete.json",
    )
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() for p in (manifest, npz, audit))
        or digest(manifest) != a.features_hash
        or digest(audit) != a.audit_hash
    ):
        raise ValueError("exact audited physical features and unused external output required")
    data, verification = json.loads(manifest.read_text()), json.loads(audit.read_text())
    if (
        data.get("status") != "RELAY_CONTINUOUS_CONTACT_DATA_NOT_TRAINED"
        or verification.get("status")
        != "RELAY_CONTINUOUS_CONTACT_FINITE_DIFFERENCE_AUDIT_NOT_CONTROL"
        or verification.get("features_hash") != a.features_hash
        or verification.get("all_labels_independently_recomputed") is not True
        or verification.get("all_features_independently_finite_differenced") is not True
        or digest(npz) != data["data_hash"]
    ):
        raise ValueError("independently audited causal source data required")
    with np.load(npz, allow_pickle=False) as f:
        x = f["features"].astype(np.float32)
        y, group, condition = (f[k].copy() for k in ("labels", "context", "condition"))
    n = data["samples"]
    if (
        x.shape != (n, 12)
        or any(v.shape != (n,) for v in (y, group, condition))
        or not np.isfinite(x).all()
        or not np.isin(y, (0, 1, 2)).all()
        or np.bincount(y, minlength=3).tolist() != data["class_counts"]
    ):
        raise ValueError("exact finite original twelve-feature dataset required")
    contexts = sorted(int(v) for v in set(group.tolist()))
    if len(contexts) != 12:
        raise ValueError("all twelve original source context groups required")
    for cid in np.unique(condition):
        if len(np.unique(group[condition == cid])) != 1:
            raise ValueError("a condition cannot cross context groups")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    a.output.mkdir(parents=True)
    rows = []
    for fold, heldout in enumerate(contexts):
        calibration_contexts = [contexts[(fold + i) % len(contexts)] for i in (1, 2, 3, 4)]
        train = ~np.isin(group, [heldout, *calibration_contexts])
        calibration, test = np.isin(group, calibration_contexts), group == heldout
        weights = split_weights(y[train])
        center = x[train].mean(axis=0)
        scale = np.maximum(x[train].std(axis=0), 0.05)
        z = np.clip((x - center) / scale, -8, 8).astype(np.float32)
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
                logits = model(torch.from_numpy(z)).numpy()
                risk = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 2]
            if not np.isfinite(logits).all() or not torch.isfinite(loss):
                raise ValueError("finite training and predicted risk required")
            cal_rows = condition_hazard_summary(
                risk[calibration], y[calibration], condition[calibration]
            )
            threshold, negatives = calibrate_negative_maxima(cal_rows, alpha=0.25)
            test_rows = condition_hazard_summary(risk[test], y[test], condition[test])
            for row in test_rows:
                row["alarm"] = row["max_risk"] > threshold
            checkpoint = a.output / f"context-{heldout}-seed-{seed}.npz"
            np.savez_compressed(
                checkpoint,
                center=center,
                scale=scale,
                weight1=model[0].weight.detach().numpy(),
                bias1=model[0].bias.detach().numpy(),
                weight2=model[2].weight.detach().numpy(),
                bias2=model[2].bias.detach().numpy(),
                heldout_indices=np.flatnonzero(test),
                heldout_logits=logits[test],
                threshold=threshold,
            )
            rows.append(
                dict(
                    heldout_context=heldout,
                    calibration_contexts=calibration_contexts,
                    seed=seed,
                    threshold=threshold,
                    calibration_negative_conditions=negatives,
                    insufficient_calibration=threshold == 1.0,
                    training_contexts=[
                        c for c in contexts if c != heldout and c not in calibration_contexts
                    ],
                    test_conditions=test_rows,
                    checkpoint=checkpoint.name,
                    checkpoint_hash=digest(checkpoint),
                )
            )
    if (
        digest(manifest) != a.features_hash
        or digest(audit) != a.audit_hash
        or digest(npz) != data["data_hash"]
    ):
        raise ValueError("all source evidence must remain unchanged")
    summaries = {}
    for seed in (17, 43, 71):
        folds = [r for r in rows if r["seed"] == seed]
        conditions = [v for r in folds for v in r["test_conditions"]]
        positive = [v for v in conditions if v["contains_nonfoot"]]
        negative = [v for v in conditions if not v["contains_nonfoot"]]
        summaries[str(seed)] = dict(
            conditions=len(conditions),
            positive_conditions=len(positive),
            positive_detected=sum(v["alarm"] for v in positive),
            negative_conditions=len(negative),
            negative_with_alarm=sum(v["alarm"] for v in negative),
            folds_without_calibration=sum(r["insufficient_calibration"] for r in folds),
        )
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CALIBRATED_CONTACT_WARNING_GROUP_HELDOUT_NOT_CONTROL",
            source_head=a.expected_head,
            features_hash=a.features_hash,
            audit_hash=a.audit_hash,
            contexts=contexts,
            split="one test context, four disjoint calibration contexts, seven training contexts",
            calibration="condition maximum risk, negative-only order statistic, alpha=0.25",
            calibration_is_not_population_guarantee=True,
            summaries=summaries,
            rows=rows,
            holdout_is_within_development_source=True,
            physical_control_authorized=False,
            independent_new_physics=False,
            online_rl=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )
    print(json.dumps(summaries))


if __name__ == "__main__":
    main()
