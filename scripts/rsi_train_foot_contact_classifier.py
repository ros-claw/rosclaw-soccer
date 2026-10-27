"""Grouped offline classification conditioned on an actual future contact event."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def labels_and_groups(records: list[dict], n: int) -> tuple[np.ndarray, np.ndarray]:
    y, group = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    used = set()
    for row in records:
        if not row["receiver_contact"]:
            if row["feature_index"] is not None:
                raise ValueError("no-contact states must not have fabricated features")
            continue
        i, categories = row["feature_index"], row["first_contact_force_categories"]
        if (
            type(i) is not int
            or i in used
            or not 0 <= i < n
            or not categories
            or any(type(c) is not int or c not in (0, 1, 2) for c in categories)
            or type(row["context"]) is not int
        ):
            raise ValueError("unique recorded force category and original context required")
        used.add(i)
        y[i], group[i] = int(2 in categories), row["context"]
    if used != set(range(n)):
        raise ValueError("all actual contact samples required")
    return y, group


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("features", "audit", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("features-hash", "audit-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    fp, npz, ap = (
        a.features / "complete.json",
        a.features / "foot-entry.npz",
        a.audit / "complete.json",
    )
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() for p in (fp, npz, ap))
        or digest(fp) != a.features_hash
        or digest(ap) != a.audit_hash
    ):
        raise ValueError("exact audited feature data and unused external output required")
    receipt, audit = json.loads(fp.read_text()), json.loads(ap.read_text())
    if (
        receipt.get("status") != "RELAY_FOOT_ENTRY_DATA_EXPORTED_NOT_TRAINED"
        or audit.get("status") != "RELAY_FOOT_ENTRY_FINITE_DIFFERENCE_AUDITED_NOT_CONTROL"
        or audit.get("feature_hash") != a.features_hash
        or audit.get("independent_finite_difference") is not True
        or digest(npz) != receipt["feature_hash"]
    ):
        raise ValueError("independently verified true-physics foot entry features required")
    with np.load(npz, allow_pickle=False) as f:
        x = f["foot_relative"].reshape(-1, 12).astype(np.float32)
    if x.shape != (receipt["contact_conditions"], 12) or not np.isfinite(x).all():
        raise ValueError("finite complete 12D feature inputs required")
    y, groups = labels_and_groups(receipt["records"], len(x))
    contexts = sorted(int(c) for c in set(groups.tolist()))
    if len(contexts) < 2 or any(len(set(y[groups != c])) != 2 for c in contexts):
        raise ValueError("both contact classes in each train-only fold required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_hash=a.features_hash,
            audit_hash=a.audit_hash,
            source_head=a.expected_head,
            contexts=contexts,
            seeds=[17, 43, 71],
            threshold=0.5,
            optimizer_steps_per_fold=300,
            model="12-32-tanh-1 binary-logit",
            split="leave original passer context out",
            target="first receiver force includes non-foot category",
            retrospective_contact_condition=True,
            no_contact_training_samples=0,
            physical_control_authorized=False,
            hardware_authorized=False,
        ),
    )
    rows = []
    for context in contexts:
        train, test = groups != context, groups == context
        center = x[train].mean(axis=0)
        scale = np.maximum(x[train].std(axis=0), 0.05)
        z = np.clip((x - center) / scale, -8, 8).astype(np.float32)
        for seed in (17, 43, 71):
            torch.manual_seed(seed)
            model = torch.nn.Sequential(
                torch.nn.Linear(12, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)
            )
            optim = torch.optim.Adam(model.parameters(), lr=0.003)
            for _ in range(300):
                logits = model(torch.from_numpy(z[train])).squeeze(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.from_numpy(y[train].astype(np.float32))
                )
                optim.zero_grad()
                loss.backward()
                optim.step()
            with torch.no_grad():
                scores = torch.sigmoid(model(torch.from_numpy(z[test])).squeeze(1)).numpy()
            if not np.isfinite(scores).all() or not bool(torch.isfinite(loss)):
                raise ValueError("finite trained predictions required")
            cp = a.output / f"context-{context}-seed-{seed}.npz"
            np.savez_compressed(
                cp,
                center=center,
                scale=scale,
                weight1=model[0].weight.detach().numpy(),
                bias1=model[0].bias.detach().numpy(),
                weight2=model[2].weight.detach().numpy(),
                bias2=model[2].bias.detach().numpy(),
            )
            rows.append(
                dict(
                    context=context,
                    seed=seed,
                    training_samples=int(train.sum()),
                    checkpoint=cp.name,
                    checkpoint_hash=digest(cp),
                    heldout=[
                        dict(
                            index=int(i),
                            target_nonfoot=bool(y[i]),
                            score_nonfoot=float(score),
                            prediction_nonfoot=bool(score >= 0.5),
                        )
                        for i, score in zip(np.flatnonzero(test), scores, strict=True)
                    ],
                )
            )
    if (
        digest(fp) != a.features_hash
        or digest(ap) != a.audit_hash
        or digest(npz) != receipt["feature_hash"]
    ):
        raise ValueError("source changed during training")
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="RELAY_FOOT_CONTACT_EVENT_CLASSIFIER_NOT_CONTROL",
            source_head=a.expected_head,
            features_hash=a.features_hash,
            audit_hash=a.audit_hash,
            contexts=contexts,
            rows=rows,
            optimizer_updates=len(rows) * 300,
            threshold=0.5,
            retrospective_contact_condition=True,
            physical_control_authorized=False,
            independent_physics_evaluation=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
