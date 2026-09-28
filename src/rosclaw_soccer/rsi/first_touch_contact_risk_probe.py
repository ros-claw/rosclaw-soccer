"""Train-only, episode-grain G1 first-touch risk probe; never a motor policy.

The fixed frame-60 observation precedes every first contact in the committed
training domain. Hyperparameters are selected on whole earlier seeds only;
the final seed is opened once as a diagnostic holdout, never for promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

FRAME = 60
L2_GRID = (0.01, 0.1, 1.0, 10.0)


def _probability(z: np.ndarray) -> np.ndarray:
    result: np.ndarray = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    return result


def fit_logistic(
    x: np.ndarray, y: np.ndarray, l2: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic regularized Newton fit with train-only normalization."""
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or len(x) < 8
        or not np.isfinite(x).all()
        or not np.isin(y, (0, 1)).all()
        or len(set(y.tolist())) != 2
        or l2 not in L2_GRID
    ):
        raise ValueError("invalid independent binary contact-risk training batch")
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    design = np.concatenate((np.ones((len(x), 1)), (x - mean) / scale), axis=1)
    weight = np.zeros(design.shape[1])
    penalty = np.diag([1e-6, *([l2] * x.shape[1])])
    for _ in range(40):
        p = _probability(design @ weight)
        gradient = design.T @ (p - y) / len(x) + penalty @ weight
        curvature = p * (1 - p)
        hessian = design.T @ (curvature[:, None] * design) / len(x) + penalty
        step = np.linalg.solve(hessian, gradient)
        weight -= step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return weight, mean, scale


def predict(x: np.ndarray, fit: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    weight, mean, scale = fit
    if x.ndim != 2 or x.shape[1] != len(mean) or not np.isfinite(x).all():
        raise ValueError("invalid contact-risk inference features")
    design = np.concatenate((np.ones((len(x), 1)), (x - mean) / scale), axis=1)
    return _probability(design @ weight)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    q = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.mean(-y * np.log(q) - (1 - y) * np.log(1 - q)))


def _scores(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    decision = p >= 0.5
    positives = y == 1
    negatives = y == 0
    return {
        "correct": int(np.count_nonzero(decision == positives)),
        "total": len(y),
        "clean_recall": float(np.mean(decision[positives])) if positives.any() else 0.0,
        "nonfoot_recall": float(np.mean(~decision[negatives])) if negatives.any() else 0.0,
        "log_loss": _log_loss(y, p),
    }


def evaluate(dataset_dir: Path, *, holdout_seed: int) -> dict[str, Any]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    data_path = dataset_dir / "episodes.npz"
    if (
        manifest.get("schema") != "rsi_isaac_first_touch_proprio_dataset_v1"
        or manifest.get("data_hash") != hash_bytes(data_path.read_bytes())
        or manifest.get("manifest_hash")
        != hash_json({k: v for k, v in manifest.items() if k != "manifest_hash"})
        or manifest.get("policy_input_excludes_current_action") is not True
        or manifest.get("episode_count", 0) < 48
    ):
        raise ValueError("unauthenticated or insufficient proprioceptive dataset")
    with np.load(data_path, allow_pickle=False) as data:
        if set(data.files) != {
            "features",
            "action_target_rad",
            "precontact_mask",
            "contact_label",
            "first_contact_frame",
            "training_course_seed",
            "environment",
        }:
            raise ValueError("dataset array contract changed")
        features = data["features"]
        masks = data["precontact_mask"]
        label = data["contact_label"]
        seeds = data["training_course_seed"]
        first = data["first_contact_frame"]
        lanes = data["environment"]
        actions = data["action_target_rad"]
    if (
        features.shape != (manifest["episode_count"], 300, len(manifest["feature_names"]))
        or actions.shape != (len(features), 300, 6)
        or masks.shape != (len(features), 300)
        or label.shape != (len(features),)
        or seeds.shape != (len(features),)
        or first.shape != (len(features),)
        or lanes.shape != (len(features),)
        or not np.array_equal(
            masks,
            np.arange(300)[None, :] < np.where(first[:, None] >= 0, first[:, None], 300),
        )
        or not masks[:, FRAME].all()
        or not np.isfinite(features).all()
        or not np.isfinite(actions).all()
        or not np.isin(label, (-1, 0, 1)).all()
        or not np.array_equal(
            np.unique(seeds),
            np.asarray(sorted(row["training_course_seed"] for row in manifest["sources"])),
        )
        or any(
            sorted(lanes[seeds == seed].tolist()) != list(range(16)) for seed in np.unique(seeds)
        )
        or holdout_seed not in seeds
        or len(np.unique(seeds)) < 3
        or np.count_nonzero(seeds == holdout_seed) != 16
    ):
        raise ValueError("invalid seed-held-out contact-risk dataset")
    train_seeds = sorted(set(seeds.tolist()) - {holdout_seed})
    if max(train_seeds) >= holdout_seed or any(
        np.count_nonzero(seeds == seed) != 16 for seed in train_seeds
    ):
        raise ValueError("holdout must be a complete, later independent course seed")
    x = features[:, FRAME].astype(np.float64)
    y = (label == 1).astype(np.int64)
    train = seeds != holdout_seed
    test = ~train
    modes = {"ball_only": tuple(range(5)), "ball_plus_proprio": tuple(range(x.shape[1]))}
    results = {}
    for mode, columns in modes.items():
        scores = {}
        for l2 in L2_GRID:
            losses = []
            for seed in train_seeds:
                inner_train = train & (seeds != seed)
                inner_test = seeds == seed
                fit = fit_logistic(x[inner_train][:, columns], y[inner_train], l2)
                losses.append(_log_loss(y[inner_test], predict(x[inner_test][:, columns], fit)))
            scores[l2] = float(np.mean(losses))
        selected = min(L2_GRID, key=lambda value: (scores[value], value))
        fit = fit_logistic(x[train][:, columns], y[train], selected)
        results[mode] = {
            "chosen_l2": selected,
            "train_seed_cv_log_loss": scores[selected],
            "heldout": _scores(y[test], predict(x[test][:, columns], fit)),
        }
    rule_prediction = (x[test, 3] > 0).astype(np.float64)
    result: dict[str, Any] = {
        "schema": "rsi_isaac_first_touch_contact_risk_probe_v1",
        "activation_ceiling": "SIM_ONLY",
        "motor_policy": False,
        "promotion_authorized": False,
        "fresh_opened": False,
        "dataset_manifest_hash": manifest["manifest_hash"],
        "training_seeds": train_seeds,
        "holdout_seed": holdout_seed,
        "observation_frame": FRAME,
        "independent_train_episodes": int(np.count_nonzero(train)),
        "independent_holdout_episodes": int(np.count_nonzero(test)),
        "ball_direction_rule": _scores(y[test], rule_prediction),
        "models": results,
    }
    result["report_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--holdout-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.dataset_dir, holdout_seed=args.holdout_seed)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
