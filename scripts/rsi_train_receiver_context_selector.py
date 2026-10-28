"""Train a SIM-only receiver selector on paired development rollouts."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.receiver_contextual_selector import train_context_selector
from rosclaw_soccer.rsi.receiver_probe_trust_update import _proxy, development_trust_update
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "baseline",
        "positive",
        "negative",
        "positive-review",
        "negative-review",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "baseline-hash",
        "positive-hash",
        "negative-hash",
        "positive-review-hash",
        "negative-review-hash",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    paths = (
        (a.baseline, a.baseline_hash),
        (a.positive, a.positive_hash),
        (a.negative, a.negative_hash),
        (a.positive_review, a.positive_review_hash),
        (a.negative_review, a.negative_review_hash),
    )
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    for path, expected in paths:
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError("exact paired source receipts required")
    baseline, positive, negative, plus, minus = (json.loads(p.read_text()) for p, _ in paths)
    plus_vector = [0.0, 0.0, 0.0, 0.1, -0.05, 0.0] * 2
    if (
        baseline.get("status") != "RELAY_LEG_RESIDUAL_ZERO_OWN_REPLAY_NOT_FOOTBALL_SUCCESS"
        or baseline.get("probe_leg_rad") != [0.0] * 12
        or positive.get("probe_leg_rad") != plus_vector
        or negative.get("probe_leg_rad") != [-x for x in plus_vector]
        or plus.get("baseline_hash") != a.baseline_hash
        or minus.get("baseline_hash") != a.baseline_hash
        or plus.get("candidate_hash") != a.positive_hash
        or minus.get("candidate_hash") != a.negative_hash
        or any(
            r.get("status") != "MATCHED_RECEIVER_PHYSICAL_PROBE_NOT_CATCH_CERTIFICATE"
            for r in (plus, minus)
        )
        or any(
            r.get("status") != "RELAY_LEG_RESIDUAL_PHYSICAL_PROBE_NOT_FOOTBALL_SUCCESS"
            for r in (positive, negative)
        )
        or any(
            r.get(k) is not False
            for r in (baseline, positive, negative, plus, minus)
            for k in ("promotion_authorized", "hardware_authorized", "goal_complete")
        )
        or any(
            positive.get(k) != baseline.get(k) or negative.get(k) != baseline.get(k)
            for k in ("source_head", "source_hash", "audit_hash", "snapshot_hash", "frontier_hash")
        )
    ):
        raise ValueError("unpromoted matching zero and symmetric rollout evidence required")
    development_trust_update(plus, minus)  # Require paired equal exposure and healthy bodies.
    by_run = [{r["index"]: r for r in d["rows"]} for d in (baseline, positive, negative)]
    by_review = [{r["index"]: r for r in d["rows"]} for d in (plus, minus)]
    if any(len(d["rows"]) != 8 for d in (baseline, positive, negative)) or any(
        len(x) != 8 or set(x) != set(by_run[0]) for x in (*by_run, *by_review)
    ):
        raise ValueError("all eight exact paired development cases required")
    observations, labels, costs, indices = [], [], [], []
    for index in sorted(by_run[0]):
        own_observations, first_qpos, first_qvel = [], [], []
        for (receipt, _), run in zip(paths[:3], by_run, strict=True):
            record = run[index]
            artifact = receipt.parent / record["physical_trajectory"]
            if (
                artifact.is_symlink()
                or artifact.resolve().parent != receipt.parent.resolve()
                or digest(artifact) != record["physical_trajectory_hash"]
            ):
                raise ValueError("confined hash-bound actual rollout required")
            with np.load(artifact, allow_pickle=False) as archive:
                observation = archive["receiver_observation"]
                action = archive["executed_receiver_action"]
                if (
                    observation.shape != (36, 98)
                    or observation.dtype != np.float32
                    or action.shape != (36, 29)
                    or action.dtype != np.float32
                    or not np.isfinite(observation).all()
                    or not np.isfinite(action).all()
                    or not np.array_equal(observation[1:, 69:98], action[:-1])
                ):
                    raise ValueError("complete own actual observation and action history required")
                own_observations.append(observation[0].copy())
                first_qpos.append(archive["qpos"][0].copy())
                first_qvel.append(archive["qvel"][0].copy())
            if digest(artifact) != record["physical_trajectory_hash"]:
                raise ValueError("source artifact changed during read")
        if (
            any(not np.array_equal(x, own_observations[0]) for x in own_observations[1:])
            or any(not np.array_equal(x, first_qpos[0]) for x in first_qpos[1:])
            or any(not np.array_equal(x, first_qvel[0]) for x in first_qvel[1:])
        ):
            raise ValueError("exact same physical and observed initial condition required")
        x, y = by_review[0][index], by_review[1][index]
        scores = [_proxy(x["baseline"]), _proxy(x["candidate"]), _proxy(y["candidate"])]
        choice = int(np.argmin(scores))
        if scores[0] - scores[choice] < 0.01:
            choice = 0  # Abstain from marginal proxy changes.
        observations.append(own_observations[0])
        labels.append(choice)
        costs.append(scores)
        indices.append(index)
    X = np.stack(observations)
    y = np.asarray(labels, dtype=np.int64)
    weights, training = train_context_selector(X, y, seed=1234, steps=200)
    for path, expected in paths:
        if digest(path) != expected:
            raise ValueError("training inputs changed")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    weights_path, dataset_path = a.output / "weights.npz", a.output / "development-data.npz"
    np.savez_compressed(weights_path, **weights)
    np.savez_compressed(
        dataset_path,
        observations=X,
        labels=y,
        costs=np.asarray(costs),
        indices=np.asarray(indices, dtype=np.int64),
    )
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_CONTEXT_SELECTOR_TRAINED_DEV_ONLY_NOT_PHYSICALLY_VALIDATED",
            source_head=a.expected_head,
            inputs={str(p): h for p, h in paths},
            weights_hash=digest(weights_path),
            dataset_hash=digest(dataset_path),
            choice_counts=np.bincount(y, minlength=3).tolist(),
            label_method="posthoc fixed measured contact-speed-distance proxy with 0.01 abstention",
            inference_uses_future=False,
            optimizer=training,
            physical_policy_validated=False,
            independent_fresh=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
