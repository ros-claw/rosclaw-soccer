"""Bind receiver training episodes to failure times in replayed native histories."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def failure_timing(failure: np.ndarray, controls: int, actual: dict) -> dict:
    """Read the first measured two-body failure; retain the full outcome."""
    if (
        type(controls) is not int
        or controls < 1
        or failure.dtype != np.bool_
        or failure.ndim != 2
        or failure.shape[1] != 2
        or len(failure) < controls
    ):
        raise ValueError("full aligned two-body failure history required")
    first = np.flatnonzero(failure.any(axis=1))
    failed = failure.any(axis=0)
    if (
        type(actual.get("body_failure")) is not bool
        or type(actual.get("A_failed")) is not bool
        or type(actual.get("B_failed")) is not bool
        or actual["body_failure"] != bool(first.size)
        or actual["A_failed"] != bool(failed[0])
        or actual["B_failed"] != bool(failed[1])
    ):
        raise ValueError("recorded and native physical failure outcomes differ")
    return dict(
        failure_control=int(first[0]) if first.size else None,
        failed_bodies=[name for name, value in zip(("A", "B"), failed, strict=True) if value],
        physical_controls=len(failure),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-hash", required=True)
    parser.add_argument(
        "--pair",
        action="append",
        nargs=4,
        required=True,
        metavar=("ROOT", "AUDIT", "SOURCE_HASH", "AUDIT_HASH"),
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    dataset_path = args.dataset / "complete.json"
    if (
        len(args.pair) != 3
        or dataset_path.is_symlink()
        or digest(dataset_path) != args.dataset_hash
        or args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
    ):
        raise ValueError("three pinned sources, dataset and unused external output required")
    data = json.loads(dataset_path.read_text())
    if (
        data.get("status") != "RECEIVER_ONPOLICY_PHYSICAL_EPISODES_NOT_TRAINED"
        or data.get("reward_profile") != "center-strict-full-history-v1"
        or data.get("episodes") != 72
        or len(data.get("records", [])) != 72
        or any(data.get(k) is not False for k in ("promotion_authorized", "hardware_authorized"))
    ):
        raise ValueError("complete unpromoted physical sampling dataset required")
    sources, pins = {}, [(dataset_path, args.dataset_hash)]
    for root_name, audit_name, source_hash, audit_hash in args.pair:
        root, audit_path = Path(root_name), Path(audit_name)
        source, _, observed_hash = report(
            root,
            audit_path,
            audit_hash,
            "NEURAL_RELAY_RECEIVER_CENTER_EXPLORATION32_PENDING_OWN_REPLAY",
        )
        declaration = root / "declaration.json"
        block = json.loads(declaration.read_text()).get("development_context_block")
        if (
            observed_hash != source_hash
            or declaration.is_symlink()
            or digest(declaration) != source["declaration_hash"]
            or type(block) is not int
            or block not in (0, 1, 2)
            or block in sources
        ):
            raise ValueError("distinct exact declared replayed blocks required")
        sources[block] = (root, source)
        pins.extend(
            (
                (root / "complete.json", source_hash),
                (audit_path, audit_hash),
                (declaration, source["declaration_hash"]),
            )
        )
    rows, seen = [], set()
    for episode in data["records"]:
        root, source = sources[episode["source_block"]]
        trial = episode["source_trial"]
        if type(trial) is not int or not 0 <= trial < len(source["records"]):
            raise ValueError("invalid source trial")
        row = source["records"][trial]
        course = row["course"]
        key = (episode["context"], episode["side"], episode["seed"])
        if key in seen or key != (
            course["context"],
            course["lateral_m"],
            course["arm"]["exploration_seed"],
        ):
            raise ValueError("unique matching physical course required")
        seen.add(key)
        physical = bind(root, row, "physics")
        if (
            physical is None
            or row["artifacts"]["physics"]["hash"] != episode["source_physics_hash"]
        ):
            raise ValueError("exact original native physics required")
        with np.load(physical, allow_pickle=False) as f:
            if f["body_failed"].shape != (row["controls"], 2):
                raise ValueError("full physics control history required")
            outcome = failure_timing(f["body_failed"], episode["controls"], episode["actual"])
        pins.append((physical, episode["source_physics_hash"]))
        rows.append(
            dict(
                context=key[0],
                side=key[1],
                seed=key[2],
                sampled_controls=episode["controls"],
                source_block=episode["source_block"],
                source_trial=trial,
                source_physics_hash=episode["source_physics_hash"],
                **outcome,
            )
        )
    if len(seen) != 72 or any(digest(path) != h for path, h in pins):
        raise ValueError("complete unique stable trajectories required")
    clean(args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        dict(
            status="RECEIVER_FAILURE_TIMELINE_REPLAY_BOUND_NOT_TRAINED",
            source_head=args.expected_head,
            dataset_hash=args.dataset_hash,
            source_pairs=[
                dict(block=b, source_hash=digest(root / "complete.json"))
                for b, (root, _) in sorted(sources.items())
            ],
            records=rows,
            failure_count=sum(r["failure_control"] is not None for r in rows),
            actor_training=False,
            independent_fresh=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
