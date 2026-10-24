"""Authenticated exploratory proposal data for frozen-actor critic studies."""

import json


def admit_feedback(bank_root, bank_hash, training_root, training_hash, pins):
    from scripts.rsi_native_context_recovery_common import digest

    reports = []
    for root, expected in ((bank_root, bank_hash), (training_root, training_hash)):
        path = root / "complete.json"
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("exact complete feedback/checkpoint lineage required")
        report = json.loads(path.read_text())
        for key, value in report["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting feedback ancestry")
            pins[key] = value
        pins[str(path)] = expected
        reports.append(report)
    bank, old = reports
    if (
        bank["status"] != "CORRELATED_NATIVE_EXPLORATION_FEEDBACK_BANK_REVIEWED_NOT_PROMOTED"
        or bank["stochastic_trials"] != 32
        or not bank["all_paired_replays_exact"]
        or old["actual_BC_actor_optimizer_updates"] != 390
        or old["SAC_update_index"] != 0
        or old["critic_optimizer_updates"] != 0
    ):
        raise ValueError("whole32 exploratory family and unchanged BC390 anchor required")
    from pathlib import Path

    def pin(root, entry):
        path = root / entry["path"]
        if (
            path.is_symlink()
            or path.parent.resolve() != root.resolve()
            or digest(path) != entry["hash"]
        ):
            raise ValueError("confined immutable training input required")
        pins[str(path)] = entry["hash"]
        return path

    checkpoint = pin(training_root, old["checkpoint"])
    policy = pin(training_root, old["policy"])
    files = []
    for row in bank["records"]:
        if row["scale"] == 0:
            continue  # identity replays are not new exploration examples
        root = Path(row["source"])
        ep = row["episode_artifact"]
        files.append(
            (
                row,
                pin(root, ep),
                pin(root, ep["guarded_proposal_trace"]),
                pin(bank_root, row["feedback_artifact"]),
            )
        )
    if len(files) != 32 or len({(row["seed"], row["index"]) for row, *_ in files}) != 32:
        raise ValueError("all32 independent seed/context pairs required, no success-only filtering")
    return old, checkpoint, policy, files


def critic_arrays(files):
    import numpy as np

    groups = {"train": [], "validation": []}
    for row, ep_path, trace_path, feedback_path in files:
        with (
            np.load(ep_path) as ep,
            np.load(trace_path) as trace,
            np.load(feedback_path) as feedback,
        ):
            n = row["native_steps"]
            obs = ep["observations"]
            valid = trace["actor_proposal_valid"]
            if (
                obs.shape != (n, 102)
                or valid.shape != (n,)
                or valid.dtype != np.bool_
                or not np.array_equal(valid, feedback["actor_proposal_valid"])
                or np.any(trace["actor_proposals"][~valid])
            ):
                raise ValueError("exact causal observation/proposal/missingness alignment required")
            indices = np.unique(np.append(np.arange(15, n, 10), n - 1))
            indices = indices[valid[indices]]
            if not len(indices):
                raise ValueError("empty actual proposal episode is not silently dropped")
            target = np.stack(
                [
                    feedback[name + "_returns"][indices]
                    for name in ("rewards", "fall_costs", "constraint_costs")
                ],
                axis=1,
            )
            values = (
                np.stack([obs[k - 15 : k + 1] for k in indices]),
                trace["actor_proposals"][indices].copy(),
                target.copy(),
            )
            split = "train" if row["index"] in (0, 3) else "validation"
            if row["split"] != split or not all(np.isfinite(value).all() for value in values):
                raise ValueError("fixed untouched validation split and finite tensors required")
            groups[split].append(values)
    return {
        split: tuple(np.concatenate([item[i] for item in episodes]) for i in range(3))
        for split, episodes in groups.items()
    }
