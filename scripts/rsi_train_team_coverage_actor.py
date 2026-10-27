"""Fit conservative offline actors to audited paired SIM outcomes, not RL."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.rsi.team_coverage_actor import TeamCoverageActor
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--dataset-hash", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    source = args.dataset / "complete.json"
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or source.is_symlink()
        or digest(source) != args.dataset_hash
    ):
        raise ValueError("exact dataset and new external output required")
    receipt = json.loads(source.read_text())
    data_path = args.dataset / "dataset.npz"
    if (
        receipt["status"] != "TEAM_COVERAGE_PAIRED_DATASET_NOT_PROMOTED"
        or receipt["promotion_authorized"] is not False
        or data_path.is_symlink()
        or digest(data_path) != receipt["dataset_hash"]
    ):
        raise ValueError("unpromoted exact coverage dataset required")
    with np.load(data_path, allow_pickle=False) as data:
        x = data["features"].copy()
        success = data["controlled_reception"].copy()
        safe = data["source_safe_no_fault"].copy()
        active = data["activation_enabled_unprotected"].copy()
        parent = data["parent_safe_controlled"].copy()
        offsets = data["action_offsets"].copy()
    if (
        x.shape != (6, 10)
        or offsets.shape != (16, 12)
        or any(a.shape != (6, 16) or a.dtype != np.bool_ for a in (success, safe, active))
        or parent.shape != (6,)
        or parent.dtype != np.bool_
        or not np.isfinite(x).all()
        or not np.isfinite(offsets).all()
    ):
        raise ValueError("complete finite paired dataset required")
    allowed = success & safe & active
    norms = np.linalg.norm(offsets, axis=1)
    targets = []
    for i in range(6):
        options = np.flatnonzero(allowed[i])
        targets.append(
            16 if parent[i] or not len(options) else int(options[np.argmin(norms[options])])
        )
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 0.01)
    xt = torch.from_numpy(((x - center) / scale).astype(np.float32))
    yt = torch.tensor(targets, dtype=torch.long)
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            dataset_hash=args.dataset_hash,
            source_head=args.expected_head,
            seeds=[17, 43, 71],
            epochs=1200,
            actor="10-32-tanh-17",
            method="supervised outcome imitation; not RL",
            target="minimum norm verified clean success, otherwise exact unchanged parent",
            raw_support_radius=0.004,
            source_safety_not_independently_recomputed=True,
            hardware_authorized=False,
        ),
    )
    torch.set_num_threads(1)
    records = []
    for seed in (17, 43, 71):
        torch.manual_seed(seed)
        actor = torch.nn.Sequential(
            torch.nn.Linear(10, 32), torch.nn.Tanh(), torch.nn.Linear(32, 17)
        )
        opt = torch.optim.Adam(actor.parameters(), lr=0.005)
        for _ in range(1200):
            loss = torch.nn.functional.cross_entropy(actor(xt), yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        value = dict(
            schema="soccer.team_coverage_actor.v1",
            seed=seed,
            source_dataset_hash=args.dataset_hash,
            center=center.tolist(),
            scale=scale.tolist(),
            weight1=actor[0].weight.detach().numpy().tolist(),
            bias1=actor[0].bias.detach().numpy().tolist(),
            weight2=actor[2].weight.detach().numpy().tolist(),
            bias2=actor[2].bias.detach().numpy().tolist(),
            support=x.tolist(),
            allowed=allowed.tolist(),
            offsets=offsets.tolist(),
            radius=0.004,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        model = TeamCoverageActor.from_dict(value)
        selected = [model.select(tuple(row))[0] for row in x]
        path = args.output / f"seed-{seed}.json"
        write_once(path, value)
        records.append(
            dict(
                seed=seed,
                model_path=path.name,
                model_hash=digest(path),
                targets=targets,
                selected=selected,
                training_match=sum(a == b for a, b in zip(targets, selected, strict=True)),
            )
        )
    if digest(source) != args.dataset_hash or digest(data_path) != receipt["dataset_hash"]:
        raise ValueError("dataset changed during training")
    write_once(
        args.output / "complete.json",
        dict(
            status="TEAM_COVERAGE_SUPERVISED_ACTORS_NOT_EVALUATED",
            source_head=args.expected_head,
            dataset_hash=args.dataset_hash,
            records=records,
            offline=True,
            online_rl=False,
            physics_evaluated=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
