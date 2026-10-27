"""Train three offline team-reception actors on report-only paired SIM data.

The old source did not retain raw traces; new model training never promotes a
policy or certifies its behavior in the shared physical world.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.rsi.team_receiving_selector import TeamReceivingSelector
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean

DATA_HASH = "sha256:ece1a34f5228db6d39bdea74f197056bf78fd1382903eb5d16b5b51dc8cdbf33"
SEEDS = (17, 43, 71)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    report_path = args.dataset / "complete.json"
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path(__file__).resolve().parents[1])
        or report_path.is_symlink()
        or digest(report_path) != DATA_HASH
    ):
        raise ValueError("unused external output and exact exported training data required")
    report = json.loads(report_path.read_text())
    path = args.dataset / "dataset.npz"
    if (
        report["status"] != "TEAM_RECEIVING_PAIRED_DATASET_REPORT_ONLY_NOT_PROMOTED"
        or report["independent_replay_verified"] is not False
        or path.is_symlink()
        or digest(path) != report["dataset_hash"]
    ):
        raise ValueError("exact data and explicit evidence ceiling required")
    with np.load(path, allow_pickle=False) as data:
        x = data["features"].copy()
        actions = data["action_offsets"].copy()
        rewards = data["rewards"].copy()
        safe = data["safe"].copy()
        eligible = data["eligible"].copy()
    if (
        x.shape != (12, 10)
        or actions.shape != (16, 12)
        or rewards.shape != (12, 16)
        or safe.shape != (12, 16)
        or eligible.shape != (12,)
        or eligible.dtype != np.bool_
        or safe.dtype != np.bool_
        or not np.isfinite(x).all()
        or not np.isfinite(actions).all()
        or not np.isfinite(rewards).all()
        or int(eligible.sum()) != 11
        or not np.isin(rewards, [0, 1]).all()
    ):
        raise ValueError("finite original paired bank and eleven eligible states required")
    x = x[eligible]
    y = np.where(safe[eligible], rewards[eligible], -1.0).astype(np.float32)
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 0.01)
    z = ((x - center) / scale).astype(np.float32)
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            source_head=args.expected_head,
            dataset_hash=DATA_HASH,
            seed_values=SEEDS,
            architecture="10-32-tanh-16",
            critic_epochs=1200,
            actor_epochs=1200,
            critic_loss="full paired Q regression, unsafe -1",
            actor_objective="expected frozen Q plus parent retention penalty",
            parent_deviation_penalty=0.10,
            entropy_weight=0.01,
            scaling_floor=0.01,
            selection_outside_five_normalized_units="parent",
            offline=True,
            original_trajectory_only_hashes=True,
            eligible_training_states=11,
            development_retention_required=True,
            fresh_multibody_evaluation_required=True,
            evaluation_no_seed_selection=True,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    torch.set_num_threads(1)
    xt, yt = torch.from_numpy(z), torch.from_numpy(y)
    records = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        actor = torch.nn.Sequential(
            torch.nn.Linear(10, 32), torch.nn.Tanh(), torch.nn.Linear(32, 16)
        )
        critic = torch.nn.Sequential(
            torch.nn.Linear(10, 32), torch.nn.Tanh(), torch.nn.Linear(32, 16)
        )
        opt_c = torch.optim.Adam(critic.parameters(), lr=0.005)
        for _ in range(1200):
            loss_c = ((critic(xt) - yt) ** 2).mean()
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()
        with torch.no_grad():
            q = critic(xt).detach()
        opt_a = torch.optim.Adam(actor.parameters(), lr=0.005)
        for _ in range(1200):
            logp = torch.log_softmax(actor(xt), dim=1)
            probability = logp.exp()
            loss_a = (
                -(probability * q).sum(dim=1).mean()
                + 0.1 * (1 - probability[:, 0]).mean()
                + 0.01 * (probability * logp).sum(dim=1).mean()
            )
            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()
        value = dict(
            schema="soccer.team_receiving_offline_actor.v1",
            seed=seed,
            source_dataset_hash=DATA_HASH,
            center=center.tolist(),
            scale=scale.tolist(),
            weight1=actor[0].weight.detach().numpy().tolist(),
            bias1=actor[0].bias.detach().numpy().tolist(),
            weight2=actor[2].weight.detach().numpy().tolist(),
            bias2=actor[2].bias.detach().numpy().tolist(),
            critic_weight1=critic[0].weight.detach().numpy().tolist(),
            critic_bias1=critic[0].bias.detach().numpy().tolist(),
            critic_weight2=critic[2].weight.detach().numpy().tolist(),
            critic_bias2=critic[2].bias.detach().numpy().tolist(),
            offsets=actions.tolist(),
            promotion_authorized=False,
            hardware_authorized=False,
        )
        selector = TeamReceivingSelector.from_dict(value)
        selected = [selector.select(tuple(float(v) for v in row))[0] for row in x]
        model_path = args.output / f"seed-{seed}.json"
        write_once(model_path, value)
        records.append(
            dict(
                seed=seed,
                model_path=model_path.name,
                model_hash=digest(model_path),
                critic_mse=float(((critic(xt) - yt) ** 2).mean().detach()),
                selected_train_actions=selected,
                selected_train_successes=int(sum(y[i, a] == 1 for i, a in enumerate(selected))),
                parent_train_successes=int(np.count_nonzero(y[:, 0] == 1)),
            )
        )
    write_once(
        args.output / "complete.json",
        dict(
            status="TEAM_RECEIVING_OFFLINE_ACTORS_TRAINED_NOT_EVALUATED",
            source_head=args.expected_head,
            source_dataset_hash=DATA_HASH,
            eligible_training_states=11,
            action_count=16,
            records=records,
            training_evidence_report_only=True,
            independent_physics_evaluation=False,
            offline=True,
            online_rl=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
