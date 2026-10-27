"""Train three offline contextual actor-critics on authenticated SIM branches.

Four training states cannot establish generalization. All actions are discrete,
all rewards come from recorded force-based H300 outcomes, and the learned actor
is never authorized to execute or promoted by this script.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.rsi.sim_contact_bandit import (
    ACTION_NAMES,
    contact_features,
    offline_contact_reward,
    select_contact_action,
)
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean

ROOT = Path(os.environ["ROSCLAW_SOCCER_EVIDENCE"]).resolve(strict=True)
AUDIT = ROOT / "rsi-approach-training-audit-20261027/complete.json"
AUDIT_HASH = "sha256:d58a37ad318d3edcc50710adb83ff1e88f19e0bf8720b853b06ee364fd78d50a"
CASES = ("yminus-vminus", "yminus-vplus", "yplus-vminus", "yplus-vplus")
SEEDS = (17, 43, 71)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    if args.output.exists() or AUDIT.is_symlink() or digest(AUDIT) != AUDIT_HASH:
        raise ValueError("unused output and exact independent training audit required")
    audit = json.loads(AUDIT.read_text())
    if audit["status"] != "STATIC_TRAINING_HASH_FORCE_GEOMETRY_READBACK_NOT_EXECUTION" or set(
        audit["records"]
    ) != set(CASES):
        raise ValueError("all four training cases required")
    contexts, rewards = [], []
    inputs = {}
    for case in CASES:
        group = audit["records"][case]
        if group["shared_prefix_exact"] is not True or set(group["arms"]) != set(ACTION_NAMES):
            raise ValueError("all paired actions and shared physical prefix required")
        root = ROOT / f"rsi-approach-train-{case}-20261027"
        scores = []
        for arm in ACTION_NAMES:
            path = root / f"{arm}.npz"
            expected = group["arms"][arm]["artifact_hash"]
            if path.is_symlink() or digest(path) != expected:
                raise ValueError("exact audited physical training artifact required")
            inputs[str(path)] = expected
            scores.append(offline_contact_reward(group["arms"][arm]["outcome"]))
        with np.load(root / "parent.npz", allow_pickle=False) as data:
            q, v = data["qpos"], data["qvel"]
            if (
                q.shape != (3000, 43)
                or v.shape != (3000, 41)
                or not np.isfinite(q).all()
                or not np.isfinite(v).all()
            ):
                raise ValueError("complete native recorded state required")
            contexts.append(
                contact_features(float(q[1069, 1]), float(q[1069, 8]), float(v[1069, 1]))
            )
        rewards.append(scores)
    x = np.stack(contexts)
    y = np.asarray(rewards, dtype=np.float32)
    if x.shape != (4, 2) or y.shape != (4, 9):
        raise ValueError("exact four by nine train cohort required")
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            source_head=args.expected_head,
            training_audit_hash=AUDIT_HASH,
            inputs=inputs,
            seeds=SEEDS,
            actor_type="linear_softmax",
            critic_type="linear_Q_all_actions",
            critic_epochs=1000,
            actor_epochs=1000,
            learning_rate=0.01,
            entropy_weight=0.01,
            fixed_reward="task_pass + 0.1*clipped_forward; no contact -0.2; unsafe -2",
            independent_eval=(
                "initial ball y +/-0.03m and vy +/-0.015m/s; "
                "four combinations; all three seeds, no selection"
            ),
            offline=True,
            online_updates=False,
            privileged_sim_ball_velocity=True,
            hardware_authorized=False,
            task_promoted=False,
        ),
    )
    torch.set_num_threads(1)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    records = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        actor, critic = torch.nn.Linear(2, 9), torch.nn.Linear(2, 9)
        opt_c = torch.optim.Adam(critic.parameters(), lr=0.01)
        for _ in range(1000):
            loss_c = torch.mean((critic(xt) - yt) ** 2)
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()
        with torch.no_grad():
            frozen_q = critic(xt).detach()
        opt_a = torch.optim.Adam(actor.parameters(), lr=0.01)
        for _ in range(1000):
            logp = torch.log_softmax(actor(xt), dim=1)
            probability = logp.exp()
            actor_loss = -torch.mean(torch.sum(probability * frozen_q, dim=1)) + 0.01 * torch.mean(
                torch.sum(probability * logp, dim=1)
            )
            opt_a.zero_grad()
            actor_loss.backward()
            opt_a.step()
        w, b = actor.weight.detach().numpy(), actor.bias.detach().numpy()
        selected = [select_contact_action(row, w, b) for row in x]
        artifact = dict(
            schema="soccer.sim.offline_contact_bandit.v1",
            seed=seed,
            source_audit_hash=AUDIT_HASH,
            actor_weight=w.tolist(),
            actor_bias=b.tolist(),
            critic_weight=critic.weight.detach().numpy().tolist(),
            critic_bias=critic.bias.detach().numpy().tolist(),
            action_names=ACTION_NAMES,
            task_promoted=False,
            hardware_authorized=False,
        )
        path = args.output / f"seed-{seed}.json"
        write_once(path, artifact)
        records.append(
            dict(
                seed=seed,
                model_hash=digest(path),
                model_path=path.name,
                critic_mse=float(torch.mean((critic(xt) - yt) ** 2).detach()),
                selected_train_actions=[ACTION_NAMES[i] for i in selected],
                selected_train_rewards=[float(y[i, j]) for i, j in enumerate(selected)],
                parent_train_rewards=y[:, 0].tolist(),
                oracle_train_rewards=y.max(axis=1).tolist(),
            )
        )
    write_once(
        args.output / "complete.json",
        dict(
            status="OFFLINE_CONTEXTUAL_ACTOR_CRITIC_TRAINED_NOT_EVALUATED",
            source_head=args.expected_head,
            training_audit_hash=AUDIT_HASH,
            cases=CASES,
            training_contexts=x.tolist(),
            training_rewards=y.tolist(),
            records=records,
            number_training_states=4,
            physical_training_branches=36,
            independent_test_performed=False,
            offline=True,
            online_updates=False,
            privileged_sim_ball_velocity=True,
            task_promoted=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
