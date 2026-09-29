"""SIM_ONLY online actor-critic for bilateral R1 receiving in a physical proxy.

The 500 Hz MuJoCo body and continuous contact teacher are recomputed on policy.
The higher-level teammate/teacher admission schedule is still frozen from two
consumed eight-player episodes. This is not a qualified match policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from numpy.typing import NDArray
from rsi_mjx_contact_replay_smoke import _contact_masks
from rsi_r1_dynamic_teacher_proxy_v141 import teacher_torque
from rsi_r1_pd_proxy_search_v140 import load_tape

from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.training.receiving_classroom import r1_contact_tap_receiving_configuration
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

SCHEMA = "rosclaw_soccer.rsi.r1_online_contact_ppo_v145.result.v1"
HORIZON = 60
ACTION_SIZE = 12
OBS_SIZE = 45


class ReceivingProxy:
    def __init__(self, model: mujoco.MjModel, tape: dict[str, NDArray[np.float64]]) -> None:
        self.model = model
        self.tape = tape
        self.data = mujoco.MjData(model)
        self.ball_geom, self.robot_mask, self.foot_mask = _contact_masks(model)
        self.shin_geoms = (model.geom("left_shin").id, model.geom("right_shin").id)
        self.foot_geom_ids = tuple(int(g) for g in np.flatnonzero(self.foot_mask))
        self.guard = 0.85 * np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
        _, self.teacher_base = r1_contact_tap_receiving_configuration()
        self.wrench = np.zeros(6, dtype=np.float64)
        self.closest = np.zeros(6, dtype=np.float64)
        self.frame = 0
        self.filtered = np.zeros(29, dtype=np.float64)
        self.first_foot: int | None = None
        self.nonfoot_frames: set[int] = set()
        self.minimum_height = float("inf")
        self.minimum_shin_gap = float("inf")
        self.reset()

    def reset(self) -> NDArray[np.float32]:
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self.tape["initial_local_qpos"]
        self.data.qvel[:] = self.tape["initial_local_qvel"]
        mujoco.mj_forward(self.model, self.data)
        self.frame = 0
        self.filtered[:] = 0
        self.first_foot = None
        self.nonfoot_frames.clear()
        self.minimum_height = float(self.data.qpos[2])
        self.minimum_shin_gap = float("inf")
        return self.observe()

    def observe(self) -> NDArray[np.float32]:
        q = self.data.qpos
        v = self.data.qvel
        ball = q[36:39]
        left_foot = self.data.xpos[self.model.body("left_ankle_roll_link").id]
        right_foot = self.data.xpos[self.model.body("right_ankle_roll_link").id]
        values = np.r_[
            (ball - q[:3]) / 2.0,
            v[35:38] / 3.0,
            v[:3] / 3.0,
            q[3:7],
            q[7:19] / 2.0,
            v[6:18] / 8.0,
            (ball - left_foot) / 1.0,
            (ball - right_foot) / 1.0,
            self.frame / HORIZON,
            float(self.first_foot is not None),
        ].astype(np.float32)
        if values.shape != (OBS_SIZE,) or not np.isfinite(values).all():
            raise ValueError("finite bilateral proprioceptive receiving observation required")
        return np.asarray(values, dtype=np.float32)

    def step(
        self, action: NDArray[np.float64]
    ) -> tuple[NDArray[np.float32], float, bool, dict[str, Any]]:
        if (
            action.shape != (ACTION_SIZE,)
            or not np.isfinite(action).all()
            or np.any(np.abs(action) > 1.000001)
            or self.frame >= HORIZON
        ):
            raise ValueError("bounded consecutive bilateral leg action required")
        frame = self.frame
        desired = np.zeros(29, dtype=np.float64)
        desired[:12] = 0.08 * action
        self.filtered += np.clip(0.25 * (desired - self.filtered), -0.02, 0.02)
        new_foot = False
        new_nonfoot = False
        for substep in range(10):
            index = frame * 10 + substep
            kp = self.tape["motor_kp"][index]
            kd = self.tape["motor_kd"][index]
            target = self.tape["motor_pd_target_rad"][index] + self.filtered
            position_torque = kp * (target - self.data.qpos[7:36])
            pd = position_torque - kd * self.data.qvel[6:35]
            teacher_row = self.tape["motor_teacher_inputs"][index]
            extra = teacher_torque(
                self.model, self.data, teacher_row, self.teacher_base, position_torque
            )
            self.data.ctrl[:] = np.clip(pd + extra, -self.guard, self.guard)
            mujoco.mj_step(self.model, self.data)
            self.minimum_height = min(self.minimum_height, float(self.data.qpos[2]))
            if 20 <= frame <= 38:
                gap = min(
                    float(
                        mujoco.mj_geomDistance(
                            self.model, self.data, geom, self.ball_geom, 10.0, self.closest
                        )
                    )
                    for geom in self.shin_geoms
                )
                self.minimum_shin_gap = min(self.minimum_shin_gap, gap)
            for contact_id in range(self.data.ncon):
                contact = self.data.contact[contact_id]
                a, b = int(contact.geom1), int(contact.geom2)
                if a != self.ball_geom and b != self.ball_geom:
                    continue
                other = b if a == self.ball_geom else a
                if not self.robot_mask[other]:
                    continue
                mujoco.mj_contactForce(self.model, self.data, contact_id, self.wrench)
                if self.wrench[0] <= 0:
                    continue
                if self.foot_mask[other]:
                    if self.first_foot is None:
                        self.first_foot = frame
                        new_foot = True
                elif frame not in self.nonfoot_frames:
                    self.nonfoot_frames.add(frame)
                    new_nonfoot = True
        self.frame += 1
        ball = self.data.qpos[36:39]
        foot_distance = min(
            float(np.linalg.norm(self.data.geom_xpos[g] - ball)) for g in self.foot_geom_ids
        )
        ball_speed = float(np.linalg.norm(self.data.qvel[35:38]))
        stable = (
            self.minimum_height >= 0.65
            and np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
        )
        reward = -0.001 * float(np.square(action).sum())
        if 15 <= frame <= 35:
            reward += 0.02 * max(-0.1, min(0.1, self.minimum_shin_gap))
        if new_foot:
            reward += 12.0
        if new_nonfoot:
            reward -= 18.0
        if not stable:
            reward -= 50.0
        done = self.frame >= HORIZON or not stable
        if done:
            reward += (
                24.0
                if self.first_foot is not None and not self.nonfoot_frames
                else -24.0
                if self.first_foot is None
                else -8.0
            )
            if self.first_foot is not None:
                reward -= 3.0 * min(foot_distance, 2.0) + 2.0 * min(ball_speed, 3.0)
        return (
            self.observe() if not done else np.zeros(OBS_SIZE, dtype=np.float32),
            reward,
            done,
            {
                "first_foot_frame": self.first_foot,
                "nonfoot_frames": sorted(self.nonfoot_frames),
                "safe": bool(stable),
                "minimum_shin_gap_m": self.minimum_shin_gap,
                "terminal_foot_distance_m": foot_distance,
                "terminal_ball_speed_mps": ball_speed,
            },
        )


class ActorCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(OBS_SIZE, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, ACTION_SIZE),
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(OBS_SIZE, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1),
        )
        self.log_std = torch.nn.Parameter(torch.full((ACTION_SIZE,), -0.9))
        output = self.actor[-1]
        assert isinstance(output, torch.nn.Linear)
        torch.nn.init.zeros_(output.weight)
        torch.nn.init.zeros_(output.bias)

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(
            self.actor(observation), self.log_std.exp().expand_as(self.actor(observation))
        )


def collect(
    model: ActorCritic, environments: tuple[ReceivingProxy, ReceivingProxy], episodes: int
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    memory: dict[str, list[Any]] = {
        name: [] for name in ("observation", "action", "logp", "value", "reward", "done")
    }
    summaries = []
    for episode in range(episodes):
        env = environments[episode % 2]
        observation = env.reset()
        episode_reward = 0.0
        for _ in range(HORIZON):
            with torch.no_grad():
                state = torch.from_numpy(observation)
                distribution = model.distribution(state)
                raw_action = distribution.sample()  # type: ignore[no-untyped-call]
                logp = distribution.log_prob(raw_action).sum()  # type: ignore[no-untyped-call]
                value = model.critic(state).squeeze()
            bounded = raw_action.clamp(-1, 1).numpy().astype(np.float64)
            next_observation, reward, done, info = env.step(bounded)
            for name, item in (
                ("observation", state.numpy()),
                ("action", raw_action.numpy()),
                ("logp", float(logp)),
                ("value", float(value)),
                ("reward", reward),
                ("done", done),
            ):
                memory[name].append(item)
            episode_reward += reward
            observation = next_observation
            if done:
                summaries.append({"side": episode % 2, "reward": episode_reward, **info})
                break
    batch = {
        name: torch.as_tensor(np.asarray(values), dtype=torch.float32)
        for name, values in memory.items()
    }
    return batch, summaries


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    epochs: int,
) -> dict[str, float]:
    rewards = batch["reward"]
    dones = batch["done"]
    values = batch["value"]
    advantages = torch.zeros_like(rewards)
    gae = torch.tensor(0.0)
    for index in range(len(rewards) - 1, -1, -1):
        next_value = (
            values[index + 1]
            if index + 1 < len(rewards) and not bool(dones[index])
            else torch.tensor(0.0)
        )
        delta = rewards[index] + 0.99 * next_value * (1 - dones[index]) - values[index]
        gae = delta + 0.99 * 0.95 * (1 - dones[index]) * gae
        advantages[index] = gae
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    losses = []
    for _ in range(epochs):
        for indices in torch.randperm(len(rewards)).split(256):  # type: ignore[no-untyped-call]
            observation = batch["observation"][indices]
            action = batch["action"][indices]
            distribution = model.distribution(observation)
            new_logp = distribution.log_prob(action).sum(-1)  # type: ignore[no-untyped-call]
            ratio = (new_logp - batch["logp"][indices]).exp()
            surrogate = torch.minimum(
                ratio * advantages[indices], ratio.clamp(0.8, 1.2) * advantages[indices]
            )
            value = model.critic(observation).squeeze(-1)
            loss = (
                -surrogate.mean()
                + 0.5 * torch.square(value - returns[indices]).mean()
                - 0.005 * distribution.entropy().sum(-1).mean()  # type: ignore[no-untyped-call]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    return {
        "loss_mean": float(np.mean(losses)),
        "log_std_mean": float(model.log_std.mean().detach()),
    }


def evaluate(
    model: ActorCritic, environments: tuple[ReceivingProxy, ReceivingProxy]
) -> list[dict[str, Any]]:
    results = []
    for side, env in enumerate(environments):
        observation = env.reset()
        for _ in range(HORIZON):
            with torch.no_grad():
                action = (
                    model.actor(torch.from_numpy(observation)).tanh().numpy().astype(np.float64)
                )
            observation, _, done, info = env.step(action)
            if done:
                results.append({"side": side, **info})
                break
    return results


def train(
    asset_root: Path,
    capture_dir: Path,
    output: Path,
    *,
    updates: int,
    episodes_per_update: int,
    seed: int,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if (
        output.exists()
        or output.resolve().is_relative_to(root)
        or not 1 <= updates <= 100
        or not 4 <= episodes_per_update <= 128
        or episodes_per_update % 2
        or not 0 <= seed < 2**31
    ):
        raise ValueError("new bounded external SIM_ONLY bilateral PPO evidence required")
    capture = json.loads((capture_dir / "report.json").read_text())
    if capture["status"] != "CAPTURE_QUALIFIED" or capture["report_hash"] != hash_json(
        {k: v for k, v in capture.items() if k != "report_hash"}
    ):
        raise ValueError("qualified bilateral frame-zero physical proxy required")
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    model.opt.timestep = 0.002
    tapes = tuple(
        load_tape(capture_dir / f"course-{row['course']['seed']}.npz", row["trace_hash"])
        for row in capture["rows"]
    )
    environments = tuple(ReceivingProxy(model, tape) for tape in tapes)
    if len(environments) != 2:
        raise ValueError("two qualified opposite-side receiving courses required")
    torch.manual_seed(seed)
    np.random.seed(seed)
    actor = ActorCritic()
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_online_contact_ppo_v145.py",
            "scripts/rsi_r1_dynamic_teacher_proxy_v141.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    baseline = evaluate(actor, environments)
    history = []
    for update in range(updates):
        batch, summaries = collect(actor, environments, episodes_per_update)
        optimization = ppo_update(actor, optimizer, batch, epochs=4)
        deterministic = evaluate(actor, environments)
        row = {
            "update": update + 1,
            "rollout_episodes": len(summaries),
            "mean_reward": float(np.mean([item["reward"] for item in summaries])),
            "sampled_clean_count": sum(
                item["first_foot_frame"] is not None and not item["nonfoot_frames"] and item["safe"]
                for item in summaries
            ),
            "deterministic": deterministic,
            **optimization,
        }
        history.append(row)
        (output / "progress.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "update": row["update"],
                    "mean_reward": row["mean_reward"],
                    "sampled_clean_count": row["sampled_clean_count"],
                    "deterministic": deterministic,
                }
            ),
            flush=True,
        )
    policy_path = output / "actor-state.pt"
    torch.save(
        {key: value.detach().cpu() for key, value in actor.state_dict().items()}, policy_path
    )
    final = evaluate(actor, environments)
    report = {
        "schema": SCHEMA,
        "capture_report_hash": capture["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_BILATERAL_FIXED_HIGH_LEVEL_PROXY_ONLINE_RL",
        "updates": updates,
        "episodes_per_update": episodes_per_update,
        "physical_episode_count": updates * episodes_per_update + 2 + 2 * updates + 2,
        "seed": seed,
        "baseline": baseline,
        "final": final,
        "history": history,
        "policy_hash": hash_bytes(policy_path.read_bytes()),
        "status": "DEVELOPMENT_BILATERAL_CLEAN_GAIN_UNVALIDATED"
        if all(
            item["first_foot_frame"] is not None and not item["nonfoot_frames"] and item["safe"]
            for item in final
        )
        else "REJECTED_BILATERAL_CLEAN_GATE",
        "fixed_high_level_teacher_schedule": True,
        "full_world_audition_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during online physical PPO")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.capture_dir,
        args.output,
        updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        seed=args.seed,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
