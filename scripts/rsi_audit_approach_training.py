"""Reconstruct new40 raw outcomes, credit, counters and safe checkpoint history.

Zero integration, inference or optimization. Recomputed GAE targets are NOT a
claim that exact optimizer minibatch/target arrays were persisted (they were not).
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    import torch

    from rosclaw_soccer.rsi.approach_option_credit import approach_head_penalty, approach_option_gae
    from rosclaw_soccer.rsi.contact_option_credit import recovery_terminal_reward
    from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
    from rosclaw_soccer.rsi.native_contact_window import native_contact_window
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--hash", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    receipt = a.root / "complete.json"
    if a.output.exists() or digest(receipt) != a.hash:
        raise ValueError("unused output and exact completed40 required")
    r = json.loads(receipt.read_text())
    if (
        r["status"] != "NATIVE_APPROACH40_REQUIRES_FULL_EXAM"
        or r["end_episode"] != 40
        or r["mode"] != "train40"
        or r["training_seed"] != 7608
        or [x["episode"] for x in r["results"]] != list(range(40))
    ):
        raise ValueError("actual completed fixed40 history required")
    pins = {**r["input_file_hashes"], str(receipt): a.hash}
    for f in (
        Path(__file__).resolve(),
        Path("src/rosclaw_soccer/rsi/approach_option_credit.py").resolve(),
        Path("src/rosclaw_soccer/rsi/contact_option_credit.py").resolve(),
        Path("src/rosclaw_soccer/rsi/native_contact_window.py").resolve(),
    ):
        pins[str(f)] = digest(f)

    def pinned(suffix):
        matches = [Path(f) for f in pins if f.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError("exact actual source required: " + suffix)
        return matches[0]

    def load(path):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    check_pins(pins)
    bank = load(pinned("/precontact-resets.npz"))
    lengths = [
        len(
            load(
                pinned(
                    "/rsi-train-starts-body-bound-index-check-20261021/"
                    + name.removesuffix(".npz")
                    + "-plan.npz"
                )
            )["joint"]
        )
        for name in TRAIN_MOTION_NAMES
    ]
    model = mujoco.MjModel.from_binary_path(
        str(pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb"))
    )
    layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
    qi = [*range(layout.rq, layout.rq + 7), *layout.qi, *range(layout.bq, layout.bq + 7)]
    vi = [*range(layout.rv, layout.rv + 6), *layout.vi, *range(layout.bv, layout.bv + 6)]
    totals = dict(
        native_steps=0,
        head_rows=0,
        critic_rows=0,
        tail_rows=0,
        optimizer_updates=0,
        critic_only_updates=0,
        PPO_updates=0,
        actor_updates=0,
        positive_reward=0,
        negative_reward=0,
        quality_reward=0,
        plastic_rows=0,
        far_rows=0,
    )
    for row in r["results"]:
        ep = row["episode"]
        parts = []
        for name in ("head", "tail"):
            path = a.root / f"episode{ep:02d}" / (name + ".npz")
            if digest(path) != row[name + "_hash"]:
                raise ValueError("raw learning trajectory changed")
            pins[str(path)] = row[name + "_hash"]
            parts.append(load(path))
        head, tail = parts
        ids = head["bank_index"]
        if ids.shape != (32,) or not np.array_equal(ids, tail["bank_index"]):
            raise ValueError("same actual32world identities required")
        if row["lead"] != (10, 20, 40, 80)[ep // 2 % 4] or not np.all(
            bank["lead_controls"][ids] == row["lead"]
        ):
            raise ValueError("declared earlier-approach curriculum changed")
        if not np.array_equal(bank["source_id"][ids], np.arange(32) % 8):
            raise ValueError("balanced TRAIN8 source roles required")
        rq = layout.rq
        hq = head["qpos"]
        bodybad = (
            (hq[:, :, :, rq + 2] < 0.65)
            | (1 - 2 * (hq[:, :, :, rq + 4] ** 2 + hq[:, :, :, rq + 5] ** 2) < 0.9)
        ).any(axis=2) & head["valid"]
        if not np.array_equal(bodybad, head["body_failed"]) or not np.array_equal(
            (head["guards"].any(axis=2) & head["valid"]) | bodybad, head["terminated"]
        ):
            raise ValueError("actual all-substep head body guard differs")
        reward = np.stack(
            [
                approach_head_penalty(
                    {
                        k: head[k][t]
                        for k in ("valid", "tracking", "action_rate", "limits", "terminated")
                    }
                )
                for t in range(len(head["valid"]))
            ]
        )
        if not np.array_equal(reward, head["reward"]):
            raise ValueError("new nonpositive head credit differs from raw measurements")
        for w, index in enumerate(ids):
            arrays = {k: [] for k in ("qpos", "qvel", "forces")}
            for part in (head, tail):
                mask = part["valid"][:, w]
                for key, width in (("qpos", 43), ("qvel", 41), ("forces", 3)):
                    if mask.any():
                        arrays[key].append(part[key][mask, w].reshape(-1, width))
            q, v, f = (np.concatenate(arrays[k]) for k in ("qpos", "qvel", "forces"))
            q, v = q[:, qi], v[:, vi]
            hc, tc = int(head["valid"][:, w].sum()), int(tail["valid"][:, w].sum())
            hits = np.flatnonzero((head["forces"][:hc, w, :, :2] > 10).any(axis=2).reshape(-1))
            first = int(hits[0] // 10) if len(hits) else None
            frame = int(bank["reference_frame"][index])
            horizon = lengths[int(bank["source_id"][index])] - frame
            end = min(horizon, first + 15) if first is not None else horizon
            if not np.array_equal(head["frame_before"][:hc, w], np.arange(frame, frame + hc)) or (
                not head["terminated"][:, w].any() and hc != end
            ):
                raise ValueError("head extended source frames or ignored measured contact clock")
            if tc and (
                tail["observation_before"][:tc, w, 98:].any()
                or tail["observation_before"][:tc, w, 6:9].any()
            ):
                raise ValueError("recovery used future reference or ball commands")
            actual = dict(
                world=w,
                bank_index=int(index),
                postcontact=native_contact_window(
                    force=f[:, :2],
                    canonical_qpos=q,
                    canonical_qvel=v,
                    goal=bank["goal_world"][index],
                    intended_foot=int(bank["expected_foot"][index]),
                ),
                contact_failed=bool(head["terminated"][:, w].any()),
                recovery_failed=bool(tail["terminated"][:, w].any()),
                reference_controls=hc,
                recovery_controls=tc,
                forbidden=bool((f[:, 2] > 1).any()),
                stable=bool(
                    q[:, 2].min() >= 0.65 and (1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2)).min() >= 0.9
                ),
            )
            expected = np.zeros(len(tail["valid"]), dtype=np.float32)
            if tc:
                expected[tc - 1] = recovery_terminal_reward(actual)
            if not np.array_equal(expected, tail["reward"][:, w]):
                raise ValueError("new actual terminal ball reward differs")
            actual["recovery_reward"] = float(expected.sum())
            if actual != row["outcomes"][w]:
                raise ValueError("actual full-trajectory outcome differs")
            totals["positive_reward"] += int(actual["recovery_reward"] > 0)
            totals["negative_reward"] += int(actual["recovery_reward"] < 0)
            totals["quality_reward"] += int(actual["recovery_reward"] >= 10)
        advantage, returns = approach_option_gae(
            reward,
            head["old_value"],
            head["next_value"],
            head["terminated"],
            head["truncated"],
            head["valid"],
            tail_reward=tail["reward"],
            tail_valid=tail["valid"],
            tail_terminal=tail["terminal"],
        )
        if not np.isfinite(advantage).all() or not np.isfinite(returns).all():
            raise ValueError("nonfinite reconstructed actual .995 targets")
        obs = head["observation_before"]
        distance = torch.from_numpy(obs[:, :, 168:170].copy()).norm(dim=2).numpy()
        plastic = (distance < 0.7) & (np.abs(obs[:, :, 168:177]).sum(axis=2) > 0) & head["valid"]
        if (
            not np.array_equal(plastic, head["plastic"])
            or head["old_logp"][head["valid"] & ~plastic].any()
        ):
            raise ValueError("recorded near/far actor likelihood domain differs")
        h, t = int(head["valid"].sum()), int(tail["valid"].sum())
        totals["head_rows"] += h
        totals["critic_rows"] += 2 * h - 32
        totals["tail_rows"] += t
        totals["native_steps"] += (h + t) * 10
        totals["plastic_rows"] += int(plastic.sum())
        totals["far_rows"] += int((head["valid"] & ~plastic).sum())
        updates = len(row["updates"])
        totals["optimizer_updates"] += updates
        totals["critic_only_updates" if ep < 8 else "PPO_updates"] += updates
        totals["actor_updates"] += sum(ep >= 8 and x["plastic_rows"] > 0 for x in row["updates"])
        c = row["counts"]
        if (
            row["head_rows"] != h
            or row["plastic_rows"] != int(plastic.sum())
            or row["native_steps"] != totals["native_steps"]
            or c["head_policy_rows"] != totals["head_rows"]
            or c["head_critic_rows"] != totals["critic_rows"]
            or c["recovery_foundation_rows"] != totals["tail_rows"]
            or c["optimizer_completed"] != totals["optimizer_updates"]
            or c["optimizer_attempted"] != totals["optimizer_updates"]
            or c["value_only_steps"] != totals["critic_only_updates"]
            or c["PPO_steps"] != totals["PPO_updates"]
            or c["actor_update_steps"] != totals["actor_updates"]
        ):
            raise ValueError("actual per-episode physical/neural/update accounting differs")
        if ep % 8 == 7:
            print(json.dumps(dict(episode=ep, **totals)), flush=True)
    snapshots = {}
    for cp in r["checkpoints"]:
        path = a.root / cp["file"]
        if digest(path) != cp["hash"]:
            raise ValueError("saved safe checkpoint changed")
        pins[str(path)] = cp["hash"]
        snapshots[cp["episode"]] = torch.load(path, map_location="cpu", weights_only=True)
    if set(snapshots) != {0, 8, 24, 40} or r["native_steps"] != totals["native_steps"]:
        raise ValueError("complete fixed checkpoint history and physical totals required")
    initial = snapshots[0]
    for ep, state in snapshots.items():
        if any(
            not torch.equal(v, state["actor_state_dict"][k])
            for k, v in initial["actor_state_dict"].items()
            if k.startswith("base.")
        ):
            raise ValueError("immutable neural foundation changed")
        if any(
            not torch.equal(v, state["policy_state_dict"][k])
            for k, v in initial["policy_state_dict"].items()
            if k.startswith("baseline.")
        ):
            raise ValueError("initial132 stability anchor changed")
        if ep == 8 and (
            any(
                not torch.equal(v, state["actor_state_dict"][k])
                for k, v in initial["actor_state_dict"].items()
            )
            or not torch.equal(
                initial["policy_state_dict"]["log_std"], state["policy_state_dict"]["log_std"]
            )
        ):
            raise ValueError("warmup modified actual actor or exploration")
        optimizer = state["optimizer_state_dict"]
        groups = optimizer["param_groups"]
        if [len(x["params"]) for x in groups] != [6, 6, 1]:
            raise ValueError("bounded six-adapter/six-critic/one-std optimizer required")
        expected_ids = (
            set()
            if ep == 0
            else set(groups[1]["params"])
            if ep == 8
            else {i for group in groups for i in group["params"]}
        )
        if set(optimizer["state"]) != expected_ids:
            raise ValueError("Adam warmup or actual learning slots differ")
        for group_id, group in enumerate(groups):
            for key in group["params"]:
                if key not in optimizer["state"]:
                    continue
                expected_steps = state["counts"][
                    "optimizer_completed" if group_id == 1 else "actor_update_steps"
                ]
                if float(optimizer["state"][key]["step"]) != expected_steps:
                    raise ValueError("actual per-parameter Adam history differs")
    if snapshots[40]["counts"] != r["counts"]:
        raise ValueError("final checkpoint and final receipt counters differ")
    check_pins(pins)
    write_once(
        a.output,
        dict(
            schema="soccer.rsi.approach40_training_raw_audit.v1",
            input_file_hashes=pins,
            totals=totals,
            raw_episodes=40,
            actual_outcomes_rewards_counters_exact=True,
            original_foundation_and_initial132_anchor_unchanged=True,
            critic_only_warmup_exact=True,
            safe_checkpoint_Adam_steps_exact=True,
            reconstructed_targets_finite=True,
            optimizer_target_arrays_not_persisted=True,
            minibatch_indices_not_persisted=True,
            new_physics_steps=0,
            new_neural_rows=0,
            optimizer_updates=0,
            goal_complete=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
