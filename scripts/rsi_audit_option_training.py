"""Read-only measured-outcome audit of the fixed132 continuation, no inference.

Reconstruct all112 new episodes from saved native trajectories. This does not
approve the learned policy, restore an optimizer, or execute any physics.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def reconstruct(head, tail, *, qi, vi, goals, feet, indices):
    from rosclaw_soccer.rsi.contact_option_credit import recovery_terminal_reward
    from rosclaw_soccer.rsi.native_contact_window import native_contact_window

    worlds = len(indices)
    for part, maximum in ((head, 24), (tail, 60)):
        valid = part["valid"]
        if valid.dtype != np.bool_ or valid.ndim != 2 or valid.shape[1] != worlds:
            raise ValueError("aligned actual boolean masks required")
        if not 1 <= len(valid) <= maximum:
            raise ValueError("bounded physical control interval required")
        for value in part.values():
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError("nonfinite raw training data")
        for w in range(worlds):
            if not np.array_equal(valid[:, w], np.arange(len(valid)) < valid[:, w].sum()):
                raise ValueError("physical execution must be a contiguous prefix")
    if not np.array_equal(tail["terminal"], tail["terminated"] | tail["completed"]):
        raise ValueError("terminal mask differs from actual finite boundaries")
    if (tail["reward"][~tail["valid"]] != 0).any():
        raise ValueError("reward after physical execution")
    outcomes = []
    for w, bank_id in enumerate(indices):
        chunks = {key: [] for key in ("qpos", "qvel", "forces")}
        for part in (head, tail):
            mask = part["valid"][:, w]
            for key, width in (("qpos", 43), ("qvel", 41), ("forces", 3)):
                if mask.any():
                    chunks[key].append(part[key][mask, w].reshape(-1, width))
        if not chunks["qpos"]:
            raise ValueError("no actual native trajectory")
        q = np.concatenate(chunks["qpos"])[:, qi]
        v = np.concatenate(chunks["qvel"])[:, vi]
        f = np.concatenate(chunks["forces"])
        failed = bool(head["terminated"][:, w].any())
        count = int(tail["valid"][:, w].sum())
        last = np.flatnonzero(tail["terminal"][:, w])
        if (count and not np.array_equal(last, [count - 1])) or (not count and len(last)):
            raise ValueError("one terminal event at the last actual tail sample required")
        if failed == bool(count):
            raise ValueError("tail must follow exactly a nonfailed contact head")
        post = native_contact_window(
            force=f[:, :2],
            canonical_qpos=q,
            canonical_qvel=v,
            goal=goals[bank_id],
            intended_foot=int(feet[bank_id]),
        )
        result = dict(
            world=w,
            bank_index=int(bank_id),
            postcontact=post,
            contact_failed=failed,
            recovery_failed=bool(tail["terminated"][:, w].any()),
            recovery_controls=count,
            forbidden=bool((f[:, 2] > 1).any()),
            stable=bool(
                q[:, 2].min() >= 0.65 and (1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2)).min() >= 0.9
            ),
        )
        reward = np.zeros(len(tail["valid"]), dtype=np.float32)
        if count:
            reward[count - 1] = recovery_terminal_reward(result)
        if not np.array_equal(reward, tail["reward"][:, w]):
            raise ValueError("terminal reward differs from the measured native outcome")
        result["recovery_reward"] = float(reward.sum())
        outcomes.append(result)
    return outcomes


def main():
    import mujoco

    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--hash", required=True)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    receipt = a.root / "complete.json"
    if a.output.exists() or digest(receipt) != a.hash:
        raise ValueError("unused output and exact final receipt required")
    r = json.loads(receipt.read_text())
    if (
        r["status"] != "NATIVE_OPTION_CONTINUATION132_REQUIRES_FULL_EXAM"
        or r["end_episode"] != 132
        or [x["episode"] for x in r["results"]] != list(range(20, 132))
    ):
        raise ValueError("fixed complete112 new episodes required")
    pins = {**r["input_file_hashes"], str(receipt): a.hash}
    for file in (
        Path(__file__).resolve(),
        Path("src/rosclaw_soccer/rsi/native_contact_window.py").resolve(),
        Path("src/rosclaw_soccer/rsi/contact_option_credit.py").resolve(),
    ):
        pins[str(file)] = digest(file)

    def pinned(suffix):
        matches = [Path(x) for x in pins if x.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError("ambiguous actual input: " + suffix)
        return matches[0]

    def load(path):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    check_pins(pins)
    bank = load(pinned("/precontact-resets.npz"))
    model = mujoco.MjModel.from_binary_path(
        str(pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb"))
    )
    layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
    qi = [*range(layout.rq, layout.rq + 7), *layout.qi, *range(layout.bq, layout.bq + 7)]
    vi = [*range(layout.rv, layout.rv + 6), *layout.vi, *range(layout.bv, layout.bv + 6)]
    totals = dict(
        episodes=0,
        worlds=0,
        positive_reward=0,
        negative_reward=0,
        quality_reward=0,
        head_rows=0,
        tail_rows=0,
        optimizer_updates=0,
    )
    blocks = []
    for row in r["results"]:
        parts = []
        for name, key in (("rollout", "raw_hash"), ("recovery", "recovery_raw_hash")):
            path = a.root / f"episode{row['episode']:02d}" / (name + ".npz")
            if digest(path) != row[key]:
                raise ValueError("raw trajectory hash changed")
            pins[str(path)] = row[key]
            parts.append(load(path))
        head, tail = parts
        ids = head["bank_index"]
        if (
            ids.shape != (32,)
            or ids.dtype.kind not in "iu"
            or (ids < 0).any()
            or (ids >= len(bank["goal_world"])).any()
            or not np.array_equal(ids, tail["bank_index"])
        ):
            raise ValueError("same32 actual training world identities required")
        outcomes = reconstruct(
            head,
            tail,
            qi=qi,
            vi=vi,
            goals=bank["goal_world"],
            feet=bank["expected_foot"],
            indices=ids,
        )
        if outcomes != row["recovery_outcomes"]:
            raise ValueError("raw outcome reconstruction differs")
        h, t = int(head["valid"].sum()), int(tail["valid"].sum())
        totals["episodes"] += 1
        totals["worlds"] += 32
        totals["head_rows"] += h
        totals["tail_rows"] += t
        totals["optimizer_updates"] += len(row["minibatches"])
        rewards = [x["recovery_reward"] for x in outcomes]
        for key, value in (
            ("positive_reward", sum(x > 0 for x in rewards)),
            ("negative_reward", sum(x < 0 for x in rewards)),
            ("quality_reward", sum(x >= 10 for x in rewards)),
        ):
            totals[key] += value
        if row["native_steps"] != 10 * (totals["head_rows"] + totals["tail_rows"]):
            raise ValueError("new physical work accounting differs")
        blocks.append(
            dict(
                episode=row["episode"],
                terminal_reward_sum=sum(rewards),
                positive=sum(x > 0 for x in rewards),
                negative=sum(x < 0 for x in rewards),
                quality=sum(x >= 10 for x in rewards),
            )
        )
        if totals["episodes"] % 16 == 0:
            print(json.dumps(totals), flush=True)
    before = r["counts_before_resume"]
    if (
        r["native_steps"] != 10 * (totals["head_rows"] + totals["tail_rows"])
        or r["new_optimizer_updates"] != totals["optimizer_updates"]
        or r["counts"]["rollout_policy_rows"] - before["rollout_policy_rows"] != totals["head_rows"]
        or r["counts"]["recovery_foundation_rows"] - before["recovery_foundation_rows"]
        != totals["tail_rows"]
    ):
        raise ValueError("final actual work accounting differs")
    check_pins(pins)
    write_once(
        a.output,
        dict(
            schema="soccer.rsi.native_option_training_outcome_audit.v1",
            input_file_hashes=pins,
            totals=totals,
            episodes=blocks,
            actual_tail_rewards_and_outcomes_exact=True,
            new_physics_steps=0,
            new_neural_rows=0,
            optimizer_updates=0,
            training_outcomes_not_independent_skill_exam=True,
            goal_complete=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
