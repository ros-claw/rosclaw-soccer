"""Independently recompute dense receiver feedback from original native traces."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, action="append", required=True)
    for name in ("dense", "original", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("dense-hash", "original-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    dp, op = a.dense / "complete.json", a.original / "complete.json"
    if (
        len(a.source) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or digest(dp) != a.dense_hash
        or digest(op) != a.original_hash
    ):
        raise ValueError("three sources, pinned data, unused external output required")
    dense, original = json.loads(dp.read_text()), json.loads(op.read_text())
    if (
        dense.get("status") != "RECEIVER_DENSE_PHYSICAL_FEEDBACK_EXPORTED_NOT_TRAINED"
        or dense.get("dataset_hash") != a.original_hash
        or dense.get("episodes") != 72
        or original.get("episodes") != 72
    ):
        raise ValueError("complete original paired episodes required")
    sources = {}
    for root in a.source:
        dec = json.loads((root / "declaration.json").read_text())
        block = dec["development_context_block"]
        if block in sources:
            raise ValueError("distinct source blocks required")
        source = json.loads((root / "complete.json").read_text())
        model = mujoco.MjModel.from_xml_path(str(root / "scene.xml"))
        sources[block] = (
            root,
            source,
            NativeVelocityLayout(
                model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
            ),
        )
    max_error, total, seen = 0.0, 0, set()
    for row, old in zip(dense["records"], original["records"], strict=True):
        key = (row["context"], row["side"], row["seed"])
        if (
            key in seen
            or key != (old["context"], old["side"], old["seed"])
            or row["original_episode_hash"] != old["hash"]
        ):
            raise ValueError("unique original episode identities required")
        seen.add(key)
        root, source, layout = sources[old["source_block"]]
        trial = source["records"][old["source_trial"]]
        path = bind(root, trial, "physics")
        target, orig = a.dense / row["path"], a.original / old["path"]
        if (
            digest(path) != old["source_physics_hash"]
            or digest(target) != row["hash"]
            or digest(orig) != old["hash"]
        ):
            raise ValueError("exact original and exported data required")
        with (
            np.load(path, allow_pickle=False) as raw,
            np.load(target, allow_pickle=False) as got,
            np.load(orig, allow_pickle=False) as ep,
        ):
            n = old["controls"]
            q = raw["qpos"][:n, -1, :]
            receiver, ball = q[:, layout.rq : layout.rq + 2], q[:, layout.bq : layout.bq + 2]
            force, failed = raw["ball_forces"][:n, :, 1, :], raw["body_failed"][:n, 1]
            if not all(
                np.array_equal(got[k], v)
                for k, v in (
                    ("receiver_xy", receiver),
                    ("ball_xy", ball),
                    ("ball_forces", force),
                    ("body_failed", failed),
                    ("strict_terminal_rewards", ep["rewards"]),
                )
            ):
                raise ValueError("native post-control states or strict reward differ")
            distance = np.minimum(np.sqrt(((receiver - ball) ** 2).sum(axis=1)), 4.0)
            expected = np.zeros(n, dtype=np.float64)
            expected[1:] = distance[:-1] - distance[1:]
            foot = np.flatnonzero((force[:, :, :2] > 10).any(axis=(1, 2)))
            nonfoot = np.flatnonzero((force[:, :, 2] > 1).any(axis=1))
            falls = np.flatnonzero(failed)
            if len(foot):
                expected[foot[0]] += 1
            if len(nonfoot):
                expected[nonfoot[0]] -= 1
            if len(falls):
                expected[falls[0]] -= 5
            error = float(np.max(np.abs(got["dense_rewards"] - expected)))
            if error > 1e-6:
                raise ValueError("independently recomputed dense reward mismatch")
            max_error = max(max_error, error)
            total += n
    if (
        len(seen) != 72
        or total != original["sampled_controls"]
        or digest(dp) != a.dense_hash
        or digest(op) != a.original_hash
    ):
        raise ValueError("all unmodified source episodes required")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_DENSE_FEEDBACK_INDEPENDENTLY_RECOMPUTED_NOT_TRAINED",
            source_head=a.expected_head,
            dense_hash=a.dense_hash,
            original_hash=a.original_hash,
            episodes=len(seen),
            controls=total,
            maximum_reward_error=max_error,
            physical_sources_rechecked=True,
            fresh_physics=False,
            trained=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
