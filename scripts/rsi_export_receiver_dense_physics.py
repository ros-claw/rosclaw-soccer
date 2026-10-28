"""Align audited sampled actions with native SIM feedback; never train/promote."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.receiving_progress_reward import receiving_progress_rewards
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("SOURCE", "AUDIT"))
    for name in ("dataset", "dataset-audit", "review", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "dataset-hash",
        "dataset-audit-hash",
        "review-hash",
        "producer-head",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    dataset_path = a.dataset / "complete.json"
    pins = (
        (dataset_path, a.dataset_hash),
        (a.dataset_audit, a.dataset_audit_hash),
        (a.review, a.review_hash),
    )
    if (
        len(a.pair) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() or digest(p) != h for p, h in pins)
    ):
        raise ValueError("exact audited data, review, and unused external output required")
    data, audit, review = [json.loads(p.read_text()) for p, _ in pins]
    if (
        data.get("status") != "RECEIVER_ONPOLICY_PHYSICAL_EPISODES_NOT_TRAINED"
        or data.get("reward_profile") != "center-strict-full-history-v1"
        or data.get("review_hash") != a.review_hash
        or audit.get("status") != "RECEIVER_ONPOLICY_READBACK_AND_DENSITY_NOT_TRAINED"
        or audit.get("dataset_hash") != a.dataset_hash
        or audit.get("exact_inputs_actions_rewards_and_source_replayed") is not True
        or review.get("own_physics_neural_and_density_audited") is not True
        or len(data.get("records", [])) != 72
    ):
        raise ValueError("complete density-audited on-policy history required")
    review_pins = {v["block"]: v for v in review["source_pins"]}
    blocks, source_info = set(), {}
    for source_name, audit_name in a.pair:
        root, own = Path(source_name), Path(audit_name)
        dec = json.loads((root / "declaration.json").read_text())
        block = dec["development_context_block"]
        if block in blocks or block not in review_pins:
            raise ValueError("distinct reviewed source blocks required")
        blocks.add(block)
        src, _, source_hash = report(
            root,
            own,
            review_pins[block]["own_audit_hash"],
            "NEURAL_RELAY_RECEIVER_CENTER_EXPLORATION32_PENDING_OWN_REPLAY",
            worlds=32,
        )
        if (
            source_hash != review_pins[block]["source_hash"]
            or src["source_head"] != a.producer_head
        ):
            raise ValueError("exact source and replay identity required")
        model = mujoco.MjModel.from_xml_path(str(root / "scene.xml"))
        layout = NativeVelocityLayout(
            model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
        )
        source_info[block] = (
            root,
            src,
            layout,
            own,
            source_hash,
            review_pins[block]["own_audit_hash"],
        )
    if blocks != set(review_pins):
        raise ValueError("all three reviewed blocks required")
    a.output.mkdir(parents=True)
    rows, total = [], 0
    for i, record in enumerate(data["records"]):
        root, source, layout, _, _, _ = source_info[record["source_block"]]
        trial = source["records"][record["source_trial"]]
        course = trial["course"]
        if (course["context"], course["lateral_m"], course["arm"]["exploration_seed"]) != (
            record["context"],
            record["side"],
            record["seed"],
        ):
            raise ValueError("sample and native source identity mismatch")
        ep_path = a.dataset / record["path"]
        velocity, physics = bind(root, trial, "receiver_velocity"), bind(root, trial, "physics")
        if (
            ep_path.is_symlink()
            or ep_path.resolve().parent != a.dataset.resolve()
            or digest(ep_path) != record["hash"]
            or digest(velocity) != record["source_velocity_hash"]
            or digest(physics) != record["source_physics_hash"]
        ):
            raise ValueError("exact original episode and physical artifacts required")
        with (
            np.load(ep_path, allow_pickle=False) as ep,
            np.load(velocity, allow_pickle=False) as v,
            np.load(physics, allow_pickle=False) as raw,
        ):
            n = record["controls"]
            if (
                not np.array_equal(v["control"], np.arange(n))
                or not np.array_equal(
                    ep["inputs"], np.concatenate((v["obs"], v["context14"]), axis=1)
                )
                or not np.array_equal(ep["sampled_action"], v["SAM_executed_action"])
                or raw["qpos"].shape[0] < n
                or len(ep["rewards"]) != n
                or ep["rewards"][-1] != record["terminal_reward"]
            ):
                raise ValueError("sampled action and post-control physics alignment required")
            q = raw["qpos"][:n, -1, :]
            receiver, ball = (
                q[:, layout.rq : layout.rq + 2].copy(),
                q[:, layout.bq : layout.bq + 2].copy(),
            )
            forces = raw["ball_forces"][:n, :, 1, :].copy()
            failed = raw["body_failed"][:n, 1].copy()
            dense = receiving_progress_rewards(receiver, ball, forces, failed)
            strict = ep["rewards"].copy()
        target = a.output / f"episode-{i:02d}.npz"
        np.savez_compressed(
            target,
            receiver_xy=receiver,
            ball_xy=ball,
            ball_forces=forces,
            body_failed=failed,
            dense_rewards=dense,
            strict_terminal_rewards=strict,
        )
        rows.append(
            dict(
                path=target.name,
                hash=digest(target),
                original_episode=record["path"],
                original_episode_hash=record["hash"],
                source_physics_hash=record["source_physics_hash"],
                context=record["context"],
                side=record["side"],
                seed=record["seed"],
                controls=n,
                dense_sum=float(dense.sum()),
                strict_terminal_reward=record["terminal_reward"],
            )
        )
        total += n
    if (
        total != data["sampled_controls"]
        or any(digest(p) != h for p, h in pins)
        or any(
            digest(root / "complete.json") != sh or digest(own) != ah
            for root, _, _, own, sh, ah in source_info.values()
        )
    ):
        raise ValueError("all source identities must remain unchanged")
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_DENSE_PHYSICAL_FEEDBACK_EXPORTED_NOT_TRAINED",
            source_head=a.expected_head,
            dataset_hash=a.dataset_hash,
            dataset_audit_hash=a.dataset_audit_hash,
            review_hash=a.review_hash,
            episodes=len(rows),
            controls=total,
            records=rows,
            post_control_alignment=True,
            dense_is_not_success=True,
            ball_motion_confound=True,
            strict_reward_unmodified=True,
            independently_recomputed=False,
            trained=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
