"""Export reviewed physical exploration to a small, development-only dataset."""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.passer_curriculum_outcome import strict_pass_receive
from rosclaw_soccer.rsi.receiver_onpolicy_episode import build_receiver_episode
from rosclaw_soccer.rsi.receiver_strict_terminal_reward import strict_terminal_reward
from rosclaw_soccer.rsi.receiving_progress_reward import receiver_motion_progress_rewards
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, measured, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("exploration", "review", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("review-hash", "exploration-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.review.is_symlink()
        or digest(a.review) != a.review_hash
    ):
        raise ValueError("exact reviewed pilot and unused external output required")
    review = json.loads(a.review.read_text())
    if (
        review.get("status") != "LEARNED_RECEIVER_EXPLORATION_PILOT_REVIEWED"
        or review.get("parent_trajectory_identical") is not True
        or not math.isfinite(review.get("maximum_density_error", float("nan")))
        or review.get("maximum_density_error", 1) > 1e-4
        or review.get("sampled_episodes") != 24
        or len(review.get("records", [])) != 24
        or any(
            review.get(k) is not False
            for k in ("training_performed", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("complete replayed development-only pilot review required")
    root, audit = a.exploration / "source", a.exploration / "audit/complete.json"
    source, _, source_hash = report(
        root,
        audit,
        review["sampled_audit_hash"],
        "NEURAL_RELAY_RECEIVER_LEARNED_EXPLORATION32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    dec = json.loads((root / "declaration.json").read_text())
    if (
        source_hash != review["sampled_source_hash"]
        or source["source_head"] != a.exploration_head
        or dec["receiver_checkpoint_hash"] != review["receiver_checkpoint_hash"]
    ):
        raise ValueError("exact physically replayed behavior candidate required")
    configs = [
        Path(k)
        for k in source["input_file_hashes"]
        if k.endswith("/velocity/v0/params/deploy.yaml")
    ]
    models = [
        Path(k)
        for k in source["input_file_hashes"]
        if k.endswith("/velocity/v0/exported/policy.onnx")
    ]
    if len(configs) != 1 or len(models) != 1:
        raise ValueError("unique pinned reference policy and codec required")
    model_identity = (
        str(models[0]),
        source["input_file_hashes"][str(models[0])],
        str(configs[0]),
        source["input_file_hashes"][str(configs[0])],
    )
    params = VelocityPolicyParameters.load(configs[0], expected_hash=model_identity[3])
    scale = np.asarray(params.action_scale, np.float32)
    if scale.shape != (29,) or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("finite positive pinned action scales required")
    layout = NativeVelocityLayout(
        mujoco.MjModel.from_xml_path(str(root / "scene.xml")),
        robot_prefix="b/",
        ball_body_name="soccer_ball",
        player_count=2,
    )
    indexed = {
        (
            r["course"]["context"],
            r["course"]["lateral_m"],
            r["course"]["arm"]["exploration_seed"],
        ): r
        for r in source["records"]
    }
    expected = {(r["context"], r["side"], r["seed"]): r for r in review["records"]}
    if len(indexed) != 32 or len(expected) != 24:
        raise ValueError("complete unique development seed factorial required")
    a.output.mkdir(parents=True)
    rows = []
    for key, reviewed in sorted(expected.items()):
        trial = indexed[key]
        if (
            trial["index"] != reviewed["source_trial"]
            or measured(root, trial) != reviewed["candidate"]
        ):
            raise ValueError("actual source trajectory differs from reviewed episode")
        velocity, physics = bind(root, trial, "receiver_velocity"), bind(root, trial, "physics")
        if velocity is None or physics is None:
            raise ValueError("actual sampled neural and native physics histories required")
        with (
            np.load(velocity, allow_pickle=False) as v,
            np.load(physics, allow_pickle=False) as raw,
        ):
            outcome = strict_pass_receive(
                raw["ball_forces"], raw["body_failed"], handoff=trial["result"]["handoff_executed"]
            )
            if outcome != reviewed["physical_clean_outcome"]:
                raise ValueError("physical contact outcome differs from reviewed episode")
            episode = build_receiver_episode(dict(v), reward=strict_terminal_reward(outcome))
            n = len(episode["inputs"])
            if n != reviewed["stochastic_controls"] or raw["qpos"].shape[0] < n:
                raise ValueError("sampled controls must align with actual post-control physics")
            q = raw["qpos"][:n, -1, :]
            dense = receiver_motion_progress_rewards(
                q[:, layout.rq : layout.rq + 2].copy(),
                q[:, layout.bq : layout.bq + 2].copy(),
                raw["ball_forces"][:n, :, 1, :].copy(),
                raw["body_failed"][:n, 1].copy(),
            )
        target = a.output / f"episode-{len(rows):02d}.npz"
        np.savez_compressed(target, **episode, action_scale=scale, dense_rewards=dense)
        rows.append(
            dict(
                context=key[0],
                side=key[1],
                seed=key[2],
                source_trial=trial["index"],
                path=target.name,
                hash=digest(target),
                controls=n,
                source_velocity_hash=digest(velocity),
                source_physics_hash=digest(physics),
                physical_clean_outcome=outcome,
            )
        )
    if (
        len(rows) != 24
        or digest(a.review) != a.review_hash
        or digest(root / "complete.json") != source_hash
        or digest(audit) != review["sampled_audit_hash"]
    ):
        raise ValueError("complete unchanged source, review and replay required")
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="LEARNED_RECEIVER_PILOT_PHYSICAL_DATA_NOT_TRAINED",
            source_head=a.expected_head,
            review_hash=a.review_hash,
            source_hash=source_hash,
            source_audit_hash=review["sampled_audit_hash"],
            receiver_checkpoint_hash=dec["receiver_checkpoint_hash"],
            model_identity=model_identity,
            conditions=8,
            episodes=24,
            sampled_controls=sum(r["controls"] for r in rows),
            records=rows,
            dense_reward_profile="receiver-motion-kinematic-v2",
            independent_export_audit=False,
            research_training_sufficiency_proven=False,
            training_performed=False,
            online_rl=False,
            fresh_holdout=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
