"""Read back learned-centered pilot episodes against own physical histories."""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
from rosclaw_soccer.rsi.motor_sampling_law import history_motor_logp
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.passer_curriculum_outcome import strict_pass_receive
from rosclaw_soccer.rsi.receiver_onpolicy_episode import build_receiver_episode
from rosclaw_soccer.rsi.receiver_strict_terminal_reward import strict_terminal_reward
from rosclaw_soccer.rsi.receiving_progress_reward import receiver_motion_progress_rewards
from rosclaw_soccer.rsi.relay_event_reward import strict_relay_event_rewards
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "exploration", "review", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("dataset-hash", "review-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--event-rewards", action="store_true")
    p.add_argument("--precontact", action="store_true")
    a = p.parse_args()
    clean(a.expected_head)
    manifest = a.dataset / "complete.json"
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or manifest.is_symlink()
        or a.review.is_symlink()
        or digest(manifest) != a.dataset_hash
        or digest(a.review) != a.review_hash
    ):
        raise ValueError("exact dataset and review and unused external audit required")
    data, review = json.loads(manifest.read_text()), json.loads(a.review.read_text())
    if (
        data.get("status")
        != (
            "PRECONTACT_RECEIVER_PILOT_PHYSICAL_DATA_NOT_TRAINED"
            if a.precontact
            else "LEARNED_RECEIVER_PILOT_PHYSICAL_DATA_NOT_TRAINED"
        )
        or data.get("review_hash") != a.review_hash
        or data.get("episodes") != 24
        or data.get("event_reward_profile")
        != ("strict-event-redistribution-v1" if a.event_rewards else None)
        or data.get("precontact_reward_profile")
        != ("full-history-contact-timed-v1" if a.precontact else None)
        or len(data.get("records", [])) != 24
        or review.get("sampled_source_hash") != data.get("source_hash")
        or any(
            data.get(k) is not False
            for k in (
                "independent_export_audit",
                "training_performed",
                "online_rl",
                "fresh_holdout",
                "promotion_authorized",
                "hardware_authorized",
            )
        )
    ):
        raise ValueError("complete untrained development pilot required")
    root, audit = a.exploration / "source", a.exploration / "audit/complete.json"
    source, _, source_hash = report(
        root,
        audit,
        data["source_audit_hash"],
        "NEURAL_RELAY_RECEIVER_PRECONTACT_EXPLORATION32_PENDING_OWN_REPLAY"
        if a.precontact
        else "NEURAL_RELAY_RECEIVER_LEARNED_EXPLORATION32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    if source_hash != data["source_hash"]:
        raise ValueError("exact original physical source required")
    model, model_hash, config, config_hash = data["model_identity"]
    if (
        source["input_file_hashes"].get(model) != model_hash
        or source["input_file_hashes"].get(config) != config_hash
        or digest(Path(model)) != model_hash
    ):
        raise ValueError("pinned original policy and codec required")
    params = VelocityPolicyParameters.load(Path(config), expected_hash=config_hash)
    expected_scale = np.asarray(params.action_scale, dtype=np.float32)
    layout = NativeVelocityLayout(
        mujoco.MjModel.from_xml_path(str(root / "scene.xml")),
        robot_prefix="b/",
        ball_body_name="soccer_ball",
        player_count=2,
    )
    reviewed = {(r["context"], r["side"], r["seed"]): r for r in review["records"]}
    trials = {
        (
            r["course"]["context"],
            r["course"]["lateral_m"],
            r["course"]["arm"]["exploration_seed"],
        ): r
        for r in source["records"]
    }
    seen, total, max_error = set(), 0, 0.0
    for row in data["records"]:
        key = (row["context"], row["side"], row["seed"])
        if key in seen or key not in reviewed or key not in trials:
            raise ValueError("unique reviewed sampled episode required")
        seen.add(key)
        trial = trials[key]
        velocity, physics = bind(root, trial, "receiver_velocity"), bind(root, trial, "physics")
        path = a.dataset / row["path"]
        if (
            row["source_trial"] != trial["index"]
            or velocity is None
            or physics is None
            or path.is_symlink()
            or path.resolve().parent != a.dataset.resolve()
            or digest(path) != row["hash"]
            or digest(velocity) != row["source_velocity_hash"]
            or digest(physics) != row["source_physics_hash"]
        ):
            raise ValueError("exact confined native histories required")
        with (
            np.load(path, allow_pickle=False) as ep,
            np.load(velocity, allow_pickle=False) as v,
            np.load(physics, allow_pickle=False) as raw,
        ):
            outcome = strict_pass_receive(
                raw["ball_forces"], raw["body_failed"], handoff=trial["result"]["handoff_executed"]
            )
            if (
                outcome != row["physical_clean_outcome"]
                or outcome != reviewed[key]["physical_clean_outcome"]
            ):
                raise ValueError("full physical outcome must match both source and review")
            expected = build_receiver_episode(dict(v), reward=strict_terminal_reward(outcome))
            n = len(expected["inputs"])
            if n != row["controls"] or n != reviewed[key]["stochastic_controls"]:
                raise ValueError("exact contiguous executed sample count required")
            extra = (
                {"action_scale", "dense_rewards"}
                | ({"event_rewards"} if a.event_rewards else set())
                | ({"precontact_rewards"} if a.precontact else set())
            )
            if set(ep.files) != set(expected) | extra or any(
                not np.array_equal(ep[k], val) for k, val in expected.items()
            ):
                raise ValueError("exported actions, histories or terminal rewards changed")
            if a.event_rewards:
                events, evidence = strict_relay_event_rewards(
                    raw["ball_forces"],
                    raw["body_failed"],
                    handoff=trial["result"]["handoff_executed"],
                    controls=n,
                )
                if (
                    not np.array_equal(ep["event_rewards"], events)
                    or row["event_reward_evidence"] != evidence
                ):
                    raise ValueError("measured event timing and score redistribution differ")
            if a.precontact:
                from rosclaw_soccer.rsi.precontact_feedback import full_history_precontact_feedback

                contact_rewards, contact_evidence = full_history_precontact_feedback(
                    raw["ball_forces"][:, :, 1, :], raw["body_failed"], controls=n
                )
                if (
                    not np.array_equal(ep["precontact_rewards"], contact_rewards)
                    or row["precontact_reward_evidence"] != contact_evidence
                ):
                    raise ValueError("actual full-history contact or failure timing differs")
            if not np.array_equal(ep["action_scale"], expected_scale):
                raise ValueError("sampled motor scale must match pinned policy codec")
            q = raw["qpos"][:n, -1, :]
            dense = receiver_motion_progress_rewards(
                q[:, layout.rq : layout.rq + 2].copy(),
                q[:, layout.bq : layout.bq + 2].copy(),
                raw["ball_forces"][:n, :, 1, :].copy(),
                raw["body_failed"][:n, 1].copy(),
            )
            if not np.array_equal(ep["dense_rewards"], dense):
                raise ValueError("measured dense feedback differs")
            logp = history_motor_logp(
                "box-truncated-ar1",
                torch.from_numpy(ep["old_mean"]),
                torch.from_numpy(ep["sampled_action"]),
                torch.from_numpy(ep["reference"]),
                torch.from_numpy(ep["action_scale"]),
                torch.full((29,), math.log(0.02)),
                rho=0.9,
            )
            error = float(np.max(np.abs(logp.detach().numpy() - ep["old_logp"])))
            if not math.isfinite(error) or error > 1e-4:
                raise ValueError("sample probability differs from executed behavior history")
            max_error = max(max_error, error)
            total += n
    if (
        len(seen) != 24
        or total != data["sampled_controls"]
        or digest(manifest) != a.dataset_hash
        or digest(a.review) != a.review_hash
        or digest(root / "complete.json") != source_hash
        or digest(audit) != data["source_audit_hash"]
    ):
        raise ValueError("all unchanged physical episodes and source receipts required")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="PRECONTACT_RECEIVER_PILOT_DATA_READBACK_AUDITED"
            if a.precontact
            else "LEARNED_RECEIVER_PILOT_DATA_READBACK_AUDITED",
            source_head=a.expected_head,
            dataset_hash=a.dataset_hash,
            review_hash=a.review_hash,
            episodes=24,
            controls=total,
            maximum_density_error=max_error,
            strict_reward_and_physical_history_recomputed=True,
            dense_feedback_uses_shared_reward_function=True,
            event_reward_profile="strict-event-redistribution-v1" if a.event_rewards else None,
            precontact_reward_profile="full-history-contact-timed-v1" if a.precontact else None,
            precontact_feedback_uses_shared_reward_function=a.precontact,
            fresh_holdout=False,
            training_performed=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
