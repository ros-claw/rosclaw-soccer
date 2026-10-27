"""Read back every receiver episode and independently reconstruct density/reward."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
from rosclaw_soccer.rsi.motor_sampling_law import history_motor_logp
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, measured, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("SOURCE", "AUDIT"))
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for name in ("review-hash", "dataset-hash", "producer-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    dp = a.dataset / "complete.json"
    if (
        len(a.pair) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.review.is_symlink()
        or dp.is_symlink()
        or digest(a.review) != a.review_hash
        or digest(dp) != a.dataset_hash
    ):
        raise ValueError("exact reviewed sampled episodes and unused output required")
    review, receipt = json.loads(a.review.read_text()), json.loads(dp.read_text())
    if (
        review.get("status") != "RELAY_RECEIVER_EXPLORATION_PAIRED_NOT_TRAINED"
        or review.get("own_physics_neural_and_density_audited") is not True
        or review.get("zero_parent_byte_identical") is not True
        or receipt.get("status") != "RECEIVER_ONPOLICY_PHYSICAL_EPISODES_NOT_TRAINED"
        or receipt.get("source_head") != a.producer_head
        or receipt.get("review_hash") != a.review_hash
        or receipt.get("conditions") != 24
        or receipt.get("episodes") != 72
        or any(receipt.get(k) != 0 for k in ("actor_updates", "critic_updates"))
        or any(
            receipt.get(k) is not False
            for k in ("online_rl", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("complete untrained own physical dataset required")
    pins = {r["block"]: r for r in review["source_pins"]}
    rows = {(r["context"], r["side"], r["seed"]): r for r in receipt["records"]}
    if len(pins) != 3 or len(rows) != 72:
        raise ValueError("all unique source blocks and episodes required")
    blocks, seen, frozen, controls = set(), set(), [], 0
    maximum_density_error = 0.0
    for root_name, audit_name in a.pair:
        root, audit = Path(root_name), Path(audit_name)
        dec = json.loads((root / "declaration.json").read_text())
        block = dec["development_context_block"]
        if block in blocks or block not in pins:
            raise ValueError("unique originally reviewed source block required")
        blocks.add(block)
        source, _, sh = report(
            root,
            audit,
            pins[block]["own_audit_hash"],
            "NEURAL_RELAY_RECEIVER_EXPLORATION32_PENDING_OWN_REPLAY",
            worlds=32,
        )
        if sh != pins[block]["source_hash"]:
            raise ValueError("exact reviewed source required")
        model, mh, config, ch = receipt["model_identity"]
        if (
            source["input_file_hashes"].get(model) != mh
            or source["input_file_hashes"].get(config) != ch
            or digest(Path(model)) != mh
        ):
            raise ValueError("exact original motor and codec required")
        params = VelocityPolicyParameters.load(Path(config), expected_hash=ch)
        scale = np.asarray(params.action_scale, np.float32)
        for trial in source["records"]:
            seed = trial["course"]["arm"]["exploration_seed"]
            if seed is None:
                continue
            key = (trial["course"]["context"], trial["course"]["lateral_m"], seed)
            if key in seen or key not in rows:
                raise ValueError("unique sampled condition and seed required")
            seen.add(key)
            row = rows[key]
            if (
                row["source_block"] != block
                or row["source_trial"] != trial["index"]
                or row["actual"] != measured(root, trial)
            ):
                raise ValueError("exact actual source and terminal outcomes required")
            velocity, physics = bind(root, trial, "receiver_velocity"), bind(root, trial, "physics")
            episode = a.dataset / row["path"]
            if (
                velocity is None
                or physics is None
                or digest(velocity) != row["source_velocity_hash"]
                or digest(physics) != row["source_physics_hash"]
                or episode.is_symlink()
                or episode.resolve().parent != a.dataset.resolve()
                or digest(episode) != row["hash"]
            ):
                raise ValueError("exact confined original and exported trajectories required")
            with (
                np.load(velocity, allow_pickle=False) as v,
                np.load(physics, allow_pickle=False) as raw,
                np.load(episode, allow_pickle=False) as e,
            ):
                n = len(v["control"])
                if n != row["controls"] or not np.array_equal(v["control"], np.arange(n)):
                    raise ValueError("exact contiguous own sampled controls required")
                expected = {
                    "inputs": np.concatenate((v["obs"], v["context14"]), axis=1),
                    "reference": v["reference_action"],
                    "old_mean": v["action"],
                    "sampled_action": v["SAM_executed_action"],
                    "old_logp": v["SAM_logp"],
                    "rng_before": v["SAM_rng_before"],
                    "rng_after": v["SAM_rng_after"],
                    "action_scale": scale,
                }
                if set(e.files) != set(expected) | {"rewards"} or any(
                    not np.array_equal(e[k], value) for k, value in expected.items()
                ):
                    raise ValueError(
                        "actual exported own observations, actions or probabilities differ"
                    )
                actual = row["actual"]
                score = (
                    2.0 * actual["return_0p1m_and_healthy"]
                    + actual["handoff"]
                    + 0.25 * actual["outgoing_foot_strike"]
                    - 2.0 * actual["body_failure"]
                )
                reward = np.zeros(n, np.float32)
                reward[-1] = score
                if row["terminal_reward"] != score or not np.array_equal(e["rewards"], reward):
                    raise ValueError("measured terminal outcome reward differs")
                if row["receiver_nonfoot_contact_during_sampling"] != bool(
                    (raw["ball_forces"][:n, :, 1, 2] > 1).any()
                ):
                    raise ValueError("measured receiver nonfoot contact label differs")
                density = (
                    history_motor_logp(
                        "box-truncated-ar1",
                        torch.from_numpy(e["old_mean"]),
                        torch.from_numpy(e["sampled_action"]),
                        torch.from_numpy(e["reference"]),
                        torch.from_numpy(scale),
                        torch.full((29,), math.log(0.02), dtype=torch.float32),
                        rho=0.9,
                    )
                    .detach()
                    .numpy()
                )
                error = float(np.max(np.abs(density - e["old_logp"])))
                maximum_density_error = max(maximum_density_error, error)
                if error > 1e-4:
                    raise ValueError("actual full-history probability differs")
                controls += n
        frozen.extend(((root / "complete.json", sh), (audit, pins[block]["own_audit_hash"])))
    if (
        blocks != set(pins)
        or seen != set(rows)
        or controls != receipt["sampled_controls"]
        or digest(a.review) != a.review_hash
        or digest(dp) != a.dataset_hash
        or any(digest(path) != h for path, h in frozen)
    ):
        raise ValueError("all unchanged reviewed actual sample histories required")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="RECEIVER_ONPOLICY_READBACK_AND_DENSITY_NOT_TRAINED",
            source_head=a.expected_head,
            dataset_hash=a.dataset_hash,
            review_hash=a.review_hash,
            conditions=24,
            episodes=72,
            sampled_controls=controls,
            max_density_error=maximum_density_error,
            exact_inputs_actions_rewards_and_source_replayed=True,
            independent_new_physics=False,
            actor_updates=0,
            critic_updates=0,
            optimizer_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
