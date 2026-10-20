"""Automatic closed four-group admission and critic-free advantage bank, no NN."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "core-source", "plan", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("expected-head", "expected-inputs-hash", "expected-plan-hash"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_grouped_motion_data import load_grouped_motion
    from scripts.rsi_motion_portfolio_common import admit
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, bd, pins, _ = admit(a)
    if digest(a.plan) != a.expected_plan_hash:
        raise ValueError("exact preregistered four-group collection plan required")
    plan = json.loads(a.plan.read_text())
    pins[str(a.plan)] = a.expected_plan_hash
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.rsi.grouped_advantage_bank.v1",
            source_head=a.expected_head,
            plan_hash=a.expected_plan_hash,
            expected_contexts=32,
            expected_groups=4,
            input_file_hashes=pins,
            expected_behavior_hash=plan["anchor_checkpoint_hash"],
            new_native_steps=0,
            optimizer_steps=0,
            neural_inference_calls=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    try:
        import mujoco
        import numpy as np

        from rosclaw_soccer.rsi.grouped_motion_returns import leave_one_out_returns

        world = mujoco.MjModel.from_binary_path(spec["world"])
        data = load_grouped_motion(plan, bd, world, pins)
        advantages, diagnostics = leave_one_out_returns(data["rewards"])
        artifact = a.output / "grouped-motion-bank.npz"
        np.savez_compressed(
            artifact,
            **{key: data[key] for key in ("features", "raw", "mean", "std", "log_prob", "rewards")},
            advantages=advantages,
            course_indices=np.asarray(data["course_indices"], dtype=np.int64),
        )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="FOUR_GROUP_CRITIC_FREE_MOTION_BANK_CLOSED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                artifact=artifact.name,
                artifact_hash=digest(artifact),
                behavior_checkpoint_hash=plan["anchor_checkpoint_hash"],
                diagnostics=diagnostics,
                training_contexts=32,
                validation_contexts=0,
                episode_executions=128,
                reviewed_native_steps=768000,
                new_native_steps=0,
                optimizer_steps=0,
                neural_inference_calls=0,
                raw_outcomes=data["outcomes"],
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        print("FOUR_GROUP_ADVANTAGE_BANK_CLOSED", diagnostics, flush=True)
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                new_native_steps=0,
                optimizer_steps=0,
                automatic_retry_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
