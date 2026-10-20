"""Declared successor: authenticate whole40 native grouped-actor examination."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "core-source", "exam", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("expected-head", "expected-inputs-hash", "expected-plan-hash", "learner-unit"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_motion_portfolio_common import admit
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, bd, pins, _ = admit(a)
    if digest(a.exam / "declaration.json") != a.expected_plan_hash:
        raise ValueError("sealed whole40 exam plan required")
    plan = json.loads((a.exam / "declaration.json").read_text())
    if (
        len(plan["jobs"]) != 4
        or plan["independent_contexts"] != 40
        or plan["planned_native_steps"] != 240000
    ):
        raise ValueError("complete four-shard declaration required")
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            plan_hash=a.expected_plan_hash,
            raw_primitive_review=True,
            new_native_steps=0,
            optimizer_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    try:
        wait_for_terminal_collection(a.learner_unit)
        learner = Path(plan["learner_root"])
        d = json.loads((learner / "declaration.json").read_text())
        c = json.loads((learner / "complete.json").read_text())
        if (
            c["source_head"] != plan["learner_source_head"]
            or d["source_head"] != c["source_head"]
            or c["status"] != "GROUPED_ACTOR_UPDATED_REQUIRES_NATIVE_EXAM"
            or not 1 <= c["accepted_actor_optimizer_steps"] <= 4
            or c["checkpoint_hash"] != plan["candidate_hash"]
            or digest(learner / "candidate.pt") != plan["candidate_hash"]
            or c["behavior_checkpoint_hash"] != plan["anchor_checkpoint_hash"]
        ):
            raise ValueError("closed accepted-proposal identity differs")
        for r in c["history"]:
            if r["accepted"] and (not 0 < r["alpha"] <= 1 or not -1e-6 <= r["mean_kl"] <= 0.005):
                raise ValueError("actor proposal exceeds preregistered trust region")
        for k, h in d["input_file_hashes"].items():
            if k in pins and pins[k] != h:
                raise ValueError("conflicting learner inputs")
            pins[k] = h
        for name in ("declaration.json", "complete.json", "candidate.pt"):
            pins[str(learner / name)] = digest(learner / name)
        pins[str(a.exam / "declaration.json")] = a.expected_plan_hash
        for part, job in enumerate(plan["jobs"]):
            if (
                job["indices"] != list(range(40))[part::4]
                or Path(job["output"]) != a.exam / f"evaluate{part}"
            ):
                raise ValueError("exact original course partition required")
            wait_for_terminal_collection(job["unit"] + ".service", wait=True)
        check_pins(pins)
        import mujoco

        from rosclaw_soccer.rsi.grouped_actor_verdict import grouped_actor_verdict
        from scripts.rsi_audit_online_kinematic_motion import read_phase

        world = mujoco.MjModel.from_binary_path(spec["world"])
        phases = read_phase(
            SimpleNamespace(seed=17, expected_pipeline_source=plan["worker_source_head"]),
            bd,
            pins,
            world,
            a.exam,
            "evaluate",
            0,
            plan["candidate_hash"],
        )
        rows = [r for _, part in phases for r in part["result"]["results"]]
        verdict = grouped_actor_verdict(rows)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="GROUPED_ACTOR_ALL40_PRIMITIVES_AUTHENTICATED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                candidate_hash=plan["candidate_hash"],
                verdict=verdict,
                reviewed_native_steps=240000,
                new_native_steps=0,
                optimizer_steps=0,
                independently_reintegrated_by_this_review=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        print("GROUPED_ACTOR_ALL40_REVIEWED", verdict, flush=True)
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
