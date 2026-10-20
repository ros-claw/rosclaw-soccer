"""Bounded automatic successor: closed learner -> whole40 physics -> raw audit.

Exactly one fixed candidate. Zero accepted updates skip unchanged physics.
Never retries a failed child or promotes a model. Goal's next Agent turn must
interpret the sealed outcome and choose a new hypothesis, not rerun this loop.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "core-source", "learner", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "expected-head",
        "expected-inputs-hash",
        "expected-learner-head",
        "learner-unit",
        "expected-anchor-hash",
    ):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_motion_portfolio_common import admit
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, bd, pins, _ = admit(a)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            learner_unit=a.learner_unit,
            expected_learner_head=a.expected_learner_head,
            expected_anchor_hash=a.expected_anchor_hash,
            maximal_new_native_steps=240000,
            maximum_candidates=1,
            automatic_retry_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    try:
        wait_for_terminal_collection(a.learner_unit, wait=True)
        d = json.loads((a.learner / "declaration.json").read_text())
        c = json.loads((a.learner / "complete.json").read_text())
        n = c["accepted_actor_optimizer_steps"]
        if (
            type(n) is not int
            or n not in range(5)
            or d["source_head"] != a.expected_learner_head
            or c["source_head"] != a.expected_learner_head
            or c["behavior_checkpoint_hash"] != a.expected_anchor_hash
            or c["protected_train_token_memory"] is not True
            or d["protected_deterministic_train_memory"] != [0, 10, 15, 16]
            or d["maximum_protected_token_changes"] != 0
            or c["checkpoint_hash"] != digest(a.learner / "candidate.pt")
            or c["status"]
            != ("GROUPED_ACTOR_UPDATED_REQUIRES_NATIVE_EXAM" if n else "GROUPED_ACTOR_UNCHANGED")
        ):
            raise ValueError("exact closed guarded learner required")
        accepted = [r for r in c["history"] if r["accepted"]]
        if len(accepted) != n or any(
            r["protected_token_changes"] != 0 or not -1e-6 <= r["mean_kl"] <= 0.005
            for r in accepted
        ):
            raise ValueError("accepted proposals violate fixed action memory gate")
        for k, h in d["input_file_hashes"].items():
            if k in pins and pins[k] != h:
                raise ValueError("conflicting learner commitment")
            pins[k] = h
        for name in ("declaration.json", "complete.json", "candidate.pt"):
            pins[str(a.learner / name)] = digest(a.learner / name)
        check_pins(pins)
        verdict, native = None, 0
        if n:
            import mujoco

            from rosclaw_soccer.rsi.grouped_actor_verdict import grouped_actor_verdict
            from scripts.rsi_audit_online_kinematic_motion import read_phase
            from scripts.rsi_online_kinematic_motion import run_shards

            exam = a.output / "exam"
            exam.mkdir()
            a.seed = 17
            write_once(
                exam / "declaration.json",
                dict(
                    source_head=a.expected_head,
                    candidate_hash=c["checkpoint_hash"],
                    contexts=list(range(40)),
                    planned_native_steps=240000,
                    original_far_or_private_fresh=False,
                    promotion_authorized=False,
                    hardware_authorized=False,
                ),
            )
            run_shards(
                a,
                exam,
                "evaluate",
                a.learner / "candidate.pt",
                c["checkpoint_hash"],
                list(range(40)),
                0,
            )
            world = mujoco.MjModel.from_binary_path(spec["world"])
            phases = read_phase(
                SimpleNamespace(seed=17, expected_pipeline_source=a.expected_head),
                bd,
                pins,
                world,
                exam,
                "evaluate",
                0,
                c["checkpoint_hash"],
            )
            verdict = grouped_actor_verdict(
                [r for _, phase in phases for r in phase["result"]["results"]]
            )
            native = 240000
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="GUARDED_ACTOR_ALL40_REVIEWED"
                if n
                else "GUARDED_ADAM_DIRECTION_INFEASIBLE_PIVOT_REQUIRED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                candidate_hash=c["checkpoint_hash"],
                actor_updates_observed=n,
                verdict=verdict,
                new_native_steps=native,
                optimizer_steps=0,
                learner_updates_not_recounted=True,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
                next_agent_action="INTERPRET_SEALED_RESULT_SELECT_NEW_HYPOTHESIS_NOT_WAIT_ON_TERMINATED_JOBS",
            ),
        )
        print("GUARDED_SUCCESSOR_CLOSED", n, verdict, flush=True)
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                automatic_retry_authorized=False,
                partial_physical_charge_requires_child_receipts=True,
                goal_complete=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
