"""Run frozen dev-case torque candidates for the full original H300 exam."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-hash", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--external-source", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    source_path = a.source / "complete.json"
    bank_path = a.bank / "complete.json"
    if (
        a.output.exists()
        or a.review.is_symlink()
        or source_path.is_symlink()
        or bank_path.is_symlink()
        or digest(a.review) != a.review_hash
    ):
        raise ValueError("unused H300 and bound independent review required")
    review = json.loads(a.review.read_text())
    source = json.loads(source_path.read_text())
    if (
        review["status"] != "TORQUE_RESPONSE_ALL15_NN_AND_MANUAL_NATIVE_EXACT"
        or review["complete_arms"] != 15
        or review["new_neural_calls"] != 2400
        or review["new_native_steps"] != 24000
        or review["source_hash"] != digest(source_path)
        or source["status"] != "CFTRACK_TORQUE_RESPONSE_DEVELOPMENT_RECORDED"
        or len(source["records"]) != 5
        or not source["parent_prefixes_exact"]
        or not source["cold_template_unchanged"]
    ):
        raise ValueError("all fifteen independently verified local trajectories required")
    pins = dict(review["input_file_hashes"])
    pins[str(a.review)] = a.review_hash
    if pins.get(str(source_path)) != digest(source_path) or pins.get(str(bank_path)) != digest(
        bank_path
    ):
        raise ValueError("unchanged source and independently verified parent bank required")
    bank = json.loads(bank_path.read_text())
    external = a.external_source.resolve()
    if str(external / "policy/omnicontact/OmniContact.py") not in pins:
        raise ValueError("bound original policy source required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(s) for s in pins if s.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique source-qualified canonical scene required")
    from rosclaw_soccer.rsi.cold_cftrack_trial import validate_cold_ball_xy

    cases = (
        ("nominal", (1.0, 0.0)),
        ("x-minus", (0.98, 0.0)),
        ("x-plus", (1.02, 0.0)),
        ("y-minus", (1.0, -0.02)),
        ("y-plus", (1.0, 0.02)),
    )
    if [(r["case"], tuple(r["ball_xy"])) for r in source["records"]] != list(cases):
        raise ValueError("exact consumed development cases and order required")
    for _, xy in cases:
        validate_cold_ball_xy(xy)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            cases=cases,
            horizon_controls=300,
            maximum_original_nn_calls=1500,
            maximum_native_steps=15000,
            source_candidate_plans_frozen_before_this_exam=True,
            original_nn_recomputed_from_own_physical_state=True,
            source_trajectories_not_used_for_control=True,
            existing_parent_bank_used_for_comparison_only=True,
            local_development_cases_not_fresh=True,
            privileged_plan_search_cost_is_not_hidden=True,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import numpy as np

    from rosclaw_soccer.rsi.cftrack_direct_approach import condition_direct_kick_approach
    from rosclaw_soccer.rsi.cftrack_torque_intent import CFtrackTorqueIntentPlan
    from rosclaw_soccer.rsi.cftrack_torque_trial import complete_torque_trial, run_cold_torque_trial
    from rosclaw_soccer.rsi.cold_cftrack_trial import cold_cftrack_trial
    from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch, outcome_contract
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
    from scripts.rsi_cftrack_native_context import build_cold_cftrack

    template, _, _ = build_cold_cftrack(scenes[0].read_text(), external, direct_approach=True)
    before = cftrack_state_hash(template.policy)
    records, raw_trials = [], []
    reserved = 0
    for case, (name, xy) in zip(source["records"], cases, strict=True):
        candidate = case["candidate"]
        if candidate is None or not case["planning_margin_accepted"]:
            raise ValueError(
                "predeclared full H300 requires each frozen locally successful candidate"
            )
        plan_data = dict(candidate["plan"])
        plan_data["fraction_effort_hg"] = tuple(plan_data["fraction_effort_hg"])
        plan = CFtrackTorqueIntentPlan(**plan_data)
        player = cold_cftrack_trial(template, ball_xy=xy)
        reference = condition_direct_kick_approach(player.policy)
        if plan.parent_policy_hash != cftrack_state_hash(player.policy):
            raise ValueError("frozen plan must bind the same independently initialized parent")
        reserved += 3000
        write_once(a.output / (name + "-before.json"), dict(plan=asdict(plan), reserved=reserved))
        raw, run = run_cold_torque_trial(player, plan, controls=300)
        path = a.output / (name + ".npz")
        np.savez_compressed(path, **raw)
        pins[str(path)] = digest(path)
        outcome = dict(event_status="INCOMPLETE_NATIVE_EPISODE", task_pass=False, safe=False)
        if complete_torque_trial(run, controls=300):
            force = np.zeros((300, 1, 6))
            force[:, 0, :3] = raw["ball_forces"][:, 0].reshape(300, 10, 3).max(axis=1)
            q = raw["qpos"].reshape(300, 10, 43)
            lo = player.layout
            outcome = label_first_touch(
                dict(
                    force_n=force,
                    pelvis_z_per_substep_m=q[:, :, lo.rq + 2][:, None],
                    ball_position_after_step_m=q[:, -1, lo.bq : lo.bq + 3][:, None],
                )
            ).to_dict()
        parent = next(r for r in bank["records"] if r["case"] == name and r["arm"] == "parent")
        row = dict(
            case=name,
            ball_xy=xy,
            plan=asdict(plan),
            reference_conditioning=reference,
            run=run,
            outcome=outcome,
            verified_parent_outcome=parent["outcome"],
            artifact=dict(path=path.name, hash=pins[str(path)]),
        )
        write_once(a.output / (name + ".json"), row)
        records.append(row)
        raw_trials.append(raw)
    prefix_checks = []
    for source_case, raw in zip(source["records"], raw_trials, strict=True):
        path = a.source / source_case["candidate"]["artifact"]["path"]
        if path.parent != a.source or path.is_symlink() or pins.get(str(path)) != digest(path):
            raise ValueError("bound earlier candidate required only for post-run prefix check")
        with np.load(path, allow_pickle=False) as old:
            prefix_checks.append(
                len(raw["joint_targets"]) >= 160
                and all(
                    np.array_equal(raw[key][: len(old[key])], old[key])
                    for key in (
                        "qpos",
                        "qvel",
                        "torque",
                        "parent_torque",
                        "ball_forces",
                        "joint_targets",
                        "nn_decoded_targets",
                        "neural_obs",
                        "neural_actions",
                    )
                )
            )
    unchanged = cftrack_state_hash(template.policy) == before and template.world.native_steps == 0
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_TORQUE_FROZEN_H300_DEVELOPMENT_RECORDED",
            source_head=a.expected_head,
            source_hash=digest(source_path),
            review_hash=a.review_hash,
            input_file_hashes=pins,
            outcome_contract_hash=outcome_contract()["contract_hash"],
            records=records,
            candidate_passes=sum(r["outcome"]["task_pass"] for r in records),
            verified_parent_passes=sum(r["verified_parent_outcome"]["task_pass"] for r in records),
            earlier_candidate_prefixes_exact=all(prefix_checks),
            cold_template_unchanged=unchanged,
            new_neural_calls=sum(r["run"]["neural_calls"] for r in records),
            new_native_steps=sum(r["run"]["native_steps"] for r in records),
            reserved_native_steps=reserved,
            neural_optimizer_updates=0,
            independent_h300_nn_readback_pending=True,
            independent_h300_native_replay_pending=True,
            fresh_not_evaluated=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
