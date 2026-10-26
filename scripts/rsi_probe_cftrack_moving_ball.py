"""Paired H300 ball-velocity probe using one frozen development torque plan."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

CASES = (
    ("forward", (0.1, 0.0, 0.0)),
    ("backward", (-0.1, 0.0, 0.0)),
    ("left", (0.0, 0.1, 0.0)),
    ("right", (0.0, -0.1, 0.0)),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-hash", required=True)
    parser.add_argument("--h300", type=Path, required=True)
    parser.add_argument("--external-source", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete = a.h300 / "complete.json"
    if (
        a.output.exists()
        or a.review.is_symlink()
        or complete.is_symlink()
        or digest(a.review) != a.review_hash
    ):
        raise ValueError("unused velocity study and independently verified H300 required")
    proof = json.loads(a.review.read_text())
    source = json.loads(complete.read_text())
    if (
        proof["status"] != "CFTRACK_TORQUE_H300_ALL5_NN_AND_MANUAL_NATIVE_EXACT"
        or proof["new_neural_calls"] != 1500
        or proof["new_native_steps"] != 15000
        or proof["source_hash"] != digest(complete)
        or len(source["records"]) != 5
        or source["records"][0]["case"] != "nominal"
        or not source["records"][0]["outcome"]["task_pass"]
    ):
        raise ValueError("all five independently reviewed H300 and nominal winner required")
    pins = dict(proof["input_file_hashes"])
    pins[str(a.review)] = a.review_hash
    if pins.get(str(complete)) != digest(complete):
        raise ValueError("exact source H300 commitment required")
    external = a.external_source.resolve()
    if str(external / "policy/omnicontact/OmniContact.py") not in pins:
        raise ValueError("qualified original NN source required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(s) for s in pins if s.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique canonical physical scene required")
    from rosclaw_soccer.rsi.cftrack_torque_intent import CFtrackTorqueIntentPlan
    from rosclaw_soccer.rsi.cold_cftrack_trial import (
        validate_cold_ball_angular_velocity,
        validate_cold_ball_velocity,
    )
    from rosclaw_soccer.sim.contracts import hash_json

    for _, velocity in CASES:
        validate_cold_ball_velocity(velocity)
        validate_cold_ball_angular_velocity((-velocity[1] / 0.11, velocity[0] / 0.11, 0.0))
    raw_plan = dict(source["records"][0]["plan"])
    raw_plan["fraction_effort_hg"] = tuple(raw_plan["fraction_effort_hg"])
    frozen = CFtrackTorqueIntentPlan(**raw_plan)
    frozen.offsets()
    immutable = asdict(frozen)
    immutable.pop("parent_policy_hash")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            cases=CASES,
            ball_initial_xy=[1.0, 0.0],
            arms=["parent", "frozen-nominal-torque"],
            frozen_nominal_plan_parameters=immutable,
            frozen_nominal_plan_parameters_hash=hash_json(immutable),
            per_case_policy_hash_rebinding_is_not_optimization=True,
            zero_per_case_parameter_search=True,
            horizon_controls=300,
            maximum_nn_calls=2400,
            maximum_native_steps=24000,
            ball_velocity_set_once_before_world_and_policy_entry=True,
            initial_spin_is_no_slip_rolling_for_radius_0_11_m=True,
            original_h300_exam_unchanged=True,
            exploratory_velocity_probe_not_full_fresh_exam=True,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import numpy as np

    from rosclaw_soccer.rsi.cftrack_direct_approach import condition_direct_kick_approach
    from rosclaw_soccer.rsi.cftrack_torque_trial import complete_torque_trial, run_cold_torque_trial
    from rosclaw_soccer.rsi.cold_cftrack_trial import cold_cftrack_trial
    from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch, outcome_contract
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
    from scripts.rsi_cftrack_native_context import build_cold_cftrack

    template, _, _ = build_cold_cftrack(scenes[0].read_text(), external, direct_approach=True)
    template_hash = cftrack_state_hash(template.policy)
    records = []
    reserved = 0
    for name, velocity in CASES:
        spin = (-velocity[1] / 0.11, velocity[0] / 0.11, 0.0)
        for arm in ("parent", "frozen-nominal-torque"):
            player = cold_cftrack_trial(
                template, ball_xy=(1.0, 0.0), ball_velocity=velocity, ball_angular_velocity=spin
            )
            reference = condition_direct_kick_approach(player.policy)
            observed_velocity = player.world.data.qvel[player.layout.bv : player.layout.bv + 3]
            if not np.array_equal(observed_velocity, velocity):
                raise ValueError("declared moving ball differs from actual cold velocity")
            observed_spin = player.world.data.qvel[player.layout.bv + 3 : player.layout.bv + 6]
            if not np.array_equal(observed_spin, spin):
                raise ValueError("declared rolling spin differs from actual cold angular velocity")
            plan = replace(frozen, parent_policy_hash=cftrack_state_hash(player.policy))
            if arm == "parent":
                plan = replace(plan, fraction_effort_hg=(0.0,) * 29)
            reserved += 3000
            write_once(
                a.output / (name + "-" + arm + "-before.json"),
                dict(
                    initial_ball_velocity_m_s=list(velocity), plan=asdict(plan), reserved=reserved
                ),
            )
            raw, run = run_cold_torque_trial(player, plan, controls=300)
            path = a.output / (name + "-" + arm + ".npz")
            np.savez_compressed(path, **raw)
            pins[str(path)] = digest(path)
            outcome = dict(event_status="INCOMPLETE_NATIVE_EPISODE", task_pass=False, safe=False)
            if complete_torque_trial(run, controls=300):
                forces = np.zeros((300, 1, 6))
                forces[:, 0, :3] = raw["ball_forces"][:, 0].reshape(300, 10, 3).max(axis=1)
                q = raw["qpos"].reshape(300, 10, 43)
                lo = player.layout
                outcome = label_first_touch(
                    dict(
                        force_n=forces,
                        pelvis_z_per_substep_m=q[:, :, lo.rq + 2][:, None],
                        ball_position_after_step_m=q[:, -1, lo.bq : lo.bq + 3][:, None],
                    )
                ).to_dict()
            row = dict(
                case=name,
                ball_xy=[1.0, 0.0],
                ball_velocity_m_s=velocity,
                ball_angular_velocity_rad_s=spin,
                arm=arm,
                plan=asdict(plan),
                reference_conditioning=reference,
                run=run,
                outcome=outcome,
                artifact=dict(path=path.name, hash=pins[str(path)]),
                initial_state_hash=hash_json(
                    dict(qpos=raw["initial_qpos"].tolist(), qvel=raw["initial_qvel"].tolist())
                ),
            )
            records.append(row)
            write_once(a.output / (name + "-" + arm + ".json"), row)
    unchanged = (
        cftrack_state_hash(template.policy) == template_hash and template.world.native_steps == 0
    )
    check_pins(pins)
    clean(a.expected_head)
    pairs = [records[i : i + 2] for i in range(0, len(records), 2)]
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_FROZEN_TORQUE_MOVING_BALL_EXPLORATORY_RECORDED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            review_hash=a.review_hash,
            outcome_contract_hash=outcome_contract()["contract_hash"],
            records=records,
            parent_passes=sum(p[0]["outcome"]["task_pass"] for p in pairs),
            candidate_passes=sum(p[1]["outcome"]["task_pass"] for p in pairs),
            rescue_cases=[
                p[0]["case"]
                for p in pairs
                if not p[0]["outcome"]["task_pass"] and p[1]["outcome"]["task_pass"]
            ],
            regression_cases=[
                p[0]["case"]
                for p in pairs
                if p[0]["outcome"]["task_pass"] and not p[1]["outcome"]["task_pass"]
            ],
            new_neural_calls=sum(r["run"]["neural_calls"] for r in records),
            new_native_steps=sum(r["run"]["native_steps"] for r in records),
            reserved_native_steps=reserved,
            cold_template_unchanged=unchanged,
            paired_initial_states_equal=all(
                pair[0]["initial_state_hash"] == pair[1]["initial_state_hash"] for pair in pairs
            ),
            neural_optimizer_updates=0,
            independent_nn_and_native_review_pending=True,
            fresh_exam_not_evaluated=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
