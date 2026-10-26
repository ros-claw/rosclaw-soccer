"""Bounded privileged initial-intercept reference probe on consumed moving cases."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--review-hash", required=True)
    p.add_argument("--moving", type=Path, required=True)
    p.add_argument("--external-source", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_probe_cftrack_moving_ball import CASES

    clean(a.expected_head)
    moving_path = a.moving / "complete.json"
    if (
        a.output.exists()
        or a.review.is_symlink()
        or moving_path.is_symlink()
        or digest(a.review) != a.review_hash
    ):
        raise ValueError("unused intervention and independently reviewed moving evidence required")
    proof = json.loads(a.review.read_text())
    prior = json.loads(moving_path.read_text())
    if (
        proof["status"] != "CFTRACK_MOVING_BALL_ALL8_NN_AND_MANUAL_NATIVE_EXACT"
        or proof["new_neural_calls"] != 2400
        or proof["new_native_steps"] != 24000
        or proof["source_hash"] != digest(moving_path)
        or not prior["paired_initial_states_equal"]
        or not prior["cold_template_unchanged"]
        or prior["records"][1]["case"] != "forward"
        or prior["records"][1]["arm"] != "frozen-nominal-torque"
    ):
        raise ValueError("complete independently reviewed paired motion baseline required")
    pins = dict(proof["input_file_hashes"])
    pins[str(a.review)] = a.review_hash
    if pins.get(str(moving_path)) != digest(moving_path):
        raise ValueError("source-bound prior moving trials required")
    external = a.external_source.resolve()
    if str(external / "policy/omnicontact/OmniContact.py") not in pins:
        raise ValueError("source-bound author CFgen and NN required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(s) for s in pins if s.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique canonical scene required")
    from rosclaw_soccer.rsi.cftrack_torque_intent import CFtrackTorqueIntentPlan
    from rosclaw_soccer.rsi.cold_cftrack_trial import (
        validate_cold_ball_angular_velocity,
        validate_cold_ball_velocity,
    )

    for _, velocity in CASES:
        validate_cold_ball_velocity(velocity)
        validate_cold_ball_angular_velocity((-velocity[1] / 0.11, velocity[0] / 0.11, 0.0))
    plan_data = dict(prior["records"][1]["plan"])
    plan_data["fraction_effort_hg"] = tuple(plan_data["fraction_effort_hg"])
    frozen = CFtrackTorqueIntentPlan(**plan_data)
    frozen.offsets()
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            cases=CASES,
            arms=["intercept-reference-only", "intercept-plus-fixed-nominal-torque"],
            maximum_nn_calls=2400,
            maximum_native_steps=24000,
            horizon_controls=300,
            nominal_reference_first_contact_time_s=1.44,
            assumed_approach_speed_m_s=0.8,
            constant_velocity_model_is_hypothesis_not_future_observation=True,
            privileged_sim_velocity_used_for_planning=True,
            real_observations_restored_before_nn=True,
            fixed_torque_parameters=asdict(frozen),
            no_per_case_torque_fitting_or_amplitude_grid=True,
            source_cases_are_consumed_development_not_fresh=True,
            unchanged_h300_and_torque_caps=True,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import numpy as np

    from rosclaw_soccer.rsi.cftrack_torque_trial import complete_torque_trial, run_cold_torque_trial
    from rosclaw_soccer.rsi.cftrack_velocity_intercept import condition_velocity_intercept
    from rosclaw_soccer.rsi.cold_cftrack_trial import cold_cftrack_trial
    from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch, outcome_contract
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_cftrack_native_context import build_cold_cftrack

    template, _, _ = build_cold_cftrack(scenes[0].read_text(), external, direct_approach=True)
    template_hash = cftrack_state_hash(template.policy)
    records = []
    reserved = 0
    for name, velocity in CASES:
        spin = (-velocity[1] / 0.11, velocity[0] / 0.11, 0.0)
        for arm in ("intercept-reference-only", "intercept-plus-fixed-nominal-torque"):
            player = cold_cftrack_trial(
                template, ball_xy=(1.0, 0.0), ball_velocity=velocity, ball_angular_velocity=spin
            )
            receipt = condition_velocity_intercept(player)
            plan = replace(frozen, parent_policy_hash=cftrack_state_hash(player.policy))
            if arm == "intercept-reference-only":
                plan = replace(plan, fraction_effort_hg=(0.0,) * 29)
            reserved += 3000
            write_once(
                a.output / (name + "-" + arm + "-before.json"),
                dict(reference=receipt, plan=asdict(plan), reserved_native_steps=reserved),
            )
            raw, run = run_cold_torque_trial(player, plan, controls=300)
            path = a.output / (name + "-" + arm + ".npz")
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
            row = dict(
                case=name,
                arm=arm,
                ball_velocity_m_s=velocity,
                ball_angular_velocity_rad_s=spin,
                plan=asdict(plan),
                reference=receipt,
                run=run,
                outcome=outcome,
                initial_state_hash=hash_json(
                    dict(qpos=raw["initial_qpos"].tolist(), qvel=raw["initial_qvel"].tolist())
                ),
                artifact=dict(path=path.name, hash=pins[str(path)]),
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
            status="CFTRACK_VELOCITY_INTERCEPT_DEVELOPMENT_RECORDED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            review_hash=a.review_hash,
            outcome_contract_hash=outcome_contract()["contract_hash"],
            records=records,
            reference_only_passes=sum(pair[0]["outcome"]["task_pass"] for pair in pairs),
            reference_plus_torque_passes=sum(pair[1]["outcome"]["task_pass"] for pair in pairs),
            paired_initial_states_equal=all(
                pair[0]["initial_state_hash"] == pair[1]["initial_state_hash"] for pair in pairs
            ),
            cold_template_unchanged=unchanged,
            new_neural_calls=sum(r["run"]["neural_calls"] for r in records),
            new_native_steps=sum(r["run"]["native_steps"] for r in records),
            reserved_native_steps=reserved,
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
