"""Probe measured-stage-gated lateral approach on two consumed failure cases."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

CASES = (("new-left", (0.0, 0.12, 0.0)), ("new-right", (0.0, -0.12, 0.0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-hash", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--external-source", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    source_path = a.source / "complete.json"
    if (
        a.output.exists()
        or a.review.is_symlink()
        or source_path.is_symlink()
        or digest(a.review) != a.review_hash
    ):
        raise ValueError("unused staged experiment and exact reviewed source required")
    proof, source = json.loads(a.review.read_text()), json.loads(source_path.read_text())
    if (
        proof["status"] != "CFTRACK_GUARDED_AIM_NN_NATIVE_AND_REFUSALS_EXACT"
        or proof["source_hash"] != digest(source_path)
        or proof["new_neural_calls"] != 2700
        or proof["new_native_steps"] != 27000
        or not source["all_pair_initial_states_match"]
        or not source["cold_template_unchanged"]
    ):
        raise ValueError("complete independently reviewed original velocity study required")
    pins = dict(proof["input_file_hashes"])
    pins[str(a.review)] = a.review_hash
    if pins.get(str(source_path)) != digest(source_path):
        raise ValueError("bound original candidate physical report required")
    external = a.external_source.resolve()
    if str(external / "policy/omnicontact/OmniContact.py") not in pins:
        raise ValueError("source-bound author model required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(path) for path in pins if path.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique canonical scene required")
    from rosclaw_soccer.rsi.cftrack_torque_intent import CFtrackTorqueIntentPlan
    from rosclaw_soccer.rsi.cold_cftrack_trial import (
        validate_cold_ball_angular_velocity,
        validate_cold_ball_velocity,
    )

    source_right = next(
        r for r in source["records"] if r["case"] == "new-right" and r["arm"] == "guarded"
    )
    plan_data = dict(source_right["plan"])
    plan_data["fraction_effort_hg"] = tuple(plan_data["fraction_effort_hg"])
    frozen = CFtrackTorqueIntentPlan(**plan_data)
    frozen.offsets()
    for _, velocity in CASES:
        validate_cold_ball_velocity(velocity)
        validate_cold_ball_angular_velocity((-velocity[1] / 0.11, velocity[0] / 0.11, 0.0))
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            cases=CASES,
            stage_reference_step_m=0.005,
            planned_lateral_domain_m=[0.2, 0.3],
            planned_remaining_lateral_m=0.15,
            maximum_stage_displacement_m=0.15,
            actual_stage_waypoint_tolerance_m=0.08,
            actual_ball_projection_tolerance_m=0.08,
            original_geometric_guard_must_hold_on_actual_body_before_continuation=True,
            stage_check_is_not_complete_physical_safety_proof=True,
            base_frozen_torque_plan=asdict(frozen),
            torque_start_shift_by_stage_controls=True,
            max_nn_calls=600,
            max_native_steps=6000,
            full_budget_reserved_per_attempt=True,
            current_two_failure_cases_consumed_development_not_fresh=True,
            no_waypoint_or_amplitude_grid=True,
            original_h300_and_effort_caps_unchanged=True,
            sim_privileged_initial_ball_velocity=True,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import numpy as np

    from rosclaw_soccer.rsi.cftrack_stage_boundary import CFtrackStageBoundary
    from rosclaw_soccer.rsi.cftrack_staged_lateral_reference import (
        condition_staged_lateral_intercept,
    )
    from rosclaw_soccer.rsi.cftrack_torque_trial import complete_torque_trial, run_cold_torque_trial
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
        reserved += 3000
        spin = (-velocity[1] / 0.11, velocity[0] / 0.11, 0.0)
        player = cold_cftrack_trial(
            template, ball_xy=(1.0, 0.0), ball_velocity=velocity, ball_angular_velocity=spin
        )
        initial_hash = hash_json(
            dict(qpos=player.world.data.qpos.tolist(), qvel=player.world.data.qvel.tolist())
        )
        reference = condition_staged_lateral_intercept(player)
        controls = reference["stage_controls"]
        waypoint = reference["geometry"]["planned_waypoint"]
        prediction = reference["prediction"]
        guard = CFtrackStageBoundary(
            controls,
            tuple(float(v) for v in waypoint[:2]),
            tuple(float(v) for v in prediction["predicted_intercept_position"]),
            tuple(float(v) for v in player.policy.goal_pos),
            float(prediction["predicted_horizon_s"]),
        )
        plan = replace(
            frozen,
            start_control=frozen.start_control + controls,
            parent_policy_hash=cftrack_state_hash(player.policy),
        )
        write_once(
            a.output / (name + "-before.json"),
            dict(
                reference=reference,
                stage_boundary=asdict(guard),
                plan=asdict(plan),
                reserved=reserved,
            ),
        )
        raw, run = run_cold_torque_trial(player, plan, controls=300, stage_boundary=guard)
        path = a.output / (name + ".npz")
        np.savez_compressed(path, **raw)
        pins[str(path)] = digest(path)
        outcome = dict(event_status="INCOMPLETE_NATIVE_EPISODE", task_pass=False, safe=None)
        if (
            run["runtime_failure"] is not None
            and run["runtime_failure"]["type"] == "STAGE_BOUNDARY_NOT_READY"
        ):
            outcome["event_status"] = "MEASURED_STAGE_BOUNDARY_NOT_READY"
        elif run["body_failed"]:
            outcome["event_status"] = "BODY_FAILED_BEFORE_H300"
            outcome["safe"] = False
        elif complete_torque_trial(run, controls=300):
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
        old = next(r for r in source["records"] if r["case"] == name and r["arm"] == "guarded")
        verified = next(
            r for r in proof["independently_recomputed_comparisons"] if r["case"] == name
        )
        if old["initial_state_hash"] != initial_hash:
            raise ValueError(
                "staged and independently reviewed failure must share exact initial state"
            )
        row = dict(
            case=name,
            ball_velocity_m_s=velocity,
            initial_state_hash=initial_hash,
            reference=reference,
            stage_boundary=asdict(guard),
            plan=asdict(plan),
            run=run,
            outcome=outcome,
            verified_previous_outcome=verified["candidate_outcome"],
            artifact=dict(path=path.name, hash=pins[str(path)]),
        )
        records.append(row)
        write_once(a.output / (name + ".json"), row)
    unchanged = (
        cftrack_state_hash(template.policy) == template_hash and template.world.native_steps == 0
    )
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_STAGED_LATERAL_DEVELOPMENT_RECORDED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            review_hash=a.review_hash,
            outcome_contract_hash=outcome_contract()["contract_hash"],
            records=records,
            new_neural_calls=sum(r["run"]["neural_calls"] for r in records),
            new_native_steps=sum(r["run"]["native_steps"] for r in records),
            reserved_native_steps=reserved,
            cold_template_unchanged=unchanged,
            neural_optimizer_updates=0,
            independent_review_pending=True,
            fresh_not_evaluated=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
