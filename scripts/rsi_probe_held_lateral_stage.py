"""Test one bounded actual-state-gated waypoint hold on two consumed SIM cases."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

CASES = (("new-left", (0.0, 0.12, 0.0)), ("new-right", (0.0, -0.12, 0.0)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--review-hash", required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--external-source", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    source_path = a.source / "complete.json"
    if a.output.exists() or a.review.is_symlink() or digest(a.review) != a.review_hash:
        raise ValueError("unused output and exact completed stage audit required")
    proof = json.loads(a.review.read_text())
    source = json.loads(source_path.read_text())
    if (
        proof["status"] != "CFTRACK_STAGED_LATERAL_NN_NATIVE_AND_BOUNDARY_EXACT"
        or proof["source_hash"] != digest(source_path)
        or proof["new_neural_calls"] != 314
        or proof["new_native_steps"] != 3140
        or [r["case"] for r in source["records"]] != [name for name, _ in CASES]
    ):
        raise ValueError("exact both-case independently reviewed stage source required")
    pins = dict(proof["input_file_hashes"])
    pins[str(a.review)] = a.review_hash
    external = a.external_source.resolve()
    if str(external / "policy/omnicontact/OmniContact.py") not in pins:
        raise ValueError("source-bound external author model required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(path) for path in pins if path.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique original canonical scene required")
    import numpy as np

    from rosclaw_soccer.rsi.cftrack_stage_boundary import CFtrackStageBoundary
    from rosclaw_soccer.rsi.cftrack_staged_lateral_reference import (
        condition_staged_lateral_intercept,
    )
    from rosclaw_soccer.rsi.cftrack_torque_intent import CFtrackTorqueIntentPlan
    from rosclaw_soccer.rsi.cftrack_torque_trial import complete_torque_trial, run_cold_torque_trial
    from rosclaw_soccer.rsi.cold_cftrack_trial import cold_cftrack_trial
    from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch, outcome_contract
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_cftrack_native_context import build_cold_cftrack

    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            review_hash=a.review_hash,
            cases=CASES,
            hold_reference_controls=20,
            predicted_contact_delay_s=0.4,
            stage_actual_geometry_and_ball_projection_gates_unchanged=True,
            original_h300_and_effort_caps_unchanged=True,
            max_neural_calls=600,
            max_native_steps=6000,
            full_budget_reserved_per_attempt=True,
            consumed_development_states_not_fresh=True,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    template, _, _ = build_cold_cftrack(scenes[0].read_text(), external, direct_approach=True)
    template_hash = cftrack_state_hash(template.policy)
    rows, reserved = [], 0
    try:
        for name, velocity in CASES:
            reserved += 3000
            old = next(r for r in source["records"] if r["case"] == name)
            before_plan = dict(old["plan"])
            before_plan["fraction_effort_hg"] = tuple(before_plan["fraction_effort_hg"])
            previous_plan = CFtrackTorqueIntentPlan(**before_plan)
            spin = (-velocity[1] / 0.11, velocity[0] / 0.11, 0.0)
            player = cold_cftrack_trial(
                template, ball_xy=(1.0, 0.0), ball_velocity=velocity, ball_angular_velocity=spin
            )
            initial_hash = hash_json(
                dict(qpos=player.world.data.qpos.tolist(), qvel=player.world.data.qvel.tolist())
            )
            if initial_hash != old["initial_state_hash"]:
                raise ValueError("same actual initial state as independent stage audit required")
            reference = condition_staged_lateral_intercept(player, hold_controls=20)
            stage = reference["stage_controls"]
            prediction = reference["prediction"]
            guard = CFtrackStageBoundary(
                stage,
                tuple(float(v) for v in reference["geometry"]["planned_waypoint"][:2]),
                tuple(float(v) for v in prediction["predicted_intercept_position"]),
                tuple(float(v) for v in player.policy.goal_pos),
                float(prediction["predicted_horizon_s"]),
            )
            plan = replace(
                previous_plan,
                start_control=previous_plan.start_control
                - old["reference"]["stage_controls"]
                + stage,
                parent_policy_hash=cftrack_state_hash(player.policy),
            )
            write_once(
                a.output / (name + "-before.json"),
                dict(
                    initial_state_hash=initial_hash,
                    reference=reference,
                    stage_boundary=asdict(guard),
                    plan=asdict(plan),
                    reserved_native_steps=reserved,
                ),
            )
            raw, run = run_cold_torque_trial(player, plan, controls=300, stage_boundary=guard)
            artifact = a.output / (name + ".npz")
            np.savez_compressed(artifact, **raw)
            pins[str(artifact)] = digest(artifact)
            outcome = dict(event_status="INCOMPLETE_NATIVE_EPISODE", task_pass=False, safe=None)
            if (
                run["runtime_failure"] is not None
                and run["runtime_failure"]["type"] == "STAGE_BOUNDARY_NOT_READY"
            ):
                outcome["event_status"] = "MEASURED_STAGE_BOUNDARY_NOT_READY"
            elif run["body_failed"]:
                outcome["event_status"], outcome["safe"] = "BODY_FAILED_BEFORE_H300", False
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
            row = dict(
                case=name,
                initial_state_hash=initial_hash,
                reference=reference,
                stage_boundary=asdict(guard),
                plan=asdict(plan),
                run=run,
                outcome=outcome,
                previously_verified_outcome=old["outcome"],
                artifact=dict(path=artifact.name, hash=pins[str(artifact)]),
            )
            rows.append(row)
            write_once(a.output / (name + ".json"), row)
            print(
                json.dumps(dict(case=name, outcome=outcome, controls=run["controls"])), flush=True
            )
        if cftrack_state_hash(template.policy) != template_hash or template.world.native_steps:
            raise ValueError("cold template must remain physically and neurally unchanged")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="CFTRACK_HELD_LATERAL_DEVELOPMENT_RECORDED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                review_hash=a.review_hash,
                outcome_contract_hash=outcome_contract()["contract_hash"],
                records=rows,
                cold_template_unchanged=True,
                reserved_native_steps=reserved,
                new_native_steps=sum(r["run"]["native_steps"] for r in rows),
                new_neural_calls=sum(r["run"]["controls"] for r in rows),
                neural_optimizer_updates=0,
                independent_review_pending=True,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="HELD_STAGE_FAILED_CLOSED",
                source_head=a.expected_head,
                completed_cases=len(rows),
                reserved_native_steps=reserved,
                error_type=type(error).__name__,
                reason=str(error),
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
