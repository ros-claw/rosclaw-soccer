"""Physical H300 retention and prospective velocity probes for guarded aim."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

CASES = (
    ("dev-forward", "forward", (0.1, 0.0, 0.0), False),
    ("dev-backward", "backward", (-0.1, 0.0, 0.0), False),
    ("dev-left", "left", (0.0, 0.1, 0.0), False),
    ("dev-right", "right", (0.0, -0.1, 0.0), False),
    ("new-forward", "forward", (0.12, 0.0, 0.0), True),
    ("new-backward", "backward", (-0.12, 0.0, 0.0), True),
    ("new-left", "left", (0.0, 0.12, 0.0), True),
    ("new-right", "right", (0.0, -0.12, 0.0), True),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-review", type=Path, required=True)
    parser.add_argument("--recovery-review-hash", required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--baseline-review", type=Path, required=True)
    parser.add_argument("--baseline-review-hash", required=True)
    parser.add_argument("--external-source", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    recovery_path = a.recovery / "complete.json"
    if (
        a.output.exists()
        or any(p.is_symlink() for p in (a.recovery_review, a.baseline_review, recovery_path))
        or digest(a.recovery_review) != a.recovery_review_hash
        or digest(a.baseline_review) != a.baseline_review_hash
    ):
        raise ValueError("unused trial and exact prior independent reviews required")
    review = json.loads(a.recovery_review.read_text())
    baseline = json.loads(a.baseline_review.read_text())
    recovery = json.loads(recovery_path.read_text())
    if (
        review["status"] != "CFTRACK_AIM_RETENTION_3_NN_NATIVE_AND_GUARD_EXACT"
        or review["source_hash"] != digest(recovery_path)
        or review["new_neural_calls"] != 900
        or review["new_native_steps"] != 9000
        or review["left_guard_reconstructed"] is not True
        or baseline["status"] != "CFTRACK_INTERCEPT_ALL8_NN_AND_MANUAL_NATIVE_EXACT"
        or baseline["new_neural_calls"] != 2400
        or baseline["new_native_steps"] != 24000
        or recovery["regressions"] != ["left"]
        or recovery["rescues"] != ["right"]
    ):
        raise ValueError("verified source rescue and guard-caused retention loss required")
    pins = dict(review["input_file_hashes"])
    if any(k in pins and pins[k] != v for k, v in baseline["input_file_hashes"].items()):
        raise ValueError("independent source commitments disagree")
    pins.update(baseline["input_file_hashes"])
    pins[str(a.recovery_review)] = a.recovery_review_hash
    pins[str(a.baseline_review)] = a.baseline_review_hash
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

    right = next(r for r in recovery["records"] if r["case"] == "right")
    value = dict(right["plan"])
    value["fraction_effort_hg"] = tuple(value["fraction_effort_hg"])
    frozen = CFtrackTorqueIntentPlan(**value)
    frozen.offsets()
    for _, _, velocity, _ in CASES:
        validate_cold_ball_velocity(velocity)
        validate_cold_ball_angular_velocity((-velocity[1] / 0.11, velocity[0] / 0.11, 0.0))
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            cases=CASES,
            old_states_guarded_against_reviewed_baseline=True,
            new_velocity_states_baseline_and_guarded_paired=True,
            new_velocities_prospective_not_complete_fresh_exam=True,
            requested_reference_goal_offset_y_m=-0.4,
            fallback_goal_offset_y_m=0.0,
            planning_guard_margin=1e-6,
            fixed_nominal_torque_plan=asdict(frozen),
            max_physical_arms=12,
            max_nn_calls=3600,
            max_native_steps=36000,
            charge_full_budget_before_each_arm_even_if_reference_rejected=True,
            no_per_case_parameter_optimization=True,
            no_reference_or_amplitude_grid=True,
            original_h300_and_effort_caps_unchanged=True,
            initial_velocity_is_privileged_sim_input=True,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import numpy as np

    from rosclaw_soccer.rsi.cftrack_guarded_reference_aim import condition_guarded_reference_aim
    from rosclaw_soccer.rsi.cftrack_torque_trial import complete_torque_trial, run_cold_torque_trial
    from rosclaw_soccer.rsi.cftrack_velocity_intercept import condition_velocity_intercept
    from rosclaw_soccer.rsi.cold_cftrack_trial import cold_cftrack_trial
    from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch, outcome_contract
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_cftrack_native_context import build_cold_cftrack

    baseline_sources = [
        Path(path)
        for path, value in pins.items()
        if value == baseline["source_hash"] and path.endswith("/complete.json")
    ]
    if len(baseline_sources) != 1 or digest(baseline_sources[0]) != baseline["source_hash"]:
        raise ValueError("unique independently reviewed baseline source required")
    baseline_source = json.loads(baseline_sources[0].read_text())
    template, _, _ = build_cold_cftrack(scenes[0].read_text(), external, direct_approach=True)
    template_hash = cftrack_state_hash(template.policy)
    records = []
    reserved = calls = steps = 0
    for case, original_case, velocity, is_new in CASES:
        spin = (-velocity[1] / 0.11, velocity[0] / 0.11, 0.0)
        for arm in ("baseline", "guarded") if is_new else ("guarded",):
            if reserved + 3000 > 36000:
                raise ValueError("predeclared native budget exhausted without refunds")
            reserved += 3000
            player = cold_cftrack_trial(
                template, ball_xy=(1.0, 0.0), ball_velocity=velocity, ball_angular_velocity=spin
            )
            initial_hash = hash_json(
                dict(qpos=player.world.data.qpos.tolist(), qvel=player.world.data.qvel.tolist())
            )
            error = None
            try:
                reference = (
                    condition_guarded_reference_aim(player, requested_offset_y_m=-0.4)
                    if arm == "guarded"
                    else dict(reference=condition_velocity_intercept(player), selection=None)
                )
            except ValueError as exc:
                error = dict(type=type(exc).__name__, message=str(exc))
                reference = None
            if error is not None:
                is_guard = (
                    "neither requested nor original" in error["message"]
                    or "direct approach only qualified" in error["message"]
                )
                row = dict(
                    case=case,
                    original_case=original_case,
                    new_velocity=is_new,
                    arm=arm,
                    ball_velocity_m_s=velocity,
                    initial_state_hash=initial_hash,
                    reference=None,
                    planning_error=error,
                    run=dict(controls=0, native_steps=0, neural_calls=0),
                    outcome=dict(
                        event_status="REFERENCE_GUARD_REJECTED"
                        if is_guard
                        else "REFERENCE_PLAN_ERROR",
                        task_pass=False,
                        safe=None,
                    ),
                    artifact=None,
                )
                records.append(row)
                write_once(a.output / (case + "-" + arm + ".json"), row)
                continue
            plan = replace(frozen, parent_policy_hash=cftrack_state_hash(player.policy))
            write_once(
                a.output / (case + "-" + arm + "-before.json"),
                dict(reference=reference, plan=asdict(plan), reserved_native_steps=reserved),
            )
            raw, run = run_cold_torque_trial(player, plan, controls=300)
            calls += run["neural_calls"]
            steps += run["native_steps"]
            path = a.output / (case + "-" + arm + ".npz")
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
                case=case,
                original_case=original_case,
                new_velocity=is_new,
                arm=arm,
                ball_velocity_m_s=velocity,
                initial_state_hash=initial_hash,
                reference=reference,
                planning_error=None,
                plan=asdict(plan),
                run=run,
                outcome=outcome,
                artifact=dict(path=path.name, hash=pins[str(path)]),
            )
            records.append(row)
            write_once(a.output / (case + "-" + arm + ".json"), row)
    comparisons = []
    for case, old_case, _, is_new in CASES:
        candidate = next(r for r in records if r["case"] == case and r["arm"] == "guarded")
        if is_new:
            base = next(r for r in records if r["case"] == case and r["arm"] == "baseline")
            same_initial = base["initial_state_hash"] == candidate["initial_state_hash"]
            base_outcome = base["outcome"]
        else:
            old = next(
                r
                for r in baseline_source["records"]
                if r["case"] == old_case and r["arm"] == "intercept-plus-fixed-nominal-torque"
            )
            verified = next(
                r
                for r in baseline["records"]
                if r["case"] == old_case and r["arm"] == "intercept-plus-fixed-nominal-torque"
            )
            same_initial = (
                old["initial_state_hash"] == candidate["initial_state_hash"]
                and verified["complete"]
            )
            base_outcome = verified["outcome"]
        comparisons.append(
            dict(
                case=case,
                new_velocity=is_new,
                same_initial=same_initial,
                baseline_outcome=base_outcome,
                candidate_outcome=candidate["outcome"],
                rescue=not base_outcome["task_pass"] and candidate["outcome"]["task_pass"],
                regression=base_outcome["task_pass"] and not candidate["outcome"]["task_pass"],
            )
        )
    unchanged = (
        cftrack_state_hash(template.policy) == template_hash and template.world.native_steps == 0
    )
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_GUARDED_AIM_VELOCITY_DEVELOPMENT_AND_PROSPECTIVE_RECORDED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            outcome_contract_hash=outcome_contract()["contract_hash"],
            records=records,
            comparisons=comparisons,
            all_pair_initial_states_match=all(r["same_initial"] for r in comparisons),
            cold_template_unchanged=unchanged,
            new_neural_calls=calls,
            new_native_steps=steps,
            reserved_native_steps=reserved,
            neural_optimizer_updates=0,
            independent_nn_native_review_pending=True,
            full_fresh_exam_not_evaluated=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
