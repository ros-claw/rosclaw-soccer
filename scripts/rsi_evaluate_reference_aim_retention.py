"""Test one posthoc aim probe on H300 and previously successful moving cases."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--pilot-hash", required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--review-hash", required=True)
    p.add_argument("--external-source", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_probe_cftrack_moving_ball import CASES

    clean(a.expected_head)
    pilot_path = a.pilot / "complete.json"
    if (
        a.output.exists()
        or a.review.is_symlink()
        or pilot_path.is_symlink()
        or digest(a.review) != a.review_hash
        or digest(pilot_path) != a.pilot_hash
    ):
        raise ValueError("unused retention study and exact source evidence required")
    pilot = json.loads(pilot_path.read_text())
    proof = json.loads(a.review.read_text())
    if (
        pilot["status"] != "CFTRACK_RIGHT_REFERENCE_AIM_DEVELOPMENT_RECORDED"
        or pilot["local_planning_margin_accepted"]
        or pilot["candidate_failure"] is None
        or not pilot["original_parent_prefix_exact"]
        or not pilot["original_parent_reference_exact"]
        or not pilot["paired_initial_states_equal"]
        or not pilot["cold_template_unchanged"]
        or pilot["review_hash"] != a.review_hash
        or proof["status"] != "CFTRACK_INTERCEPT_ALL8_NN_AND_MANUAL_NATIVE_EXACT"
        or proof["new_neural_calls"] != 2400
        or proof["new_native_steps"] != 24000
    ):
        raise ValueError("reviewed failure and original stricter-planning abstention required")
    probe = next(r for r in pilot["rows"] if r["name"] == "aim-minus-0.4")
    if (
        not probe["metrics"]["private_safe"]
        or not probe["metrics"]["local_post60_conditions_met"]
        or probe["reference"].get("reference_aim", {}).get("planning_only_goal_offset_y_m") != -0.4
    ):
        raise ValueError("explicit posthoc development probe must meet original local conditions")
    pins = dict(proof["input_file_hashes"])
    if any(
        path in pins and pins[path] != value for path, value in pilot["input_file_hashes"].items()
    ):
        raise ValueError("pilot and independent review source commitments disagree")
    pins.update(pilot["input_file_hashes"])
    pins[str(a.review)] = a.review_hash
    pins[str(pilot_path)] = a.pilot_hash
    external = a.external_source.resolve()
    if str(external / "policy/omnicontact/OmniContact.py") not in pins:
        raise ValueError("source-bound original policy required")
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

    plan_data = dict(probe["plan"])
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
            posthoc_nomination_from_local_original_threshold=True,
            stricter_predeclared_planning_margin_failed_not_relaxed=True,
            frozen_reference_goal_offset_y_m=-0.4,
            uniform_for_all_four_cases=True,
            reference_goal_restored_before_nn=True,
            nominal_torque_parameters_frozen=True,
            horizon_controls=300,
            maximum_nn_calls=1200,
            maximum_native_steps=12000,
            old_reviewed_baselines_used_only_for_comparison=True,
            used_states_not_fresh=True,
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
    right_raw = None
    for name, velocity in CASES:
        spin = (-velocity[1] / 0.11, velocity[0] / 0.11, 0.0)
        player = cold_cftrack_trial(
            template, ball_xy=(1.0, 0.0), ball_velocity=velocity, ball_angular_velocity=spin
        )
        reference = condition_velocity_intercept(player, reference_goal_offset_y_m=-0.4)
        plan = replace(frozen, parent_policy_hash=cftrack_state_hash(player.policy))
        write_once(a.output / (name + "-before.json"), dict(reference=reference, plan=asdict(plan)))
        raw, run = run_cold_torque_trial(player, plan, controls=300)
        path = a.output / (name + ".npz")
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
        baseline = next(
            r
            for r in proof["records"]
            if r["case"] == name and r["arm"] == "intercept-plus-fixed-nominal-torque"
        )
        if not baseline["complete"]:
            raise ValueError("independently reviewed paired baseline required")
        row = dict(
            case=name,
            ball_velocity_m_s=velocity,
            ball_angular_velocity_rad_s=spin,
            reference=reference,
            plan=asdict(plan),
            run=run,
            outcome=outcome,
            verified_baseline_outcome=baseline["outcome"],
            initial_state_hash=hash_json(
                dict(qpos=raw["initial_qpos"].tolist(), qvel=raw["initial_qvel"].tolist())
            ),
            artifact=dict(path=path.name, hash=pins[str(path)]),
        )
        records.append(row)
        write_once(a.output / (name + ".json"), row)
        if name == "right":
            right_raw = raw
    old_path = a.pilot / probe["artifact"]["path"]
    if (
        old_path.parent != a.pilot
        or old_path.is_symlink()
        or pins.get(str(old_path)) != digest(old_path)
    ):
        raise ValueError("bound nominated probe required after all new trials")
    with np.load(old_path, allow_pickle=False) as saved:
        prefix = (
            right_raw is not None
            and len(right_raw["joint_targets"]) >= 160
            and all(
                np.array_equal(right_raw[key][: len(saved[key])], saved[key])
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
    baseline_sources = [
        Path(path)
        for path, value in pins.items()
        if value == proof["source_hash"] and path.endswith("/complete.json")
    ]
    if len(baseline_sources) != 1 or digest(baseline_sources[0]) != proof["source_hash"]:
        raise ValueError("unique bound independently reviewed baseline source required")
    baseline_source = json.loads(baseline_sources[0].read_text())
    pairs_equal = all(
        row["initial_state_hash"]
        == next(
            old["initial_state_hash"]
            for old in baseline_source["records"]
            if old["case"] == row["case"] and old["arm"] == "intercept-plus-fixed-nominal-torque"
        )
        for row in records
    )
    unchanged = (
        cftrack_state_hash(template.policy) == template_hash and template.world.native_steps == 0
    )
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_REFERENCE_AIM_RETENTION_DEVELOPMENT_RECORDED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            pilot_hash=a.pilot_hash,
            review_hash=a.review_hash,
            outcome_contract_hash=outcome_contract()["contract_hash"],
            records=records,
            candidate_passes=sum(r["outcome"]["task_pass"] for r in records),
            verified_baseline_passes=sum(
                r["verified_baseline_outcome"]["task_pass"] for r in records
            ),
            rescues=[
                r["case"]
                for r in records
                if not r["verified_baseline_outcome"]["task_pass"] and r["outcome"]["task_pass"]
            ],
            regressions=[
                r["case"]
                for r in records
                if r["verified_baseline_outcome"]["task_pass"] and not r["outcome"]["task_pass"]
            ],
            nominated_right_probe_prefix_exact=prefix,
            independently_reviewed_baseline_initial_states_match=pairs_equal,
            cold_template_unchanged=unchanged,
            new_neural_calls=sum(r["run"]["neural_calls"] for r in records),
            new_native_steps=sum(r["run"]["native_steps"] for r in records),
            neural_optimizer_updates=0,
            posthoc_development_nomination=True,
            stricter_planning_margin_not_relaxed=True,
            independent_review_pending=True,
            fresh_not_evaluated=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
