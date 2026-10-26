"""One direction-reference probe and one model proposal for failed right ball."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--review-hash", required=True)
    p.add_argument("--authority", type=Path, required=True)
    p.add_argument("--authority-hash", required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--external-source", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    authority_path = a.authority / "complete.json"
    source_path = a.source / "complete.json"
    if (
        a.output.exists()
        or any(pth.is_symlink() for pth in (a.review, authority_path, source_path))
        or digest(a.review) != a.review_hash
        or digest(authority_path) != a.authority_hash
    ):
        raise ValueError("unused experiment and exact earlier diagnosis/review required")
    proof = json.loads(a.review.read_text())
    prior = json.loads(source_path.read_text())
    authority = json.loads(authority_path.read_text())
    if (
        proof["status"] != "CFTRACK_INTERCEPT_ALL8_NN_AND_MANUAL_NATIVE_EXACT"
        or proof["source_hash"] != digest(source_path)
        or proof["new_neural_calls"] != 2400
        or proof["new_native_steps"] != 24000
        or authority["status"] != "CFTRACK_RIGHT_INTERCEPT_AUTHORITY_DEVELOPMENT_RECORDED"
        or authority["review_hash"] != a.review_hash
        or authority["local_planning_margin_accepted"]
        or authority["candidate_failure"] is None
        or not authority["parent_prefix_exact"]
        or not authority["cold_template_unchanged"]
        or authority["new_neural_calls"] != 800
        or authority["new_native_steps"] != 8000
    ):
        raise ValueError(
            "verified right failure and separately bounded no-solution result required"
        )
    source = next(
        r
        for r in prior["records"]
        if r["case"] == "right" and r["arm"] == "intercept-plus-fixed-nominal-torque"
    )
    if source["outcome"]["task_pass"]:
        raise ValueError("the declared right-side development failure must remain a failure")
    pins = dict(proof["input_file_hashes"])
    if any(
        path in pins and pins[path] != commitment
        for path, commitment in authority["input_file_hashes"].items()
    ):
        raise ValueError("independent review and authority source commitments disagree")
    pins.update(authority["input_file_hashes"])
    pins[str(a.review)] = a.review_hash
    pins[str(authority_path)] = a.authority_hash
    if pins.get(str(source_path)) != digest(source_path):
        raise ValueError("exact previous right-side source required")
    external = a.external_source.resolve()
    if str(external / "policy/omnicontact/OmniContact.py") not in pins:
        raise ValueError("source-pinned original policy required")
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

    velocity = (0.0, -0.1, 0.0)
    spin = (-velocity[1] / 0.11, velocity[0] / 0.11, 0.0)
    validate_cold_ball_velocity(velocity)
    validate_cold_ball_angular_velocity(spin)
    plan_data = dict(source["plan"])
    plan_data["fraction_effort_hg"] = tuple(plan_data["fraction_effort_hg"])
    frozen = CFtrackTorqueIntentPlan(**plan_data)
    frozen.offsets()
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            ball_velocity_m_s=velocity,
            ball_angular_velocity_rad_s=spin,
            arm_order=["parent", "aim-minus-0.4", "one-model-candidate"],
            probe_goal_y_offset_m=-0.4,
            allowed_goal_y_offset_m=[-0.7, 0.0],
            planning_only_aim_target_true_goal_restored_before_inference=True,
            nominal_torque_parameters_frozen=True,
            max_nn_calls=480,
            max_native_steps=4800,
            local_horizon_controls=160,
            planning_forward_minimum_m=1.2,
            planning_direction_maximum=0.2,
            one_probe_one_model_candidate=True,
            no_goal_offset_grid=True,
            unchanged_exam_and_torque_caps=True,
            privileged_sim_velocity_for_reference=True,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import numpy as np

    from rosclaw_soccer.rsi.cftrack_motor_authority import summarize_motor_future
    from rosclaw_soccer.rsi.cftrack_reference_aim_response import propose_reference_aim
    from rosclaw_soccer.rsi.cftrack_torque_trial import complete_torque_trial, run_cold_torque_trial
    from rosclaw_soccer.rsi.cftrack_velocity_intercept import condition_velocity_intercept
    from rosclaw_soccer.rsi.cold_cftrack_trial import cold_cftrack_trial
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_cftrack_native_context import build_cold_cftrack

    template, _, _ = build_cold_cftrack(scenes[0].read_text(), external, direct_approach=True)
    template_hash = cftrack_state_hash(template.policy)
    rows = []
    reserved = calls = steps = 0

    def execute(name, offset):
        nonlocal reserved, calls, steps
        if reserved + 1600 > 4800:
            raise ValueError("fixed native budget exhausted with no refunds")
        player = cold_cftrack_trial(
            template, ball_xy=(1.0, 0.0), ball_velocity=velocity, ball_angular_velocity=spin
        )
        reference = condition_velocity_intercept(player, reference_goal_offset_y_m=offset)
        plan = replace(frozen, parent_policy_hash=cftrack_state_hash(player.policy))
        reserved += 1600
        write_once(
            a.output / (name + "-before.json"),
            dict(reference=reference, plan=asdict(plan), reserved=reserved),
        )
        raw, run = run_cold_torque_trial(player, plan)
        calls += run["neural_calls"]
        steps += run["native_steps"]
        path = a.output / (name + ".npz")
        np.savez_compressed(path, **raw)
        pins[str(path)] = digest(path)
        metrics = summarize_motor_future(
            dict(
                raw,
                controls=run["controls"],
                runtime_failure=run["runtime_failure"],
                body_failed=run["body_failed"],
            ),
            ball_address=player.layout.bq,
            root_address=player.layout.rq,
        )
        if not complete_torque_trial(run):
            metrics["private_safe"] = False
            metrics["local_post60_conditions_met"] = False
        row = dict(
            name=name,
            reference=reference,
            plan=asdict(plan),
            run=run,
            metrics=metrics,
            initial_state_hash=hash_json(
                dict(qpos=raw["initial_qpos"].tolist(), qvel=raw["initial_qvel"].tolist())
            ),
            artifact=dict(path=path.name, hash=pins[str(path)]),
        )
        rows.append(row)
        write_once(a.output / (name + ".json"), row)
        return raw, row

    parent_raw, parent = execute("parent", 0.0)
    support = model = candidate = failure = None
    if parent["metrics"]["private_safe"] and parent["metrics"]["post60_observed"]:
        _, support = execute("aim-minus-0.4", -0.4)
        try:
            model = propose_reference_aim(parent["metrics"], support["metrics"])
            write_once(a.output / "candidate-model-before.json", dict(model=model))
            _, candidate = execute("candidate", float(model["reference_goal_offset_y_m"]))
        except ValueError as error:
            failure = dict(type=type(error).__name__, message=str(error))
    else:
        failure = dict(type="UnsafeOrCensoredParent", message="no safe complete anchor")
    source_raw = a.source / source["artifact"]["path"]
    if (
        source_raw.parent != a.source
        or source_raw.is_symlink()
        or pins.get(str(source_raw)) != digest(source_raw)
    ):
        raise ValueError("bound original parent required after all proposals")
    with np.load(source_raw, allow_pickle=False) as saved:
        prefix = len(parent_raw["joint_targets"]) == 160 and all(
            np.array_equal(parent_raw[key], saved[key][: len(parent_raw[key])])
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
    unchanged = (
        cftrack_state_hash(template.policy) == template_hash and template.world.native_steps == 0
    )
    same_initial = all(row["initial_state_hash"] == parent["initial_state_hash"] for row in rows)
    m = candidate["metrics"] if candidate is not None else None
    accepted = bool(
        m
        and m["private_safe"]
        and m["post60_observed"]
        and m["forward_60_m"] >= 1.2
        and m["lateral_forward_ratio"] <= 0.2
    )
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_RIGHT_REFERENCE_AIM_DEVELOPMENT_RECORDED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            review_hash=a.review_hash,
            authority_hash=a.authority_hash,
            rows=rows,
            candidate_model=model,
            candidate_failure=failure,
            local_planning_margin_accepted=accepted,
            original_parent_prefix_exact=prefix,
            original_parent_reference_exact=hash_json(parent["reference"])
            == hash_json(source["reference"]),
            paired_initial_states_equal=same_initial,
            cold_template_unchanged=unchanged,
            new_neural_calls=calls,
            new_native_steps=steps,
            reserved_native_steps=reserved,
            neural_optimizer_updates=0,
            full_h300_not_evaluated=True,
            independent_review_pending=True,
            fresh_not_evaluated=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
