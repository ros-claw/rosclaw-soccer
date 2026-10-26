"""Probe four axes once and nominate one bounded response for the failed right ball."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path


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
    if (
        a.output.exists()
        or a.review.is_symlink()
        or source_path.is_symlink()
        or digest(a.review) != a.review_hash
    ):
        raise ValueError("unused right-side study and independently reviewed source required")
    review = json.loads(a.review.read_text())
    source = json.loads(source_path.read_text())
    if (
        review["status"] != "CFTRACK_INTERCEPT_ALL8_NN_AND_MANUAL_NATIVE_EXACT"
        or review["new_neural_calls"] != 2400
        or review["new_native_steps"] != 24000
        or review["source_hash"] != digest(source_path)
        or source["status"] != "CFTRACK_VELOCITY_INTERCEPT_DEVELOPMENT_RECORDED"
    ):
        raise ValueError("all independently reviewed intercept arms required")
    pins = dict(review["input_file_hashes"])
    pins[str(a.review)] = a.review_hash
    if pins.get(str(source_path)) != digest(source_path):
        raise ValueError("unchanged earlier intercept study required")
    external = a.external_source.resolve()
    if str(external / "policy/omnicontact/OmniContact.py") not in pins:
        raise ValueError("source-bound author policy required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(s) for s in pins if s.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique canonical physical scene required")
    from rosclaw_soccer.rsi.cftrack_multi_axis_torque_response import AXES, PROBE_FRACTION
    from rosclaw_soccer.rsi.cold_cftrack_trial import (
        validate_cold_ball_angular_velocity,
        validate_cold_ball_velocity,
    )

    velocity = (0.0, -0.1, 0.0)
    spin = (-velocity[1] / 0.11, velocity[0] / 0.11, 0.0)
    validate_cold_ball_velocity(velocity)
    validate_cold_ball_angular_velocity(spin)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            ball_velocity_m_s=velocity,
            ball_angular_velocity_rad_s=spin,
            axes_hg=AXES,
            positive_probe_effort_fraction=PROBE_FRACTION,
            arms=["parent", "probe-7", "probe-8", "probe-11", "probe-12", "one-model-candidate"],
            maximum_nn_calls=960,
            maximum_native_steps=9600,
            horizon_controls=160,
            forward_planning_minimum_m=1.2,
            direction_planning_maximum=0.2,
            coefficient_range_each=[0.0, 0.25],
            unsafe_or_censored_probe_axes_excluded=True,
            negative_axis_directions_not_assumed_safe=True,
            only_one_proposal_after_four_probes=True,
            no_amplitude_or_seed_grid=True,
            existing_development_state_not_fresh=True,
            unchanged_torque_caps=True,
            privileged_sim_velocity_for_reference=True,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import numpy as np

    from rosclaw_soccer.rsi.cftrack_motor_authority import summarize_motor_future
    from rosclaw_soccer.rsi.cftrack_multi_axis_torque_response import (
        propose_multi_axis_torque_response,
    )
    from rosclaw_soccer.rsi.cftrack_torque_intent import CFtrackTorqueIntentPlan
    from rosclaw_soccer.rsi.cftrack_torque_trial import complete_torque_trial, run_cold_torque_trial
    from rosclaw_soccer.rsi.cftrack_velocity_intercept import condition_velocity_intercept
    from rosclaw_soccer.rsi.cold_cftrack_trial import cold_cftrack_trial
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_cftrack_native_context import build_cold_cftrack

    template, _, _ = build_cold_cftrack(scenes[0].read_text(), external, direct_approach=True)
    template_hash = cftrack_state_hash(template.policy)
    reserved = calls = steps = 0
    all_rows = []

    def new_player():
        player = cold_cftrack_trial(
            template, ball_xy=(1.0, 0.0), ball_velocity=velocity, ball_angular_velocity=spin
        )
        reference = condition_velocity_intercept(player)
        return player, reference

    def execute(name, player, reference, plan):
        nonlocal reserved, calls, steps
        if reserved + 1600 > 9600:
            raise ValueError("declared native budget exhausted without refund")
        reserved += 1600
        write_once(a.output / (name + "-before.json"), dict(plan=asdict(plan), reserved=reserved))
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
            artifact=dict(path=path.name, hash=pins[str(path)]),
        )
        all_rows.append(row)
        write_once(a.output / (name + ".json"), row)
        return raw, row

    player, reference = new_player()
    body_hash = hash_json(player.layout.profile())
    zero = CFtrackTorqueIntentPlan(
        46, 24, (0.0,) * 29, body_hash, cftrack_state_hash(player.policy)
    )
    parent_raw, parent = execute("parent", player, reference, zero)
    first = parent["metrics"]["first_foot_control"]
    supports = {}
    model = candidate = failure = None

    def require_same_initial(player, reference):
        if (
            not np.array_equal(player.world.data.qpos, parent_raw["initial_qpos"])
            or not np.array_equal(player.world.data.qvel, parent_raw["initial_qvel"])
            or cftrack_state_hash(player.policy) != zero.parent_policy_hash
            or hash_json(reference) != hash_json(parent["reference"])
        ):
            raise ValueError(
                "all probes must use exactly the current parent cold state and reference"
            )

    if (
        parent["metrics"]["private_safe"]
        and parent["metrics"]["post60_observed"]
        and type(first) is int
        and 25 <= first <= 99
    ):
        for axis in AXES:
            player, reference = new_player()
            require_same_initial(player, reference)
            fractions = [0.0] * 29
            fractions[axis] = PROBE_FRACTION
            plan = CFtrackTorqueIntentPlan(
                first - 25, 24, tuple(fractions), body_hash, cftrack_state_hash(player.policy)
            )
            _, row = execute("probe-" + str(axis), player, reference, plan)
            supports[axis] = row["metrics"]
        try:
            player, reference = new_player()
            require_same_initial(player, reference)
            plan, model = propose_multi_axis_torque_response(
                parent["metrics"],
                supports,
                body_hash=body_hash,
                parent_policy_hash=cftrack_state_hash(player.policy),
            )
            write_once(
                a.output / "candidate-model-before.json", dict(model=model, plan=asdict(plan))
            )
            _, candidate = execute("candidate", player, reference, plan)
        except ValueError as error:
            failure = dict(type=type(error).__name__, message=str(error))
    else:
        failure = dict(type="UnsafeOrCensoredParent", message="no safe complete current anchor")
    old = next(
        r
        for r in source["records"]
        if r["case"] == "right" and r["arm"] == "intercept-reference-only"
    )
    old_path = a.source / old["artifact"]["path"]
    if (
        old_path.parent != a.source
        or old_path.is_symlink()
        or pins.get(str(old_path)) != digest(old_path)
    ):
        raise ValueError("verified reference parent source required after all proposals")
    with np.load(old_path, allow_pickle=False) as saved:
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
            status="CFTRACK_RIGHT_INTERCEPT_AUTHORITY_DEVELOPMENT_RECORDED",
            source_head=a.expected_head,
            source_hash=digest(source_path),
            review_hash=a.review_hash,
            input_file_hashes=pins,
            rows=all_rows,
            candidate_model=model,
            candidate_failure=failure,
            local_planning_margin_accepted=accepted,
            parent_prefix_exact=prefix,
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
