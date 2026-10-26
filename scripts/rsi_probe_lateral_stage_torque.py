"""Measure bounded bilateral hip-roll torque responses before the stage gate."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

ARMS = (
    ("left-positive", 1, 0.125),
    ("left-negative", 1, -0.125),
    ("right-positive", 7, 0.125),
    ("right-negative", 7, -0.125),
)


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
        raise ValueError("unused output and exact independently reviewed source required")
    review, source = json.loads(a.review.read_text()), json.loads(source_path.read_text())
    if (
        review["status"] != "CFTRACK_STAGED_LATERAL_NN_NATIVE_AND_BOUNDARY_EXACT"
        or review["source_hash"] != digest(source_path)
        or review["new_neural_calls"] != 314
        or review["new_native_steps"] != 3140
        or source["status"] != "CFTRACK_STAGED_LATERAL_DEVELOPMENT_RECORDED"
    ):
        raise ValueError("complete independent old stage audit required")
    prior = next(row for row in source["records"] if row["case"] == "new-left")
    if (
        prior["reference"]["stage_controls"] != 14
        or prior["run"]["controls"] != 14
        or prior["outcome"]["event_status"] != "MEASURED_STAGE_BOUNDARY_NOT_READY"
    ):
        raise ValueError("exact measured left-stage failure anchor required")
    pins = dict(review["input_file_hashes"])
    pins[str(a.review)] = a.review_hash
    external = a.external_source.resolve()
    if str(external / "policy/omnicontact/OmniContact.py") not in pins:
        raise ValueError("source-bound author model required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique canonical scene required")
    import numpy as np

    from rosclaw_soccer.rsi.cftrack_stage_boundary import CFtrackStageBoundary
    from rosclaw_soccer.rsi.cftrack_staged_lateral_reference import (
        condition_staged_lateral_intercept,
    )
    from rosclaw_soccer.rsi.cftrack_torque_intent import CFtrackTorqueIntentPlan
    from rosclaw_soccer.rsi.cftrack_torque_trial import run_cold_torque_trial
    from rosclaw_soccer.rsi.cold_cftrack_trial import cold_cftrack_trial
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_cftrack_native_context import build_cold_cftrack

    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            review_hash=a.review_hash,
            input_file_hashes=pins,
            arms=ARMS,
            effort_fraction_absolute=0.125,
            torque_start_control=0,
            torque_peak_control=24,
            stage_control=14,
            maximum_controls_each=15,
            maximum_native_steps=600,
            maximum_nn_calls=60,
            goal_is_local_stage_authority_not_kicking_success=True,
            original_measured_stage_guard_and_effort_caps_unchanged=True,
            no_candidate_selection_or_optimizer_updates=True,
            consumed_single_development_state=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    template, _, _ = build_cold_cftrack(scenes[0].read_text(), external, direct_approach=True)
    template_hash = cftrack_state_hash(template.policy)
    before_plan = dict(prior["plan"])
    before_plan["fraction_effort_hg"] = tuple(before_plan["fraction_effort_hg"])
    old_plan = CFtrackTorqueIntentPlan(**before_plan)
    guard_data = dict(prior["stage_boundary"])
    for key in ("waypoint_xy", "reference_ball_xyz", "actual_goal_xyz"):
        guard_data[key] = tuple(guard_data[key])
    stage = CFtrackStageBoundary(**guard_data)
    rows = []
    reserved = 0
    try:
        for name, axis, fraction in ARMS:
            reserved += 150
            velocity = (0.0, 0.12, 0.0)
            player = cold_cftrack_trial(
                template,
                ball_xy=(1.0, 0.0),
                ball_velocity=velocity,
                ball_angular_velocity=(-velocity[1] / 0.11, 0.0, 0.0),
            )
            initial = hash_json(
                dict(qpos=player.world.data.qpos.tolist(), qvel=player.world.data.qvel.tolist())
            )
            reference = condition_staged_lateral_intercept(player)
            if (
                initial != prior["initial_state_hash"]
                or hash_json(reference) != hash_json(prior["reference"])
                or cftrack_state_hash(player.policy) != old_plan.parent_policy_hash
            ):
                raise ValueError("exact previously reviewed initial body and reference required")
            fractions = [0.0] * 29
            fractions[axis] = fraction
            plan = replace(
                old_plan, start_control=0, peak_control=24, fraction_effort_hg=tuple(fractions)
            )
            plan.offsets()
            write_once(
                a.output / (name + "-before.json"),
                dict(
                    initial_state_hash=initial,
                    plan=asdict(plan),
                    stage_boundary=asdict(stage),
                    reserved_native_steps=reserved,
                ),
            )
            raw, run = run_cold_torque_trial(player, plan, controls=15, stage_boundary=stage)
            path = a.output / (name + ".npz")
            np.savez_compressed(path, **raw)
            pins[str(path)] = digest(path)
            receipt = run["stage_boundary_receipt"]
            baseline = prior["run"]["stage_boundary_receipt"]
            delta = None
            if receipt is not None:
                delta = dict(
                    root_xy_m=(
                        np.array(receipt["measured_root_pose"][:2])
                        - np.array(baseline["measured_root_pose"][:2])
                    ).tolist(),
                    waypoint_error_m=receipt["waypoint_error_m"] - baseline["waypoint_error_m"],
                    geometric_lateral_m=receipt["measured_geometry_to_planned_intercept"][
                        "lateral_offset_m"
                    ]
                    - baseline["measured_geometry_to_planned_intercept"]["lateral_offset_m"],
                )
            row = dict(
                arm=name,
                axis=axis,
                effort_fraction=fraction,
                initial_state_hash=initial,
                reference=reference,
                plan=asdict(plan),
                run=run,
                difference_from_reviewed_parent=delta,
                task_outcome_not_measured=True,
                artifact=dict(path=path.name, hash=pins[str(path)]),
            )
            rows.append(row)
            write_once(a.output / (name + ".json"), row)
            print(
                json.dumps(
                    dict(
                        arm=name,
                        controls=run["controls"],
                        body_failed=run["body_failed"],
                        ready=None if receipt is None else receipt["ready"],
                        delta=delta,
                    )
                ),
                flush=True,
            )
        if cftrack_state_hash(template.policy) != template_hash or template.world.native_steps:
            raise ValueError("cold template unchanged required")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="LATERAL_STAGE_TORQUE_AUTHORITY_RECORDED_NOT_TASK",
                source_head=a.expected_head,
                review_hash=a.review_hash,
                input_file_hashes=pins,
                records=rows,
                reserved_native_steps=reserved,
                new_native_steps=sum(r["run"]["native_steps"] for r in rows),
                new_neural_calls=sum(r["run"]["neural_calls"] for r in rows),
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
                status="STAGE_AUTHORITY_FAILED_CLOSED",
                source_head=a.expected_head,
                completed_arms=len(rows),
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
