"""One fitted-response proposal per current state; private native NN veto only."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibration-hash", required=True)
    parser.add_argument("--verified-bank", type=Path, required=True)
    parser.add_argument("--verified-bank-hash", required=True)
    parser.add_argument("--development-bank", type=Path, required=True)
    parser.add_argument("--external-source", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    from rosclaw_soccer.rsi.cftrack_direct_approach import condition_direct_kick_approach
    from rosclaw_soccer.rsi.cftrack_motor_authority import summarize_motor_future
    from rosclaw_soccer.rsi.cftrack_response_teacher import (
        fit_motor_response,
        propose_response_motor_plan,
    )
    from rosclaw_soccer.rsi.cold_cftrack_trial import cold_cftrack_trial, validate_cold_ball_xy
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
    from rosclaw_soccer.rsi.shared_cftrack_forecast import SharedCFtrackForecaster
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_cftrack_native_context import build_cold_cftrack
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if (
        a.output.exists()
        or a.verified_bank.is_symlink()
        or digest(a.verified_bank) != a.verified_bank_hash
    ):
        raise ValueError("unused response pilot and exact independently verified bank required")
    proof = json.loads(a.verified_bank.read_text())
    calibration = a.calibration / "complete.json"
    if (
        proof["status"] != "CFTRACK_STATIC_BANK_NATIVE_REPLAY_EXACT"
        or proof["new_native_steps"] != 30000
        or len(proof["records"]) != 10
        or not all(r["exact"] for r in proof["records"])
    ):
        raise ValueError("full independent paired bank proof required")
    pins = dict(proof["input_file_hashes"])
    pins[str(a.verified_bank)] = a.verified_bank_hash
    bank_path = a.development_bank / "complete.json"
    if (
        calibration.is_symlink()
        or pins.get(str(calibration)) != a.calibration_hash
        or digest(calibration) != a.calibration_hash
        or bank_path.is_symlink()
        or pins.get(str(bank_path)) != digest(bank_path)
        or digest(bank_path) != proof["source_bank_hash"]
    ):
        raise ValueError("bound original calibration and development bank required")
    model = fit_motor_response(json.loads(calibration.read_text()), source_hash=a.calibration_hash)
    bank = json.loads(bank_path.read_text())
    cases = (
        ("nominal", (1.0, 0.0)),
        ("x-minus", (0.98, 0.0)),
        ("x-plus", (1.02, 0.0)),
        ("y-minus", (1.0, -0.02)),
        ("y-plus", (1.0, 0.02)),
    )
    for _, xy in cases:
        validate_cold_ball_xy(xy)
    external = a.external_source.resolve()
    if str(external / "policy/omnicontact/OmniContact.py") not in pins:
        raise ValueError("qualified external policy source required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("one qualified native body required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            cases=cases,
            fitted_model=asdict(model),
            maximum_private_native_steps=16000,
            maximum_private_nn_calls=1600,
            actual_native_steps=0,
            neural_optimizer_updates=0,
            proposals_per_case=1,
            no_amplitude_or_seed_grid=True,
            planning_minimum_forward_m=1.2,
            planning_direction_ratio=0.2,
            objective_weights=[1.0, 4.0],
            quadratic_regularization=0.1,
            unchanged_motor_offset_limit_rad=0.25,
            unchanged_extra_motor_velocity_rad_s=4.0,
            unchanged_torque_fraction=0.85,
            unsafe_calibration_directions_excluded=True,
            physical_target_box_linearized_from_current_parent=True,
            unmeasured_joint_coupling_requires_native_veto=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    template, body_hash, _ = build_cold_cftrack(
        scenes[0].read_text(), external, direct_approach=True
    )
    initial_hash = cftrack_state_hash(template.policy)
    records = []
    parent_traces = {}
    nn_calls = 0
    native_steps = 0
    attempted_steps = 0

    def save(name, future):
        path = a.output / (name + ".npz")
        np.savez_compressed(path, **{k: v for k, v in future.items() if isinstance(v, np.ndarray)})
        pins[str(path)] = digest(path)
        return dict(path=path.name, hash=pins[str(path)])

    for name, xy in cases:
        player = cold_cftrack_trial(template, ball_xy=xy)
        condition_direct_kick_approach(player.policy)
        before = cftrack_state_hash(player.policy)
        forecaster = SharedCFtrackForecaster(
            player,
            expected_policy_type=type(player.policy),
            expected_model_hash=body_hash,
            native_step_budget=3200,
        )
        parent = forecaster.forecast(controls=160)
        parent_traces[name] = parent
        metrics = summarize_motor_future(
            parent, ball_address=player.layout.bq, root_address=player.layout.rq
        )
        parent_artifact = save(name + "-parent", parent)
        proposal = None
        planning = None
        failure = None
        candidate = None
        candidate_metrics = None
        candidate_artifact = None
        try:
            proposal, planning = propose_response_motor_plan(
                model,
                metrics,
                origin_counter=player.policy.counter_step,
                donor_policy_hash=before,
                parent_targets=parent["nn_decoded_targets"],
                limits_hg=player.world.model.jnt_range[player.layout.joints].astype(np.float64),
            )
        except ValueError as error:
            failure = dict(type=type(error).__name__, message=str(error))
        if proposal is not None:
            candidate = forecaster.forecast(controls=160, motor_plan=proposal)
            candidate_metrics = summarize_motor_future(
                candidate, ball_address=player.layout.bq, root_address=player.layout.rq
            )
            candidate_artifact = save(name + "-candidate", candidate)
        accepted = bool(
            candidate_metrics is not None
            and candidate_metrics["private_safe"]
            and candidate_metrics["post60_observed"]
            and candidate_metrics["forward_60_m"] >= 1.2
            and candidate_metrics["lateral_forward_ratio"] <= 0.2
        )
        counters = forecaster.counters()
        nn_calls += counters["future_nn_calls"]
        native_steps += counters["integrated_native_steps"]
        attempted_steps += counters["attempted_native_steps"]
        row = dict(
            case=name,
            ball_xy=xy,
            parent_metrics=metrics,
            parent_artifact=parent_artifact,
            motor_plan=asdict(proposal) if proposal else None,
            planning=planning,
            proposal_failure=failure,
            candidate_metrics=candidate_metrics,
            candidate_artifact=candidate_artifact,
            candidate_runtime_failure=candidate["runtime_failure"] if candidate else None,
            candidate_controls=candidate["controls"] if candidate else 0,
            private_margin_accepted=accepted,
            full_h300_not_evaluated=True,
            future_counters=counters,
        )
        write_once(a.output / (name + ".json"), row)
        records.append(row)
        if (
            cftrack_state_hash(player.policy) != before
            or player.world.native_steps != 0
            or cftrack_state_hash(template.policy) != initial_hash
            or template.world.native_steps != 0
        ):
            raise ValueError("private teacher planning changed donor/template")
    # Recorded data enter only this post-generation provenance comparison.
    for name, _ in cases:
        old = next(r for r in bank["records"] if r["case"] == name and r["arm"] == "parent")
        path = a.development_bank / old["artifact"]["path"]
        if (
            path.parent != a.development_bank
            or path.is_symlink()
            or pins.get(str(path)) != digest(path)
        ):
            raise ValueError("bound actual parent required after planning")
        with np.load(path, allow_pickle=False) as raw:
            if any(
                not np.array_equal(
                    parent_traces[name][key], raw[key][: len(parent_traces[name][key])]
                )
                for key in [
                    "qpos",
                    "qvel",
                    "torque",
                    "ball_forces",
                    "joint_targets",
                    "nn_decoded_targets",
                ]
            ):
                raise ValueError(
                    "own current prediction differs from qualified actual parent prefix"
                )
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_RESPONSE_TEACHER_PRIVATE_PROPOSALS_RECORDED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            fitted_model=asdict(model),
            records=records,
            private_margin_accepted_cases=[
                r["case"] for r in records if r["private_margin_accepted"]
            ],
            new_private_native_steps=native_steps,
            reserved_private_native_steps=attempted_steps,
            new_future_nn_calls=nn_calls,
            actual_native_steps=0,
            neural_optimizer_updates=0,
            parent_actual_prefixes_exact=True,
            donors_unmodified=True,
            full_h300_and_actual_response_controller_not_qualified=True,
            hardware_authorized=False,
            promotion_authorized=False,
            commercial_publicity_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
