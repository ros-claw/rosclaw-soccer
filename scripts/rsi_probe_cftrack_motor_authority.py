"""One preregistered nine-arm native closed-loop motor authority pilot."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motor-qualification", type=Path, required=True)
    p.add_argument("--motor-qualification-hash", required=True)
    p.add_argument("--parent-baseline", type=Path, required=True)
    p.add_argument("--external-source", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.cftrack_motor_authority import MOTOR_AXES, summarize_motor_future
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_cftrack_native_context import build_cold_cftrack
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete = a.motor_qualification / "complete.json"
    if a.output.exists() or complete.is_symlink() or digest(complete) != a.motor_qualification_hash:
        raise ValueError("unused pilot output and exact motor interface qualification required")
    qualified = json.loads(complete.read_text())
    if (
        qualified["status"] != "CFTRACK_MOTOR_OWN_NN_NATIVE_FUTURE_EXACT"
        or not qualified["matched_owned_future_actual"]
        or qualified["actual_controls"] != 160
    ):
        raise ValueError("full actual/private nonzero motor qualification required")
    pins = dict(qualified["input_file_hashes"])
    pins[str(complete)] = a.motor_qualification_hash
    parent_complete = a.parent_baseline / "complete.json"
    if parent_complete.is_symlink() or pins.get(str(parent_complete)) != digest(parent_complete):
        raise ValueError("source-pinned original direct parent required")
    parent_record = json.loads(parent_complete.read_text())
    external = a.external_source.resolve()
    if str(external / "policy/omnicontact/OmniContact.py") not in pins or any(
        G1_DDS_JOINT_NAMES[index] != name for name, index in MOTOR_AXES
    ):
        raise ValueError("qualified external policy and named HG axes required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("one qualified native body required")
    a.output.mkdir()
    arms = [
        dict(name=name, index=index, offset_rad=amount)
        for name, index in MOTOR_AXES
        for amount in [-0.15, 0.15]
    ]
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            arms=arms,
            parent_controls=160,
            arm_controls=160,
            maximum_private_native_steps=14400,
            maximum_private_nn_calls=1440,
            actual_native_steps=0,
            neural_optimizer_updates=0,
            motor_offset_limit_rad=0.25,
            additional_motor_velocity_limit_rad_s=4.0,
            torque_cap_fraction=0.85,
            control_dt_sec=0.02,
            window_samples=49,
            relative_peak=24,
            absolute_peak="own_parent_first_foot_control_minus_one",
            private_post60_is_not_h300_exam=True,
            commercial_publicity_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import numpy as np

    from rosclaw_soccer.rsi.cftrack_motor_residual_plan import CFtrackMotorResidualPlan
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
    from rosclaw_soccer.rsi.shared_cftrack_forecast import SharedCFtrackForecaster
    from rosclaw_soccer.sim.contracts import hash_json

    player, body_hash, receipt = build_cold_cftrack(
        scenes[0].read_text(), external, direct_approach=True
    )
    if hash_json(receipt) != hash_json(parent_record["reference_conditioning"]):
        raise ValueError("original direct reference does not regenerate")
    before = cftrack_state_hash(player.policy)
    forecaster = SharedCFtrackForecaster(
        player,
        expected_policy_type=type(player.policy),
        expected_model_hash=body_hash,
        native_step_budget=14400,
    )
    records = []

    def save(name, future, plan):
        path = a.output / (name + ".npz")
        np.savez_compressed(path, **{k: v for k, v in future.items() if isinstance(v, np.ndarray)})
        pins[str(path)] = digest(path)
        row = dict(
            name=name,
            motor_plan=asdict(plan) if plan else None,
            artifact=dict(path=path.name, hash=pins[str(path)]),
            future={k: v for k, v in future.items() if not isinstance(v, np.ndarray)},
            metrics=summarize_motor_future(
                future, ball_address=player.layout.bq, root_address=player.layout.rq
            ),
        )
        write_once(a.output / (name + ".json"), row)
        records.append(row)
        return row

    parent = forecaster.forecast(controls=160)
    parent_row = save("parent", parent, None)
    first = parent_row["metrics"]["first_foot_control"]
    if (
        parent["runtime_failure"] is None
        and not parent["body_failed"]
        and parent["controls"] == 160
        and first is not None
        and 25 <= first <= 99
    ):
        for number, arm in enumerate(arms):
            delta = [0.0] * 29
            delta[arm["index"]] = float(arm["offset_rad"])
            plan = CFtrackMotorResidualPlan(
                player.policy.counter_step, first - 25, 24, tuple(delta), before
            )
            save(f"arm-{number:02d}", forecaster.forecast(controls=160, motor_plan=plan), plan)
    original = a.parent_baseline / parent_record["artifact"]["path"]
    if original.is_symlink() or pins.get(str(original)) != digest(original):
        raise ValueError("original raw parent required for post-generation provenance")
    with np.load(original, allow_pickle=False) as old:
        if any(
            not np.array_equal(parent[k], old[k][: len(parent[k])])
            for k in ["qpos", "qvel", "torque", "ball_forces", "joint_targets"]
        ):
            raise ValueError("own generated parent differs from qualified actual prefix")
    if (
        cftrack_state_hash(player.policy) != before
        or player.world.native_steps != 0
        or player.motor_plan is not None
    ):
        raise ValueError("private teacher trial modified its donor")
    feasible = [r for r in records[1:] if r["metrics"]["local_post60_conditions_met"]]
    best = (
        min(
            feasible,
            key=lambda r: (
                r["metrics"]["lateral_forward_ratio"],
                -r["metrics"]["forward_60_m"],
                r["name"],
            ),
        )
        if feasible
        else None
    )
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_MOTOR_AUTHORITY_LOCAL_RESCUE"
            if best
            else "CFTRACK_MOTOR_AUTHORITY_NO_LOCAL_RESCUE",
            source_head=a.expected_head,
            input_file_hashes=pins,
            records=records,
            selected_local_candidate=best["name"] if best else None,
            future_counters=forecaster.counters(),
            actual_native_steps=0,
            neural_optimizer_updates=0,
            donor_unchanged=True,
            private_post60_is_not_full_h300_exam=True,
            commercial_publicity_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
