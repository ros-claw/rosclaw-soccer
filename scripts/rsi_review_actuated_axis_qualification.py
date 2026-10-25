"""Read actual G1 rank-one target/codec and five saved NN futures; no stepping."""

import argparse
import json
from importlib import import_module
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--qualification-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    cp = a.qualification / "complete.json"
    if a.output.exists() or cp.is_symlink() or digest(cp) != a.qualification_hash:
        raise ValueError("unused readback and exact closed task-axis qualification required")
    report = json.loads(cp.read_text())
    if (
        report["status"] != "G1_ACTUATED_TASK_DIRECTION_NN_FUTURES_EXACT"
        or report["actuated_task_axis"] != 3
        or report["actuated_task_probe_rad"] != 0.03
        or report["fitted_pca_basis_used"] is not False
        or report["actual_worlds"] != 1
        or report["actual_native_steps"] != 3600
        or report["before_proposal"] is not True
        or report["executed_target_conditioning"] is not True
        or report["all_future_nn_targets_and_native_arrays_exact"] is not True
        or report["future_target_semantics"] != "original_NN_plus_executed_target_plan"
        or report["reference_conditioning"] != "OFF_original_arrays"
        or report["executed_target_projection_mode"] != "per_axis"
        or report["reference_arrays_unchanged_by_target_plan"] is not True
        or report["actual_whole_trajectory_unchanged"] is not False
        or any(
            report[k] is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
                "teacher_effectiveness_proven",
            )
        )
        or report["optimizer_updates"] != 0
        or report["forecasts"]
        != dict(
            attempted_native_steps=6400,
            forecast_calls=5,
            future_nn_calls=640,
            integrated_native_steps=6400,
            maximum_native_steps=6400,
        )
    ):
        raise ValueError("same preregistered knee-axis semantic qualification required")
    pins = dict(report["input_file_hashes"])
    pins[str(cp)] = a.qualification_hash
    dp = a.qualification / "declaration.json"
    pins[str(dp)] = digest(dp)
    declaration = json.loads(dp.read_text())
    if any(
        declaration[k] != report[k]
        for k in (
            "source_head",
            "actuated_task_axis",
            "actuated_task_probe_rad",
            "executed_target_projection_mode",
            "reference_conditioning",
        )
    ):
        raise ValueError("saved declaration and closed producer identity differ")
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        paths = [Path(k) for k in pins if k.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError("unique pinned original task-axis asset required: " + suffix)
        return paths[0]

    import mujoco
    import numpy as np
    import yaml
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.actuated_task_response import actuated_task_plan
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_review_joint_response_pilot import verify_target_edits

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(
            one("/cached-standard/declared-scene.xml").read_text(), prefixes=("a/",)
        )
    )
    model_hash = compiled_model_hash(model)
    if model_hash != declaration["model_hash"]:
        raise ValueError("compiled body differs from the source qualification")
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    types = _load_robonaldo(one("/policy/robonaldo/FreeKick.py").parents[2])
    cfg = yaml.safe_load(one("/policy/robonaldo/config/freekick.yaml").read_text())
    module = import_module(types[2].__module__)
    low, high = module._compile_action_clip_bounds(cfg, cfg.get("clip_actions", 3.0))
    codec = (
        np.array(cfg["default_joint_pos"], np.float32),
        np.array(cfg["action_scale"], np.float32),
        np.array(types[3]),
        low,
        high,
    )

    def arrays(path):
        if pins.get(str(path)) != digest(path):
            raise ValueError("saved actual or future trajectory changed")
        with np.load(path, allow_pickle=False) as raw:
            values = {k: raw[k].copy() for k in raw.files}
        if any(not np.isfinite(v).all() for v in values.values()):
            raise ValueError("nonfinite saved native trajectory")
        return values

    actual = arrays(a.qualification / "actual.npz")
    row = report["actual_record"]
    outcome = relabel_arrays(actual, controls=360, ball_address=layout.bq, root_address=layout.rq)
    root = actual["qpos"][:, layout.rq : layout.rq + 7]
    if (
        hash_json(outcome) != hash_json(row["outcome"])
        or row["controls"] != 360
        or row["native_steps"] != 3600
        or row["healthy_full_horizon"] is not True
        or (root[:, 2] < 0.65).any()
        or (1 - 2 * (root[:, 4] ** 2 + root[:, 5] ** 2) < 0.9).any()
        or (
            abs(actual["torque"][:, layout.ai]) > 0.85 * np.array(G1_HARD_TORQUE_LIMITS) + 1e-12
        ).any()
    ):
        raise ValueError("actual native body, torque or original task label differs")
    entry = row["entry_control"]
    probes = [entry, 80, 100, 150, 220]
    if declaration["probes"] != probes or len(set(probes)) != 5:
        raise ValueError("same complete preregistered future boundaries required")
    delta = np.zeros(29, np.float64)
    delta[3] = 0.03
    plan = actuated_task_plan(entry, delta)
    verify_target_edits(
        report["executed_target_edits"],
        actual["joint_targets"][entry:],
        start=entry,
        plan=plan,
        codec=codec,
        limits=model.jnt_range[layout.joints],
    )
    if sum(bool(e["applied"]) for e in report["executed_target_edits"]) != 23:
        raise ValueError("nonzero first-lobe task direction did not actually execute")
    ticks = {}
    for start in probes:
        predicted = arrays(a.qualification / f"forecast-{start:03d}.npz")
        for key in ("qpos", "qvel", "torque", "ball_forces"):
            if not np.array_equal(predicted[key], actual[key][start * 10 : (start + 128) * 10]):
                raise ValueError("saved raw NN future differs from actual execution: " + key)
        if (
            not np.array_equal(
                predicted["joint_targets"], actual["joint_targets"][start : start + 128]
            )
            or predicted["policy_ticks"].shape != (128,)
            or not (np.diff(predicted["policy_ticks"]) == 1).all()
        ):
            raise ValueError("predicted target or continuous NN clock differs")
        for frame, tick in enumerate(predicted["policy_ticks"], start):
            if frame in ticks and ticks[frame] != int(tick):
                raise ValueError("overlapping copied futures disagree about actual policy clock")
            ticks[frame] = int(tick)
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="G1_TASK_AXIS_TARGET_RAW_CODEC_AND_FUTURES_VERIFIED",
            source_head=a.expected_head,
            qualification_hash=a.qualification_hash,
            input_file_hashes=pins,
            compiled_model_hash=model_hash,
            actual_native_steps_readback=3600,
            future_native_steps_readback=6400,
            actual_target_edits_verified=360 - entry,
            nonzero_applied_controls=23,
            original_reference_immutability_is_pinned_producer_assertion=True,
            copied_policy_math_not_independently_reimplemented=True,
            new_physical_steps=0,
            new_neural_calls=0,
            neural_optimizer_updates=0,
            physical_teacher_qualified=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
