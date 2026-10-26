"""Rebuild actual kick NN observations from saved native states; no integration."""

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-hash", required=True)
    p.add_argument("--measured-bridge", action="store_true")
    p.add_argument("--predicted-bridge", action="store_true")
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete = a.source / "complete.json"
    if (
        a.output.exists()
        or complete.is_symlink()
        or digest(complete) != a.source_hash
        or any(n in sys.modules for n in ("common", "policy", "FSM"))
    ):
        raise ValueError("unused output, exact closed source and isolated imports required")
    source = json.loads(complete.read_text())
    if a.measured_bridge and a.predicted_bridge:
        raise ValueError("one explicit source type required")
    bridge_mode = a.measured_bridge or a.predicted_bridge
    if (
        source["status"]
        != (
            "CFTRACK_PREDICTIVE_BALL_H300_PAIRED_NOT_PROMOTED"
            if a.predicted_bridge
            else "CFTRACK_MEASURED_BRIDGE_H300_PAIRED_NOT_PROMOTED"
            if a.measured_bridge
            else "CFTRACK_PRIVATE_FUTURES_REPLAYED_ACTUALLY_NOT_PROMOTED"
        )
        or source["native_steps"] != (7140 if bridge_mode else 6780)
        or source["cftrack_neural_calls"] != (600 if bridge_mode else 400)
        or [(r["entry_control"], r["actual_execution"]["arm"]) for r in source["records"]]
        != (
            [(57, "static_bridge"), (57, "predictive_bridge")]
            if a.predicted_bridge
            else [(57, "parent"), (57, "measured_bridge")]
            if a.measured_bridge
            else [
                (57, "measured_history"),
                (57, "zero_history"),
                (82, "measured_history"),
                (82, "zero_history"),
            ]
        )
    ):
        raise ValueError("complete four actual execution traces required")
    pins = dict(source["input_file_hashes"])
    pins[str(complete)] = a.source_hash
    for row in source["records"]:
        for item in (row["artifact"], row["actual_execution"]["artifact"]):
            path = a.source / item["path"]
            if path.is_symlink() or path.parent != a.source or digest(path) != item["hash"]:
                raise ValueError("exact confined actual trace required")
            pins[str(path)] = item["hash"]
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    histories = [
        Path(x) for x in pins if x.endswith("/rsi-cftrack-measured-history-20261027/history.npz")
    ]
    forecast_reports = [
        Path(x)
        for x in pins
        if x.endswith("/rsi-cftrack-history-private-future-20261027/complete.json")
    ]
    external_files = [Path(x) for x in pins if x.endswith("/policy/omnicontact/OmniContact.py")]
    bridge_artifacts = [
        Path(x) for x in pins if x.endswith("/rsi-terminal-anchor-bridge-20261027/entry57.npz")
    ]
    if not (len(scenes) == len(histories) == len(forecast_reports) == len(external_files) == 1):
        raise ValueError("unique scene, measured history, forecast and policy required")
    if bridge_mode and len(bridge_artifacts) != 1:
        raise ValueError("unique certified measured bridge required")
    static_reports = [
        Path(x)
        for x in pins
        if x.endswith("/rsi-cftrack-measured-bridge-h300-20261027/complete.json")
    ]
    if a.predicted_bridge and len(static_reports) != 1:
        raise ValueError("one exact static bridge baseline report required")
    forecast = json.loads(forecast_reports[0].read_text())
    if forecast["status"] != "CFTRACK_MEASURED_HISTORY_PRIVATE_FUTURES_RECORDED_NOT_PROMOTED":
        raise ValueError("qualified corresponding private futures required")
    static_source = json.loads(static_reports[0].read_text()) if a.predicted_bridge else None
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.cftrack_measured_bridge_reference import (
        condition_measured_bridge_reference,
    )
    from rosclaw_soccer.rsi.cftrack_trace_readback import verify_cftrack_trace
    from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    if (
        compiled_model_hash(model)
        != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("exact canonical SIM body required")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    with np.load(histories[0], allow_pickle=False) as f:
        saved_history = {k: f[k].copy() for k in f.files}
    external = external_files[0].parents[2]
    sys.path.insert(0, str(external))
    from common.ctrlcomp import PolicyOutput, StateAndCmd
    from policy.omnicontact.OmniContact import OmniContact

    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            source_hash=a.source_hash,
            input_file_hashes=pins,
            maximum_nn_calls=600 if bridge_mode else 400,
            native_steps=0,
            original_fk_and_neural_codec_shared=True,
            initial_ball_cache_from_prior_native_substep=True,
            scratch_only=True,
            no_new_scenarios=True,
            hardware_authorized=False,
        ),
    )
    rows, calls = [], 0
    for row in source["records"]:
        entry, arm = row["entry_control"], row["actual_execution"]["arm"]
        with np.load(a.source / row["artifact"]["path"], allow_pickle=False) as f:
            prefix = {k: f[k].copy() for k in f.files}
        with np.load(
            a.source / row["actual_execution"]["artifact"]["path"], allow_pickle=False
        ) as f:
            actual = {k: f[k].copy() for k in f.files}
        if prefix["qpos"].shape != (entry * 10, model.nq):
            raise ValueError("complete source physical prefix required")
        forecast_arm = "measured_history" if bridge_mode else arm
        matches = [
            f
            for r in forecast["records"]
            if r["entry_control"] == entry
            for f in r["private_forecasts"]
            if f["arm"] == forecast_arm
        ]
        if len(matches) != 1:
            raise ValueError("unique corresponding private future required")
        forecast_path = forecast_reports[0].parent / matches[0]["artifact"]["path"]
        if (
            forecast_path.is_symlink()
            or forecast_path.parent != forecast_reports[0].parent
            or digest(forecast_path) != matches[0]["artifact"]["hash"]
        ):
            raise ValueError("exact corresponding private future artifact required")
        if a.predicted_bridge and arm == "static_bridge":
            static_records = [
                r
                for r in static_source["records"]
                if r["actual_execution"]["arm"] == "measured_bridge"
            ]
            if len(static_records) != 1:
                raise ValueError("one prior actual static bridge required")
            item = static_records[0]["actual_execution"]["artifact"]
            static_path = static_reports[0].parent / item["path"]
            if (
                static_path.is_symlink()
                or static_path.parent != static_reports[0].parent
                or digest(static_path) != item["hash"]
            ):
                raise ValueError("exact actual static baseline required")
            with np.load(static_path, allow_pickle=False) as f:
                for key in (
                    "qpos",
                    "qvel",
                    "torque",
                    "ball_forces",
                    "joint_targets",
                    "neural_obs",
                    "neural_actions",
                ):
                    if not np.array_equal(f[key], actual[key]):
                        raise ValueError(f"static baseline array mismatch {key}")
        if not bridge_mode or arm == "parent":
            with np.load(forecast_path, allow_pickle=False) as f:
                for key in ("qpos", "qvel", "torque", "ball_forces", "joint_targets"):
                    length = 100 if key == "joint_targets" else 1000
                    if not np.array_equal(f[key], actual[key][:length]):
                        raise ValueError(f"actual/private array mismatch {entry}/{arm}/{key}")
        data = mujoco.MjData(model)
        data.qpos[:] = prefix["qpos"][-2]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        data.qpos[:], data.qvel[:] = prefix["qpos"][-1], prefix["qvel"][-1]
        policy = OmniContact(
            StateAndCmd(29),
            PolicyOutput(29),
            onnx_path=str(external / "policy/omnicontact/model/kick_50k.onnx"),
        )
        if tuple(policy.kinematics.joint_names) != tuple(G1_DDS_JOINT_NAMES):
            raise ValueError("canonical external FK order required")
        policy.task, policy.reference_source, policy.active_object_name = (
            "kickball",
            "CFgen",
            "ball",
        )
        policy.box_dims = policy.ball_dims = np.full(3, 0.11, np.float32)
        policy.bbox_scale = policy.box_dims * 2
        policy.bbox_offsets_scaled = policy.bbox_offsets * policy.bbox_scale.reshape(1, 3)
        order = policy.lab2mj
        pd = NativePDParameters(
            policy.kps_lab[order], policy.kds_lab[order], 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
        )
        world = SharedNativeMotorWorld(
            model, data, players={"a/": pd}, ball_body_name="soccer_ball"
        )
        player = SharedOmniContactPlayer(
            world, "a/", policy, expected_policy_type=OmniContact, goal=np.array([5.0, 0.0, 0.11])
        )
        if arm == "measured_history" or bridge_mode:
            h = saved_history[f"entry{entry}_observations"]
            policy.obs_history_buffer[1:] = h[:4]
            policy.action = h[4, 79:108].copy()
            player.measured_history_installed = True
        elif arm != "zero_history":
            raise ValueError("declared actual history arm required")
        bridge_receipt = None
        if (a.measured_bridge and arm == "measured_bridge") or a.predicted_bridge:
            with np.load(bridge_artifacts[0], allow_pickle=False) as f:
                bridge_receipt = condition_measured_bridge_reference(
                    player,
                    f["candidate_root"].copy(),
                    f["candidate_joint"].copy(),
                    expected_model_hash=compiled_model_hash(model),
                    ball_prediction_horizon_s=1.74 if arm == "predictive_bridge" else None,
                )
            if bridge_receipt != row["bridge_reference_receipt"]:
                raise ValueError("independently rebuilt measured reference differs")
        horizon = 300 if bridge_mode else 100
        original_phase = policy.ref_phase.copy()
        original_object_reference = policy.ref_object_pos.copy()
        count = verify_cftrack_trace(
            policy, data, lo, actual, controls=horizon, initial_cached_qpos=prefix["qpos"][-2]
        )
        if count != horizon or data.time != 0 or world.native_steps != 0 or player.commits != 0:
            raise ValueError("scratch network readback must not integrate or commit")
        calls += count
        evidence = dict(
            entry_control=entry,
            arm=arm,
            neural_calls=count,
            actual_vs_private_arrays_exact=not bridge_mode or arm == "parent",
            initial_cached_ball_from_prior_substep=True,
            measured_history_is_scratch_reconstruction=arm == "measured_history" or bridge_mode,
        )
        if bridge_mode:
            q_all = np.concatenate([prefix["qpos"], actual["qpos"]])[:3000]
            f_all = np.concatenate([prefix["ball_forces"], actual["ball_forces"]])[:3000]
            force = np.zeros((300, 1, 6), np.float64)
            force[:, 0, :3] = f_all[:, 0].reshape(300, 10, 3).max(axis=1)
            q300 = q_all.reshape(300, 10, model.nq)
            outcome = label_first_touch(
                dict(
                    force_n=force,
                    pelvis_z_per_substep_m=q300[:, :, lo.rq + 2][:, None],
                    ball_position_after_step_m=q300[:, -1, lo.bq : lo.bq + 3][:, None],
                )
            ).to_dict()
            if outcome != row["actual_execution"]["absolute_h300_outcome"]:
                raise ValueError("absolute H300 outcome mismatch")
            first = outcome["first_foot_contact_frame"]
            local = None if first is None else first - entry
            phase = (
                None
                if local is None or local < 0
                else int(original_phase[min(local, len(original_phase) - 1)])
            )
            evidence.update(
                absolute_h300_outcome=outcome,
                first_contact_local_control=local,
                first_contact_reference_phase=phase,
                bridge_reference_receipt=bridge_receipt,
            )
        if a.predicted_bridge:
            ball_geoms = np.flatnonzero(model.geom_bodyid == lo.ball)
            if len(ball_geoms) != 1:
                raise ValueError("one canonical ball collision geometry required")
            foot_geoms = [
                [model.geom(f"a/{side}_foot{i}_collision").id for i in range(1, 8)]
                for side in ("left", "right")
            ]
            scratch = mujoco.MjData(model)
            phases, gaps, object_error = [], [], []
            for step in range(300):
                scratch.qpos[:] = actual["qpos"][step * 10 + 9]
                scratch.qvel[:] = actual["qvel"][step * 10 + 9]
                mujoco.mj_forward(model, scratch)
                phase_index = min(step, len(original_phase) - 1)
                phases.append(int(original_phase[phase_index]))
                gaps.append(
                    [
                        min(
                            float(
                                mujoco.mj_geomDistance(
                                    model, scratch, int(ball_geoms[0]), geom, 2.0, None
                                )
                            )
                            for geom in foot
                        )
                        for foot in foot_geoms
                    ]
                )
                object_error.append(
                    scratch.qpos[lo.bq : lo.bq + 3] - original_object_reference[phase_index]
                )
            phases, gaps, object_error = (
                np.asarray(phases),
                np.asarray(gaps),
                np.stack(object_error),
            )
            diagnostic_path = a.output / f"entry{entry}-{arm}-contact-geometry.npz"
            np.savez_compressed(
                diagnostic_path, phase=phases, foot_gap_m=gaps, ball_minus_reference_m=object_error
            )
            phase14 = phases == 14
            evidence.update(
                frame_end_minimum_ball_foot_gap_m=float(np.min(gaps)),
                frame_end_minimum_gap_phase14_m=(
                    float(np.min(gaps[phase14])) if phase14.any() else None
                ),
                phase14_sampled_controls=int(np.count_nonzero(phase14)),
                phase14_max_ball_reference_error_m=(
                    float(np.max(np.linalg.norm(object_error[phase14], axis=1)))
                    if phase14.any()
                    else None
                ),
                frame_end_geometry_not_substep_contact_proof=True,
                contact_geometry_artifact=dict(
                    path=diagnostic_path.name, hash=digest(diagnostic_path)
                ),
            )
        write_once(a.output / f"entry{entry}-{arm}.json", evidence)
        rows.append(evidence)
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status=(
                "ACTUAL_CFTRACK_PREDICTIVE_H300_NN_READBACK_EXACT_NOT_PROMOTED"
                if a.predicted_bridge
                else "ACTUAL_CFTRACK_BRIDGE_H300_NN_READBACK_EXACT_NOT_PROMOTED"
                if a.measured_bridge
                else "ACTUAL_CFTRACK_FUTURES_NN_READBACK_EXACT_NOT_PROMOTED"
            ),
            source_head=a.expected_head,
            source_hash=a.source_hash,
            input_file_hashes=pins,
            records=rows,
            neural_calls=calls,
            native_steps=0,
            shared_source_fk_and_codec=True,
            no_new_scenarios=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
