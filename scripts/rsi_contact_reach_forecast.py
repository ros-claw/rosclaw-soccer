"""Preregistered SIM-only paired physics forecast from a real measured boundary.

The foot correction is a privileged geometric teacher proposal, not a learned
policy, real execution, or contact guarantee. Compare both futures at the same
absolute H300 deadline, with the unchanged first-touch contract.
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
from rosclaw_soccer.rsi.cftrack_live_history import CFtrackLiveHistory
from rosclaw_soccer.rsi.cftrack_measured_bridge_reference import condition_measured_bridge_reference
from rosclaw_soccer.rsi.cftrack_measured_history_install import install_owned_cftrack_history
from rosclaw_soccer.rsi.cftrack_motor_residual_plan import CFtrackMotorResidualPlan
from rosclaw_soccer.rsi.contact_reach_teacher import bounded_contact_reach
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
from rosclaw_soccer.rsi.shared_cftrack_forecast import SharedCFtrackForecaster
from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters, SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean

ROOT = Path("/code/rosclaw/soccer_native_evidence")
SOURCE_HASH = "sha256:75612fbae5c1e4df0741c16b4a5000473aac829f919adbb5367027dce75cb151"
AUDIT_HASH = "sha256:db3b3f1b8d6553ba7df56878bbc292517668c79cf7a454984ea7440048be0c3f"
MODEL_HASH = "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-right-ankle-roll-after-rejection", action="store_true")
    parser.add_argument("--exclude-both-ankles-after-rejection", action="store_true")
    parser.add_argument("--calibrate-hip-knee-response", action="store_true")
    parser.add_argument("--replay-contact-arm", choices=("hg7-plus", "hg9-minus"))
    args = parser.parse_args()
    clean(args.expected_head)
    source_dir = ROOT / "rsi-cftrack-predicted-ball-h300-20261027"
    source_path = source_dir / "complete.json"
    audit_path = ROOT / "rsi-cftrack-predicted-ball-review-20261027/complete.json"
    if (
        args.output.exists()
        or digest(source_path) != SOURCE_HASH
        or digest(audit_path) != AUDIT_HASH
    ):
        raise ValueError("unused output and exact completed source and independent review required")
    rejection = ROOT / "rsi-contact-reach-forecast-20261027/complete.json"
    if args.exclude_right_ankle_roll_after_rejection:
        if (
            digest(rejection)
            != "sha256:e74da7e3dbad9535e3698d66fab43d482c5ce6bd0107dac099197433496d567a"
        ):
            raise ValueError("exact earlier physical-limit rejection required")
        rejected = json.loads(rejection.read_text())["records"]["candidate"]["runtime_failure"]
        if rejected["type"] != "MotorTargetRejected" or rejected["target_rejection"][
            "violating_hg_indices"
        ] != [11]:
            raise ValueError("right ankle roll rejection required")
    if args.exclude_both_ankles_after_rejection:
        if args.exclude_right_ankle_roll_after_rejection:
            raise ValueError("select exactly one rejection-informed experiment")
        constrained = ROOT / "rsi-contact-reach-constrained-20261027/complete.json"
        if (
            digest(constrained)
            != "sha256:6abf4ea2a3f2b1bb7859828f5e90583503d4b2a69990b2cca7a9e95be25e2fee"
        ):
            raise ValueError("exact earlier second physical-limit rejection required")
        rejected = json.loads(constrained.read_text())["records"]["candidate"]["runtime_failure"]
        if rejected["type"] != "MotorTargetRejected" or rejected["target_rejection"][
            "violating_hg_indices"
        ] != [10]:
            raise ValueError("right ankle pitch rejection required")
    if args.calibrate_hip_knee_response:
        if (
            args.exclude_both_ankles_after_rejection
            or args.exclude_right_ankle_roll_after_rejection
        ):
            raise ValueError("calibration is a separate preregistered experiment")
        prior_hip = ROOT / "rsi-contact-reach-hip-knee-20261027/complete.json"
        if (
            digest(prior_hip)
            != "sha256:eea7cbbe6f37e9d210c0ee42b006637068bf3f2cfcf240615eebf7f566d6c518"
        ):
            raise ValueError("exact completed hip-knee failure required")
        failed = json.loads(prior_hip.read_text())["records"]["candidate"]
        if not failed["complete"] or failed["outcome"]["task_pass"]:
            raise ValueError("completed but unsuccessful hip-knee candidate required")
    if args.replay_contact_arm:
        if (
            args.calibrate_hip_knee_response
            or args.exclude_both_ankles_after_rejection
            or args.exclude_right_ankle_roll_after_rejection
        ):
            raise ValueError("actual replay is a separate experiment")
        calibration_dir = ROOT / "rsi-hip-knee-response-20261027"
        if (
            digest(calibration_dir / "complete.json")
            != "sha256:765c411cfde3d7f5716dc57d30df5cf8a30b44695db7b094aba38047ddbeb29a"
        ):
            raise ValueError("exact completed calibration required")
        arm_hashes = {
            "hg7-plus": "sha256:54ea1ceb0d6d28f9096c3272e9c398fd41303c2d738ad7320e9bc7285ba8960f",
            "hg9-minus": "sha256:10c18e9b6a9988626587894ff0ed454437055c32445f84dfeaffc20b6c6bcfb9",
        }
        arm_path = calibration_dir / f"{args.replay_contact_arm}.npz"
        if arm_path.is_symlink() or digest(arm_path) != arm_hashes[args.replay_contact_arm]:
            raise ValueError("exact independently captured private arm artifact required")
        with np.load(arm_path, allow_pickle=False) as f:
            replay_source = {key: f[key].copy() for key in f.files}
    source = json.loads(source_path.read_text())
    if source["status"] != "CFTRACK_PREDICTIVE_BALL_H300_PAIRED_NOT_PROMOTED":
        raise ValueError("expected failed predictive source required")
    pins = source["input_file_hashes"]

    def find(suffix):
        matches = [Path(k) for k in pins if k.endswith(suffix)]
        if (
            len(matches) != 1
            or matches[0].is_symlink()
            or digest(matches[0]) != pins[str(matches[0])]
        ):
            raise ValueError(f"exact uniquely pinned input required: {suffix}")
        return matches[0]

    scene = find("/cached-standard/declared-scene.xml")
    actor = find("/velocity/v0/exported/policy.onnx")
    config = find("/velocity/v0/params/deploy.yaml")
    external = find("/policy/omnicontact/OmniContact.py").parents[2]
    kick = find("/policy/omnicontact/model/kick_50k.onnx")
    origin_report = find("/rsi-warm-left-settle-v3-20261027/complete.json")
    history_report = find("/rsi-cftrack-measured-history-20261027/complete.json")
    bridge_report = find("/rsi-terminal-anchor-bridge-20261027/complete.json")

    def load_artifact(report, item):
        path = report.parent / item["path"]
        if path.is_symlink() or path.parent != report.parent or digest(path) != item["hash"]:
            raise ValueError("confined authenticated artifact required")
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}

    old = load_artifact(origin_report, json.loads(origin_report.read_text())["artifact"])
    history = load_artifact(history_report, json.loads(history_report.read_text())["artifact"])
    bridge = load_artifact(
        bridge_report, json.loads(bridge_report.read_text())["records"][0]["artifact"]
    )
    prior = load_artifact(source_path, source["records"][1]["actual_execution"]["artifact"])
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            source=SOURCE_HASH,
            independent_audit=AUDIT_HASH,
            source_head=args.expected_head,
            experiment=(
                "actual deterministic replay"
                if args.replay_contact_arm
                else "paired dynamics calibration"
                if args.calibrate_hip_knee_response
                else "one damped Jacobian candidate against same-state parent"
            ),
            actual_replay_arm=args.replay_contact_arm,
            calibration_artifact_hash=(
                arm_hashes[args.replay_contact_arm] if args.replay_contact_arm else None
            ),
            actual_prefix_controls=140,
            candidate_count=0
            if args.replay_contact_arm
            else 8
            if args.calibrate_hip_knee_response
            else 1,
            excluded_hg_indices=(
                [10, 11]
                if args.exclude_both_ankles_after_rejection
                else [11]
                if args.exclude_right_ankle_roll_after_rejection
                else []
            ),
            forecast_controls_per_arm=160,
            native_step_budget=3000
            if args.replay_contact_arm
            else 15800
            if args.calibrate_hip_knee_response
            else 4600,
            calibration_axes_hg=[6, 7, 8, 9] if args.calibrate_hip_knee_response else [],
            calibration_delta_rad=0.08 if args.calibrate_hip_knee_response else None,
            damping=0.02,
            joint_bound_rad=0.18,
            phase14_witness_cftrack_control=95,
            proposal_start_cftrack_control=83,
            absolute_scoring_horizon=300,
            privileged_future_teacher=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    sys.path.insert(0, str(external))
    from common.ctrlcomp import PolicyOutput, StateAndCmd
    from policy.omnicontact.OmniContact import OmniContact

    model = mujoco.MjModel.from_xml_string(shared_native_scene(scene.read_text(), prefixes=("a/",)))
    if compiled_model_hash(model) != MODEL_HASH:
        raise ValueError("compiled body mismatch")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    stream = ContinuousVelocityStream(
        actor,
        config,
        expected_model_hash=pins[str(actor)],
        expected_config_hash=pins[str(config)],
        horizon_controls=301,
    )
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = old["initial_qpos"], old["initial_qvel"]
    mujoco.mj_forward(model, data)
    effort = 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
    world = SharedNativeMotorWorld(
        model,
        data,
        players={
            "a/": NativePDParameters(
                np.array(stream.params.stiffness), np.array(stream.params.damping), effort
            )
        },
        ball_body_name="soccer_ball",
    )
    collector = CFtrackLiveHistory(world, "a/")
    raw = {key: [] for key in ("qpos", "qvel", "torque", "ball_forces")}
    for frame in range(57):
        q, v = lo.canonical(data)
        command = np.array([0, 0.25 if 25 <= frame < 57 else 0, 0], np.float32)
        target = stream.propose(q, v, lo.angular_velocity_body(data), command, frame=frame)
        row = world.advance({"a/": target})
        if not np.array_equal(target, old["joint_targets"][frame]) or any(
            not np.array_equal(row[k], old[k][frame * 10 : (frame + 1) * 10]) for k in raw
        ):
            raise ValueError(f"actual velocity replay mismatch {frame}")
        for k in raw:
            raw[k].append(row[k])
        if frame >= 52:
            collector.capture()
    policy = OmniContact(StateAndCmd(29), PolicyOutput(29), onnx_path=str(kick))
    policy.task, policy.reference_source, policy.active_object_name = "kickball", "CFgen", "ball"
    policy.box_dims = policy.ball_dims = np.full(3, 0.11, np.float32)
    policy.bbox_scale = policy.box_dims * 2
    policy.bbox_offsets_scaled = policy.bbox_offsets * policy.bbox_scale.reshape(1, 3)
    world.switch_pd_profile(
        "a/",
        NativePDParameters(policy.kps_lab[policy.lab2mj], policy.kds_lab[policy.lab2mj], effort),
    )
    player = SharedOmniContactPlayer(
        world, "a/", policy, expected_policy_type=OmniContact, goal=np.array([5.0, 0.0, 0.11])
    )
    install_owned_cftrack_history(player, collector)
    if not np.array_equal(policy.obs_history_buffer[1:], history["entry57_observations"][:4]):
        raise ValueError("live measured history mismatch")
    receipt = condition_measured_bridge_reference(
        player,
        bridge["candidate_root"],
        bridge["candidate_joint"],
        expected_model_hash=MODEL_HASH,
        ball_prediction_horizon_s=1.74,
    )
    if (
        receipt["conditioned_reference_hash"]
        != source["records"][1]["bridge_reference_receipt"]["conditioned_reference_hash"]
    ):
        raise ValueError("predictive reference mismatch")
    for frame in range(83):
        target, _ = player.propose()
        row = world.advance({"a/": target})
        if not np.array_equal(target, prior["joint_targets"][frame]) or any(
            not np.array_equal(row[k], prior[k][frame * 10 : (frame + 1) * 10]) for k in raw
        ):
            raise ValueError(f"actual CFtrack replay mismatch {frame}")
        for k in raw:
            raw[k].append(row[k])
        player.commit()
        if world.failed:
            raise ValueError("measured donor failed")
    if world.control_steps != 140 or policy.counter_step != 83:
        raise ValueError("exact causal donor clock required")
    if args.replay_contact_arm:
        axis, amount = (7, 0.08) if args.replay_contact_arm == "hg7-plus" else (9, -0.08)
        residual = np.zeros(29)
        residual[axis] = amount
        player.install_motor_plan(
            CFtrackMotorResidualPlan(
                origin_counter=83,
                start_control=83,
                peak_control=12,
                delta_hg=tuple(float(x) for x in residual),
                donor_policy_hash=cftrack_state_hash(policy),
            )
        )
        phases = policy.ref_phase.copy()
        actual_targets = []
        original_session = policy.ort_session
        obs, actions = [], []

        class Capture:
            def run(self, outputs, inputs):
                result = original_session.run(outputs, inputs)
                obs.append(inputs["obs"].copy())
                actions.append(result[0].copy())
                return result

        policy.ort_session = Capture()
        for step in range(160):
            target, _ = player.propose()
            row = world.advance({"a/": target})
            if not np.array_equal(target, replay_source["targets"][step]) or any(
                not np.array_equal(row[k], replay_source[k][1400 + step * 10 : 1410 + step * 10])
                for k in raw
            ):
                raise ValueError(
                    f"independent actual replay differs from private forecast at {step}"
                )
            for k in raw:
                raw[k].append(row[k])
            actual_targets.append(target.copy())
            player.commit()
            if world.failed:
                raise ValueError("actual replay body failed")
        arrays = {k: np.concatenate(raw[k]) for k in raw}
        positions = arrays["qpos"].reshape(300, 10, model.nq)
        forces = arrays["ball_forces"].reshape(300, 10, 1, 3).max(axis=1)
        force6 = np.zeros((300, 1, 6), dtype=np.float64)
        force6[:, :, :3] = forces
        outcome = label_first_touch(
            dict(
                force_n=force6,
                pelvis_z_per_substep_m=positions[:, :, lo.rq + 2][:, None, :],
                ball_position_after_step_m=positions[:, -1, lo.bq : lo.bq + 3][:, None, :],
            )
        ).to_dict()
        first = outcome["first_foot_contact_frame"]
        phase = int(phases[min(first - 57, len(phases) - 1)]) if first is not None else None
        path = args.output / "actual.npz"
        np.savez_compressed(
            path,
            **arrays,
            targets=np.stack(actual_targets),
            neural_obs=np.stack(obs),
            neural_actions=np.stack(actions),
        )
        write_once(
            args.output / "complete.json",
            dict(
                status="ACTUAL_CALIBRATION_CONTACT_REPLAY_NOT_PROMOTED",
                arm=args.replay_contact_arm,
                source=SOURCE_HASH,
                calibration_hash="sha256:765c411cfde3d7f5716dc57d30df5cf8a30b44695db7b094aba38047ddbeb29a",
                calibration_artifact_hash=arm_hashes[args.replay_contact_arm],
                source_head=args.expected_head,
                actual_artifact_hash=digest(path),
                actual_controls=300,
                native_steps=3000,
                neural_calls=160,
                private_future_exact=True,
                first_contact_reference_phase=phase,
                outcome=outcome,
                task_promoted=False,
                hardware_authorized=False,
            ),
        )
        return
    forecaster = SharedCFtrackForecaster(
        player,
        expected_policy_type=OmniContact,
        expected_model_hash=MODEL_HASH,
        native_step_budget=14400 if args.calibrate_hip_knee_response else 3200,
    )
    parent = forecaster.forecast(controls=160)
    if parent["controls"] != 160 or any(
        not np.array_equal(parent[k], prior[k][830:2430]) for k in raw
    ):
        raise ValueError("parent future differs from original real physics")
    scratch = mujoco.MjData(model)
    witness = 12 * 10 + 9
    scratch.qpos[:], scratch.qvel[:] = parent["qpos"][witness], parent["qvel"][witness]
    mujoco.mj_forward(model, scratch)
    ball_geom = int(np.flatnonzero(model.geom_bodyid == lo.ball)[0])
    candidates = []
    for idx in range(1, 8):
        foot = model.geom(f"a/right_foot{idx}_collision").id
        points = np.zeros(6)
        gap = mujoco.mj_geomDistance(model, scratch, ball_geom, foot, 2.0, points)
        candidates.append((gap, foot, points.copy()))
    gap, foot, points = min(candidates, key=lambda x: x[0])
    jac = np.zeros((3, model.nv))
    mujoco.mj_jac(model, scratch, jac, None, points[3:], model.geom_bodyid[foot])
    active = np.arange(
        6,
        10
        if args.exclude_both_ankles_after_rejection
        else 11
        if args.exclude_right_ankle_roll_after_rejection
        else 12,
    )
    proposal = bounded_contact_reach(
        jac[:, lo.vi[active]], points[:3] - points[3:], damping=0.02, max_joint_delta=0.18
    )
    delta = np.zeros(29)
    delta[active] = proposal.delta
    plan = CFtrackMotorResidualPlan(
        origin_counter=83,
        start_control=83,
        peak_control=12,
        delta_hg=tuple(float(x) for x in delta),
        donor_policy_hash=cftrack_state_hash(policy),
    )

    def summarize(future, arm):
        arrays = {k: np.concatenate([*raw[k], future[k]]) for k in raw}
        artifact = args.output / f"{arm}.npz"
        np.savez_compressed(artifact, **arrays, targets=future["joint_targets"])
        if future["controls"] != 160:
            return dict(
                complete=False,
                runtime_failure=future["runtime_failure"],
                native_steps=future["native_steps"],
                body_failed=future["body_failed"],
                artifact_hash=digest(artifact),
            )
        positions = arrays["qpos"].reshape(300, 10, model.nq)
        forces = arrays["ball_forces"].reshape(300, 10, 1, 3).max(axis=1)
        force6 = np.zeros((300, 1, 6), dtype=np.float64)
        force6[:, :, :3] = forces
        outcome = label_first_touch(
            dict(
                force_n=force6,
                pelvis_z_per_substep_m=positions[:, :, lo.rq + 2][:, None, :],
                ball_position_after_step_m=positions[:, -1, lo.bq : lo.bq + 3][:, None, :],
            )
        ).to_dict()
        return dict(
            complete=True,
            runtime_failure=future["runtime_failure"],
            body_failed=future["body_failed"],
            outcome=outcome,
            artifact_hash=digest(artifact),
        )

    if args.calibrate_hip_knee_response:
        records = {"parent": summarize(parent, "parent")}
        for axis in range(6, 10):
            for sign in (-1, 1):
                arm = f"hg{axis}-{'minus' if sign < 0 else 'plus'}"
                perturbation = np.zeros(29)
                perturbation[axis] = sign * 0.08
                calibration_plan = CFtrackMotorResidualPlan(
                    origin_counter=83,
                    start_control=83,
                    peak_control=12,
                    delta_hg=tuple(float(x) for x in perturbation),
                    donor_policy_hash=cftrack_state_hash(policy),
                )
                future = forecaster.forecast(controls=160, motor_plan=calibration_plan)
                summary = summarize(future, arm)
                if future["native_steps"] > witness:
                    scratch.qpos[:], scratch.qvel[:] = (
                        future["qpos"][witness],
                        future["qvel"][witness],
                    )
                    mujoco.mj_forward(model, scratch)
                    current_points = np.zeros(6)
                    current_gap = mujoco.mj_geomDistance(
                        model, scratch, ball_geom, foot, 2.0, current_points
                    )
                    summary["witness_gap_m"] = float(current_gap)
                    summary["witness_relative_xyz_m"] = (
                        current_points[:3] - current_points[3:]
                    ).tolist()
                summary["delta_hg_index"] = axis
                summary["delta_rad"] = sign * 0.08
                records[arm] = summary
                write_once(args.output / f"{arm}.json", summary)
        write_once(
            args.output / "complete.json",
            dict(
                status="CFTRACK_HIP_KNEE_PHYSICS_RESPONSE_CALIBRATION_NOT_PROMOTED",
                source=SOURCE_HASH,
                audit=AUDIT_HASH,
                source_head=args.expected_head,
                witness_cftrack_control=95,
                witness_parent_gap_m=float(gap),
                witness_parent_relative_xyz_m=(points[:3] - points[3:]).tolist(),
                witness_geom_id=int(foot),
                records=records,
                counters=forecaster.counters(),
                actual_native_steps=1400,
                simulation_privileged_teacher=True,
                physical_force_is_contact_authority=True,
                task_promoted=False,
                hardware_authorized=False,
            ),
        )
        return

    candidate = forecaster.forecast(controls=160, motor_plan=plan)
    records = {
        name: summarize(future, name)
        for name, future in [("parent", parent), ("candidate", candidate)]
    }
    write_once(
        args.output / "complete.json",
        dict(
            status="BOUNDED_FOOT_REACH_PRIVATE_PHYSICS_NOT_PROMOTED",
            source=SOURCE_HASH,
            audit=AUDIT_HASH,
            source_head=args.expected_head,
            witness_gap_m=float(gap),
            witness_points=points.tolist(),
            raw_delta=proposal.raw_delta.tolist(),
            delta=proposal.delta.tolist(),
            active_hg_indices=active.tolist(),
            jacobian=jac[:, lo.vi[active]].tolist(),
            predicted_linear_displacement=proposal.predicted_displacement.tolist(),
            records=records,
            counters=forecaster.counters(),
            actual_native_steps=1400,
            simulation_privileged_teacher=True,
            task_promoted=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
