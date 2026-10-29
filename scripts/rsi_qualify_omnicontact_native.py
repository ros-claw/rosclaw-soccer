"""One research-only CFtrack cold-entry trial in the existing standard native body.

This is a NEW controller/protocol, not a continuation of a RoboNaldo certificate.
External policy code is read from an explicitly frozen, clean SIM-only checkout.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--external-source", type=Path, required=True)
    p.add_argument("--external-head", required=True)
    p.add_argument("--body-source", type=Path, required=True)
    p.add_argument("--body-source-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--direct-approach-reference", action="store_true")
    p.add_argument("--motion-reference", type=Path)
    p.add_argument("--motion-reference-receipt", type=Path)
    p.add_argument("--motion-reference-receipt-hash")
    p.add_argument("--reference-ball-initial-velocity", action="store_true")
    p.add_argument("--reference-all-initial-velocities", action="store_true")
    p.add_argument("--ground-reference-ball", action="store_true")
    p.add_argument("--zero-reference-ball-spin", action="store_true")
    p.add_argument("--factorial-protocol", type=Path)
    p.add_argument("--factorial-protocol-hash")
    p.add_argument("--causal-ball-reference", action="store_true")
    p.add_argument("--causal-protocol", type=Path)
    p.add_argument("--causal-protocol-hash")
    p.add_argument("--projected-target", action="store_true")
    p.add_argument("--projection-protocol", type=Path)
    p.add_argument("--projection-protocol-hash")
    p.add_argument("--reference-phase-start", type=int, default=0)
    p.add_argument("--ankle-roll-reference", action="store_true")
    p.add_argument("--motor-pilot", type=Path)
    p.add_argument("--motor-pilot-hash")
    a = p.parse_args()
    factorial = a.ground_reference_ball or a.zero_reference_ball_spin
    if a.projected_target != (a.projection_protocol is not None) or a.projected_target != (
        a.projection_protocol_hash is not None
    ):
        p.error("target projection requires its exact predeclared protocol")
    if a.projected_target and (
        not a.reference_all_initial_velocities
        or a.reference_phase_start != 0
        or a.motion_reference_receipt is None
        or a.ground_reference_ball != a.zero_reference_ball_spin
        or a.causal_ball_reference
        or a.direct_approach_reference
        or a.ankle_roll_reference
        or a.motor_pilot is not None
    ):
        p.error("projection requires one of the two declared full-velocity NPZ states")
    if a.causal_ball_reference != (a.causal_protocol is not None) or a.causal_ball_reference != (
        a.causal_protocol_hash is not None
    ):
        p.error("causal feedback requires its exact predeclared protocol")
    if a.causal_ball_reference and (
        factorial or not a.reference_all_initial_velocities or a.reference_phase_start != 0
    ):
        p.error("causal feedback requires the original full-velocity reference initial state")
    if factorial != (a.factorial_protocol is not None) or factorial != (
        a.factorial_protocol_hash is not None
    ):
        p.error("ball factors require their exact predeclared protocol and hash")
    if factorial and not a.reference_all_initial_velocities:
        p.error("ball factors require complete recorded initial velocities")
    if (
        a.reference_ball_initial_velocity or a.reference_all_initial_velocities
    ) and a.motion_reference_receipt is None:
        p.error("moving-ball research initialisation requires a derived reference receipt")
    if a.reference_ball_initial_velocity and a.reference_all_initial_velocities:
        p.error("choose one explicit initial velocity protocol")
    if not 0 <= a.reference_phase_start <= 50 or (
        a.reference_phase_start and a.motion_reference_receipt is None
    ):
        p.error("bounded nonzero reference phase requires a derived reference receipt")
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.projected_target and (
        a.projection_protocol.is_symlink()
        or digest(a.projection_protocol) != a.projection_protocol_hash
    ):
        raise ValueError("exact predeclared projection protocol required")
    if factorial and (
        a.factorial_protocol.is_symlink()
        or digest(a.factorial_protocol) != a.factorial_protocol_hash
    ):
        raise ValueError("exact predeclared factorial protocol required")
    if a.causal_ball_reference and (
        a.causal_protocol.is_symlink() or digest(a.causal_protocol) != a.causal_protocol_hash
    ):
        raise ValueError("exact predeclared causal protocol required")
    external = a.external_source.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(external), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(external), "status", "--porcelain"], text=True
    )
    if (
        head != a.external_head
        or dirty
        or a.output.exists()
        or a.body_source.is_symlink()
        or digest(a.body_source) != a.body_source_hash
        or any(name in sys.modules for name in ("common", "policy", "FSM"))
    ):
        raise ValueError("fresh isolated process, unused output and exact frozen sources required")
    old = json.loads(a.body_source.read_text())
    pins = dict(old["input_file_hashes"])
    pins[str(a.body_source)] = a.body_source_hash
    derived_receipt = None
    if (a.motion_reference_receipt is None) != (a.motion_reference_receipt_hash is None):
        raise ValueError("derived motion requires its exact receipt hash")
    if a.motion_reference_receipt is not None:
        if (
            a.motion_reference is None
            or a.motion_reference_receipt.is_symlink()
            or digest(a.motion_reference_receipt) != a.motion_reference_receipt_hash
            or a.motion_reference.resolve()
            != (a.motion_reference_receipt.parent / "reference.npz").resolve()
        ):
            raise ValueError("exact derived motion and receipt required")
        derived_receipt = json.loads(a.motion_reference_receipt.read_text())
        if (
            derived_receipt.get("status") != "RETARGETED_TRAIN_RECEIVING_REFERENCE_NOT_PHYSICS"
            or derived_receipt.get("artifact_hash") != digest(a.motion_reference)
            or derived_receipt.get("noncommercial_source") is not True
            or derived_receipt.get("promotion_authorized") is not False
            or derived_receipt.get("physics_steps") != 0
        ):
            raise ValueError("exact unpromoted reference-only receipt required")
        pins[str(a.motion_reference.resolve())] = derived_receipt["artifact_hash"]
        pins[str(a.motion_reference_receipt.resolve())] = a.motion_reference_receipt_hash
    motor_plan = None
    if (a.motor_pilot is None) != (a.motor_pilot_hash is None):
        raise ValueError("motor nomination requires its explicit pinned hash")
    if a.motor_pilot is not None:
        from rosclaw_soccer.rsi.cftrack_motor_authority import selected_motor_plan

        nomination = a.motor_pilot / "complete.json"
        if (
            not a.direct_approach_reference
            or a.ankle_roll_reference
            or a.motion_reference is not None
            or nomination.is_symlink()
            or digest(nomination) != a.motor_pilot_hash
        ):
            raise ValueError("exact closed direct-CFgen-only motor nomination required")
        pilot = json.loads(nomination.read_text())
        motor_plan = selected_motor_plan(pilot)
        for path, commitment in pilot["input_file_hashes"].items():
            if path in pins and pins[path] != commitment:
                raise ValueError("nomination source commitments disagree")
            pins[path] = commitment
        pins[str(nomination)] = a.motor_pilot_hash
    # Every external source/asset is committed before import, not just the
    # selected ONNX. No external deploy_omnicontact runner is imported.
    for path in sorted(external.rglob("*")):
        if path.is_file() and path.name != ".git":
            if path.is_symlink():
                raise ValueError("ordinary frozen external assets required")
            pins[str(path)] = digest(path)
    source_pins(pins)
    check_pins(pins)
    import numpy as np

    motion = None
    motion_source = None
    reference_initial = None
    if a.motion_reference is not None:
        from rosclaw_soccer.rsi.cftrack_motion_reference import load_cftrack_motion

        if a.direct_approach_reference or a.ankle_roll_reference or a.motion_reference.is_symlink():
            raise ValueError("whole-body motion and procedural conditioning are distinct protocols")
        motion_source = a.motion_reference.resolve()
        if str(motion_source) not in pins or (
            derived_receipt is None and not motion_source.is_relative_to(external)
        ):
            raise ValueError("motion must be in the pinned external source")
        motion = load_cftrack_motion(motion_source)
        if a.reference_phase_start >= len(motion["joint_pos"]) - 8:
            raise ValueError("sufficient original reference frames after phase start required")
        if a.reference_all_initial_velocities:
            from rosclaw_soccer.rsi.omnicontact_initial_velocity import reference_initial_velocities
            from rosclaw_soccer.training.omnicontact_reference import LAB_TO_MUJOCO

            reference_initial = reference_initial_velocities(motion, np.array(LAB_TO_MUJOCO))
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.omnicontact_cold_entry_standard_body_research.v1",
            source_head=a.expected_head,
            external_head=head,
            input_file_hashes=pins,
            actual_world_ceiling=1,
            control_ceiling=600,
            native_step_ceiling=6000,
            neural_call_ceiling=600,
            neural_constructor_warmups=0,
            native_dt_sec=0.002,
            control_dt_sec=0.02,
            initial_robot_root=[0.0, 0.0, 0.77]
            if motion is None
            else motion["body_pos_w"][0, 0].tolist(),
            initial_ball=[1.0, 0.0, 0.11]
            if motion is None
            else [
                *motion["object_pos_w"][0, :2].tolist(),
                0.11 if a.ground_reference_ball else max(0.11, float(motion["object_pos_w"][0, 2])),
            ],
            goal=[5.0, 0.0, 0.11],
            initial_joint_pose="external_DefaultPose"
            if motion is None
            else "motion_frame_zero_Lab_to_HG",
            pd_profile="external_OmniContact_yaml_not_RoboNaldo",
            torque_cap_fraction=0.85,
            software_history="original_cold_zero_not_measured_five_frames",
            ball_radius_m=0.11,
            ball_mass_kg=0.43,
            original_external_ball_radius_m=0.10,
            original_external_scene_not_used=True,
            policy="kick_50k.onnx",
            reference="current_initial_body_CFgen_kickball"
            if motion is None
            else "complete_NPZmotion_frame_zero_no_trimming",
            motion_source=str(motion_source) if motion_source else None,
            derived_motion_receipt_hash=a.motion_reference_receipt_hash,
            derived_reference_ball_lift_m=None
            if derived_receipt is None
            else derived_receipt["ball_reference_lift_m"],
            reference_phase_start=a.reference_phase_start,
            initial_physical_state_remains_reference_frame_zero=True,
            motion_root_convention="author_body_index_zero_not_redundant_base_arrays",
            motion_base_body_max_position_difference_m=None
            if motion is None
            else float(
                np.linalg.norm(motion["base_pos_w"] - motion["body_pos_w"][:, 0], axis=-1).max()
            ),
            reference_contact_is_not_actual_contact=True,
            reference_joint_positions_are_not_executed_targets=True,
            initial_velocity="reference_full_body_and_ball_with_local_angular_conversion"
            if a.reference_all_initial_velocities
            else "reference_ball_linear_body_and_ball_angular_zero"
            if a.reference_ball_initial_velocity
            else "zero_author_reset_convention",
            initial_ball_linear_velocity_mps=(
                motion["object_lin_vel_w"][0].tolist()
                if (a.reference_ball_initial_velocity or a.reference_all_initial_velocities)
                and motion is not None
                else [0.0, 0.0, 0.0]
            ),
            initial_reference_velocity_components=None
            if reference_initial is None
            else {k: v.tolist() for k, v in reference_initial.items()},
            ground_reference_ball=a.ground_reference_ball,
            zero_reference_ball_spin=a.zero_reference_ball_spin,
            factorial_protocol_hash=a.factorial_protocol_hash,
            causal_ball_reference=a.causal_ball_reference,
            causal_protocol_hash=a.causal_protocol_hash,
            projected_target=a.projected_target,
            projection_protocol_hash=a.projection_protocol_hash,
            direct_approach_reference=a.direct_approach_reference,
            ankle_roll_reference=a.ankle_roll_reference,
            motor_plan=asdict(motor_plan) if motor_plan else None,
            motor_nomination_hash=a.motor_pilot_hash,
            ankle_tracking_error_frame="ankle_roll"
            if motion is not None or a.ankle_roll_reference
            else "ankle_pitch",
            intended_scope="interface_and_single_native_baseline_not_generalized_teacher",
            license_commercial_publicity_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
            neural_optimizer_updates=0,
        ),
    )
    import mujoco
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    scene = [Path(path) for path in pins if path.endswith("/cached-standard/declared-scene.xml")]
    if len(scene) != 1:
        raise ValueError("unique source-pinned standard body required")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scene[0].read_text(), prefixes=("a/",))
    )
    if (
        compiled_model_hash(model)
        != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("compiled standard body mismatch; no experiment allocated")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    sys.path.insert(0, str(external))
    from common.ctrlcomp import PolicyOutput, StateAndCmd
    from policy.omnicontact.CFgen_meta1_loco import DEFAULT_JOINT_POS_MJ
    from policy.omnicontact.OmniContact import OmniContact

    policy = OmniContact(
        StateAndCmd(29),
        PolicyOutput(29),
        onnx_path=str(external / "policy/omnicontact/model/kick_50k.onnx"),
    )
    if tuple(policy.kinematics.joint_names) != tuple(G1_DDS_JOINT_NAMES):
        raise ValueError("external FK/actual HG joint order mismatch")
    if reference_initial is not None and not np.array_equal(policy.lab2mj, LAB_TO_MUJOCO):
        raise ValueError("reference and policy joint order differ")
    policy.task, policy.reference_source = "kickball", "CFgen"
    if motion is not None:
        policy.reference_source = "NPZmotion"
        policy.npz_dir = str(motion_source)
        policy.tracking_start_frame, policy.tracking_end_frame = a.reference_phase_start, -1
    policy.active_object_name = "ball"
    policy.box_dims = policy.ball_dims = np.full(3, 0.11, np.float32)
    policy.bbox_scale = policy.box_dims * 2
    policy.bbox_offsets_scaled = policy.bbox_offsets * policy.bbox_scale.reshape(1, 3)
    data = mujoco.MjData(model)
    data.qpos[lo.rq : lo.rq + 7] = [0, 0, 0.77, 1, 0, 0, 0]
    data.qpos[lo.qi] = DEFAULT_JOINT_POS_MJ
    data.qpos[lo.bq : lo.bq + 7] = [1, 0, 0.11, 1, 0, 0, 0]
    if motion is not None:
        data.qpos[lo.rq : lo.rq + 3] = motion["body_pos_w"][0, 0]
        data.qpos[lo.rq + 3 : lo.rq + 7] = motion["body_quat_w"][0, 0]
        data.qpos[lo.qi] = motion["joint_pos"][0, policy.lab2mj]
        data.qpos[lo.bq : lo.bq + 3] = motion["object_pos_w"][0]
        data.qpos[lo.bq + 2] = max(0.11, float(data.qpos[lo.bq + 2]))
        data.qpos[lo.bq + 3 : lo.bq + 7] = motion["object_quat_w"][0]
        if a.reference_ball_initial_velocity:
            initial_velocity = motion["object_lin_vel_w"][0]
            if not np.isfinite(initial_velocity).all() or np.linalg.norm(initial_velocity) > 5:
                raise ValueError("bounded finite reference ball linear velocity required")
            data.qvel[lo.bv : lo.bv + 3] = initial_velocity
        if reference_initial is not None:
            adjusted_spin = reference_initial["ball_angular_local"]
            if factorial:
                from rosclaw_soccer.rsi.initial_ball_factorial import initial_ball_factorial

                adjusted_position, adjusted_spin = initial_ball_factorial(
                    data.qpos[lo.bq : lo.bq + 3],
                    adjusted_spin,
                    ground_ball=a.ground_reference_ball,
                    zero_spin=a.zero_reference_ball_spin,
                )
                data.qpos[lo.bq : lo.bq + 3] = adjusted_position
            data.qvel[lo.rv : lo.rv + 3] = reference_initial["root_linear"]
            data.qvel[lo.rv + 3 : lo.rv + 6] = reference_initial["root_angular_local"]
            data.qvel[lo.vi] = reference_initial["joints"]
            data.qvel[lo.bv : lo.bv + 3] = reference_initial["ball_linear"]
            data.qvel[lo.bv + 3 : lo.bv + 6] = adjusted_spin
    mujoco.mj_forward(model, data)
    order = policy.lab2mj
    params = NativePDParameters(
        policy.kps_lab[order], policy.kds_lab[order], 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
    )
    world = SharedNativeMotorWorld(
        model, data, players={"a/": params}, ball_body_name="soccer_ball"
    )
    projection_initial = data.qpos[lo.qi].copy()
    captured_obs, captured_actions = [], []
    original_session = policy.ort_session

    class CaptureSession:
        def get_inputs(self):
            return original_session.get_inputs()

        def get_outputs(self):
            return original_session.get_outputs()

        def get_providers(self):
            return original_session.get_providers()

        def run(self, outputs, inputs):
            if len(captured_obs) >= 600 or inputs["obs"].shape != (1, 1244):
                raise ValueError("declared finite CFtrack neural budget/interface exceeded")
            result = original_session.run(outputs, inputs)
            captured_obs.append(inputs["obs"].copy())
            captured_actions.append(result[0].copy())
            return result

    rows, targets, errors, decoded, causal_offsets = [], [], [], [], []
    failure = None
    reference_conditioning = None
    foot_frame_conditioning = None
    try:
        player_class = SharedOmniContactPlayer
        if a.projected_target:
            from rosclaw_soccer.rsi.sim_target_feasibility import ProjectedSharedOmniContactPlayer

            player_class = ProjectedSharedOmniContactPlayer
        player = player_class(
            world,
            "a/",
            policy,
            expected_policy_type=OmniContact,
            goal=np.array([5.0, 0.0, 0.11]),
            reference_source="CFgen" if motion is None else "NPZmotion",
        )
        causal_reference = None
        if a.causal_ball_reference:
            from rosclaw_soccer.rsi.causal_ball_reference import CausalBallReference

            causal_reference = CausalBallReference(policy)
        if a.direct_approach_reference:
            from rosclaw_soccer.rsi.cftrack_direct_approach import condition_direct_kick_approach

            reference_conditioning = condition_direct_kick_approach(policy)
            world._require_current_boundary()
            write_once(a.output / "before-actual-reference.json", reference_conditioning)
        if a.ankle_roll_reference:
            from rosclaw_soccer.rsi.cftrack_foot_frame_reference import (
                condition_cftrack_roll_reference,
            )

            foot_frame_conditioning = condition_cftrack_roll_reference(policy)
            world._require_current_boundary()
            write_once(a.output / "before-actual-foot-frame.json", foot_frame_conditioning)
        if motor_plan is not None:
            player.install_motor_plan(motor_plan)
            write_once(
                a.output / "before-actual-motor.json",
                dict(
                    motor_plan=asdict(motor_plan),
                    nomination_hash=a.motor_pilot_hash,
                    hardware_authorized=False,
                ),
            )
        policy.ort_session = CaptureSession()
        for _ in range(600):
            if causal_reference is not None:
                causal_offsets.append(causal_reference.apply(data.xpos[lo.ball].copy()))
            target, measured = player.propose()
            row = world.advance({"a/": target})
            rows.append(row)
            targets.append(target)
            decoded.append(measured["nn_decoded_target"])
            player.commit()
            # FK on scratch to measure actual endpoint, never write world.
            scratch = mujoco.MjData(model)
            scratch.qpos[:] = data.qpos
            mujoco.mj_kinematics(model, scratch)
            errors.append(
                [
                    float(
                        np.linalg.norm(
                            scratch.xpos[
                                model.body(
                                    "a/"
                                    + side
                                    + (
                                        "_ankle_roll_link"
                                        if motion is not None or a.ankle_roll_reference
                                        else "_ankle_pitch_link"
                                    )
                                ).id
                            ]
                            - goal[:3]
                        )
                    )
                    for side, goal in (
                        ("left", policy.policy_output.l_ankle_goal),
                        ("right", policy.policy_output.r_ankle_goal),
                    )
                ]
            )
            if row["body_failed"]["a/"]:
                break
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
    # Save before task assertions. A runtime failure is evidence, not a skip.
    arrays = (
        {
            key: np.concatenate([r[key] for r in rows])
            for key in ("qpos", "qvel", "torque", "ball_forces")
        }
        if rows
        else {}
    )
    arrays.update(
        joint_targets=np.array(targets),
        nn_decoded_targets=np.array(decoded),
        neural_obs=np.array(captured_obs),
        neural_actions=np.array(captured_actions),
        ankle_error_m=np.array(errors),
    )
    if a.causal_ball_reference:
        arrays["causal_ankle_offset_xy_m"] = np.asarray(causal_offsets)
    artifact = a.output / "actual-unvalidated.npz"
    np.savez_compressed(artifact, **arrays)
    check_pins(pins)
    if factorial and digest(a.factorial_protocol) != a.factorial_protocol_hash:
        raise ValueError("factorial protocol changed during actual physics")
    if a.causal_ball_reference and digest(a.causal_protocol) != a.causal_protocol_hash:
        raise ValueError("causal protocol changed during actual physics")
    if a.projected_target and digest(a.projection_protocol) != a.projection_protocol_hash:
        raise ValueError("projection protocol changed during actual physics")
    clean(a.expected_head)
    forces = arrays.get("ball_forces", np.zeros((0, 1, 3)))[:, 0]
    foot = np.flatnonzero((forces[:, :2] > 1).any(axis=1))
    nonfoot = np.flatnonzero(forces[:, 2] > 1)
    projection_stats = None
    if a.projected_target and targets:
        target_array, decoded_array = np.asarray(targets), np.asarray(decoded)
        limits = model.jnt_range[lo.joints]
        rates = np.diff(np.vstack((projection_initial, target_array)), axis=0)
        projection_stats = dict(
            initial_hg=projection_initial.tolist(),
            physical_limits_hg=limits.tolist(),
            maximum_absolute_step_rad=float(np.max(np.abs(rates))),
            maximum_target_distortion_rad=float(np.max(np.abs(target_array - decoded_array))),
            decoded_range_violations=int(
                ((decoded_array < limits[:, 0]) | (decoded_array > limits[:, 1])).sum()
            ),
            projected_range_violations=int(
                ((target_array < limits[:, 0]) | (target_array > limits[:, 1])).sum()
            ),
            executed_target_feedback_in_nn_history=False,
        )
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_NATIVE_BASELINE_RECORDED_NOT_PROMOTED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            external_head=head,
            runtime_failure=failure,
            reference_conditioning=reference_conditioning,
            foot_frame_conditioning=foot_frame_conditioning,
            motor_plan=asdict(motor_plan) if motor_plan else None,
            motor_nomination_hash=a.motor_pilot_hash,
            reference_source="CFgen" if motion is None else "NPZmotion",
            motion_source=str(motion_source) if motion_source else None,
            reference_ball_initial_velocity=a.reference_ball_initial_velocity,
            reference_all_initial_velocities=a.reference_all_initial_velocities,
            ground_reference_ball=a.ground_reference_ball,
            zero_reference_ball_spin=a.zero_reference_ball_spin,
            factorial_protocol_hash=a.factorial_protocol_hash,
            causal_ball_reference=a.causal_ball_reference,
            causal_protocol_hash=a.causal_protocol_hash,
            projected_target=a.projected_target,
            projection_protocol_hash=a.projection_protocol_hash,
            projection_stats=projection_stats,
            reference_phase_start=a.reference_phase_start,
            controls=world.control_steps,
            native_steps=world.native_steps,
            new_neural_calls=len(captured_obs),
            neural_constructor_warmups=0,
            artifact=dict(path=artifact.name, hash=digest(artifact)),
            first_foot_native_index=int(foot[0]) if len(foot) else None,
            first_nonfoot_native_index=int(nonfoot[0]) if len(nonfoot) else None,
            maximum_nonfoot_force_n=float(forces[:, 2].max()) if len(forces) else None,
            final_ball_qpos=data.qpos[lo.bq : lo.bq + 7].tolist(),
            body_failed=world.failed,
            nn_weights_trained_by_this_trial=False,
            new_controller_new_initial_condition_not_RoboNaldo_H300_certificate=True,
            license_commercial_publicity_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
