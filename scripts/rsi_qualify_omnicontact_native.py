"""One research-only CFtrack cold-entry trial in the existing standard native body.

This is a NEW controller/protocol, not a continuation of a RoboNaldo certificate.
External policy code is read from an explicitly frozen, clean SIM-only checkout.
"""

import argparse
import json
import subprocess
import sys
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
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
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
    if a.motion_reference is not None:
        from rosclaw_soccer.rsi.cftrack_motion_reference import load_cftrack_motion

        if a.direct_approach_reference or a.motion_reference.is_symlink():
            raise ValueError("whole-body motion and procedural conditioning are distinct protocols")
        motion_source = a.motion_reference.resolve()
        if str(motion_source) not in pins or not motion_source.is_relative_to(external):
            raise ValueError("motion must be in the pinned external source")
        motion = load_cftrack_motion(motion_source)
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
                max(0.11, float(motion["object_pos_w"][0, 2])),
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
            motion_root_convention="author_body_index_zero_not_redundant_base_arrays",
            motion_base_body_max_position_difference_m=None
            if motion is None
            else float(
                np.linalg.norm(motion["base_pos_w"] - motion["body_pos_w"][:, 0], axis=-1).max()
            ),
            reference_contact_is_not_actual_contact=True,
            reference_joint_positions_are_not_executed_targets=True,
            initial_velocity="zero_author_reset_convention",
            direct_approach_reference=a.direct_approach_reference,
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
    policy.task, policy.reference_source = "kickball", "CFgen"
    if motion is not None:
        policy.reference_source = "NPZmotion"
        policy.npz_dir = str(motion_source)
        policy.tracking_start_frame, policy.tracking_end_frame = 0, -1
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
    mujoco.mj_forward(model, data)
    order = policy.lab2mj
    params = NativePDParameters(
        policy.kps_lab[order], policy.kds_lab[order], 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
    )
    world = SharedNativeMotorWorld(
        model, data, players={"a/": params}, ball_body_name="soccer_ball"
    )
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

    policy.ort_session = CaptureSession()
    rows, targets, errors = [], [], []
    failure = None
    reference_conditioning = None
    try:
        player = SharedOmniContactPlayer(
            world,
            "a/",
            policy,
            expected_policy_type=OmniContact,
            goal=np.array([5.0, 0.0, 0.11]),
            reference_source="CFgen" if motion is None else "NPZmotion",
        )
        if a.direct_approach_reference:
            from rosclaw_soccer.rsi.cftrack_direct_approach import condition_direct_kick_approach

            reference_conditioning = condition_direct_kick_approach(policy)
            world._require_current_boundary()
            write_once(a.output / "before-actual-reference.json", reference_conditioning)
        for _ in range(600):
            target, _ = player.propose()
            row = world.advance({"a/": target})
            rows.append(row)
            targets.append(target)
            player.commit()
            # FK on scratch to measure actual endpoint, never write world.
            scratch = mujoco.MjData(model)
            scratch.qpos[:] = data.qpos
            mujoco.mj_kinematics(model, scratch)
            errors.append(
                [
                    float(
                        np.linalg.norm(
                            scratch.xpos[model.body("a/" + side + "_ankle_pitch_link").id]
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
        neural_obs=np.array(captured_obs),
        neural_actions=np.array(captured_actions),
        ankle_error_m=np.array(errors),
    )
    artifact = a.output / "actual-unvalidated.npz"
    np.savez_compressed(artifact, **arrays)
    check_pins(pins)
    clean(a.expected_head)
    forces = arrays.get("ball_forces", np.zeros((0, 1, 3)))[:, 0]
    foot = np.flatnonzero((forces[:, :2] > 1).any(axis=1))
    nonfoot = np.flatnonzero(forces[:, 2] > 1)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_NATIVE_BASELINE_RECORDED_NOT_PROMOTED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            external_head=head,
            runtime_failure=failure,
            reference_conditioning=reference_conditioning,
            reference_source="CFgen" if motion is None else "NPZmotion",
            motion_source=str(motion_source) if motion_source else None,
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
