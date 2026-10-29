"""Rebuild actual CFtrack1244 histories and targets; no native integration."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def rebuild_projected_target(decoded, previous, limits):
    """Independent readback math; no source controller or actuator use."""
    import numpy as np

    bounded = np.maximum(limits[:, 0], np.minimum(limits[:, 1], decoded))
    return np.maximum(previous - 0.08, np.minimum(previous + 0.08, bounded))


def validate_scope(report, *, derived_moving_partial):
    """Admit actual complete control boundaries, including early body failure."""
    controls = report.get("controls")
    if (
        type(derived_moving_partial) is not bool
        or report.get("status") != "CFTRACK_NATIVE_BASELINE_RECORDED_NOT_PROMOTED"
        or report.get("runtime_failure") is not None
        or report.get("promotion_authorized") is not False
        or report.get("hardware_authorized") is not False
        or type(controls) is not int
        or not 1 <= controls <= 600
        or (not derived_moving_partial and controls != 600)
        or report.get("native_steps") != controls * 10
        or report.get("new_neural_calls") != controls
        or (
            derived_moving_partial
            and (
                report.get("reference_source") != "NPZmotion"
                or report.get("reference_all_initial_velocities") is not True
                or report.get("reference_ball_initial_velocity") is not False
                or report.get("reference_phase_start") != 0
            )
        )
    ):
        raise ValueError("actual unpromoted bounded native controls required")
    return controls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--qualification-hash", required=True)
    p.add_argument("--external-source", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--derived-moving-partial", action="store_true")
    p.add_argument("--factorial-protocol", type=Path)
    p.add_argument("--causal-protocol", type=Path)
    p.add_argument("--projection-protocol", type=Path)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete = a.qualification / "complete.json"
    if (
        a.output.exists()
        or complete.is_symlink()
        or digest(complete) != a.qualification_hash
        or any(x in sys.modules for x in ("common", "policy", "FSM"))
    ):
        raise ValueError("unused isolated readback and exact producer required")
    report = json.loads(complete.read_text())
    controls = validate_scope(report, derived_moving_partial=a.derived_moving_partial)
    projected = report.get("projected_target", False)
    if (
        type(projected) is not bool
        or projected != (a.projection_protocol is not None)
        or (projected and not a.derived_moving_partial)
        or (
            projected
            and (
                a.projection_protocol.is_symlink()
                or digest(a.projection_protocol) != report.get("projection_protocol_hash")
            )
        )
    ):
        raise ValueError("source-bound explicit target projection protocol required")
    causal = report.get("causal_ball_reference", False)
    if (
        type(causal) is not bool
        or causal != (a.causal_protocol is not None)
        or (causal and not a.derived_moving_partial)
        or (
            causal
            and (
                a.causal_protocol.is_symlink()
                or digest(a.causal_protocol) != report.get("causal_protocol_hash")
            )
        )
    ):
        raise ValueError("source-bound explicit causal protocol required")
    ground = report.get("ground_reference_ball", False)
    no_spin = report.get("zero_reference_ball_spin", False)
    factorial = ground or no_spin
    if (
        type(ground) is not bool
        or type(no_spin) is not bool
        or factorial != (a.factorial_protocol is not None)
        or (factorial and not a.derived_moving_partial)
        or (
            factorial
            and (
                a.factorial_protocol.is_symlink()
                or digest(a.factorial_protocol) != report.get("factorial_protocol_hash")
            )
        )
    ):
        raise ValueError("source-bound explicit factorial protocol required")
    pins = dict(report["input_file_hashes"])
    pins[str(complete)] = a.qualification_hash
    artifact = a.qualification / report["artifact"]["path"]
    if (
        artifact.parent != a.qualification
        or artifact.is_symlink()
        or digest(artifact) != report["artifact"]["hash"]
    ):
        raise ValueError("bound native raw required")
    pins[str(artifact)] = report["artifact"]["hash"]
    external = a.external_source.resolve()
    if str(external / "policy/omnicontact/OmniContact.py") not in pins:
        raise ValueError("exact producer-pinned external policy source required")
    source_pins(pins)
    check_pins(pins)
    reference_source = report.get("reference_source", "CFgen")
    motion = None
    motion_path = None
    if reference_source == "NPZmotion":
        from rosclaw_soccer.rsi.cftrack_motion_reference import load_cftrack_motion

        motion_path = Path(report["motion_source"])
        derived_receipt = motion_path.parent / "complete.json"
        if (
            str(motion_path) not in pins
            or motion_path.is_symlink()
            or (not a.derived_moving_partial and not motion_path.is_relative_to(external))
            or report.get("reference_conditioning") is not None
            or report.get("foot_frame_conditioning") is not None
            or report.get("motor_plan") is not None
        ):
            raise ValueError("pinned complete motion without procedural conditioning required")
        if a.derived_moving_partial:
            if (
                derived_receipt.is_symlink()
                or str(derived_receipt) not in pins
                or digest(derived_receipt) != pins[str(derived_receipt)]
            ):
                raise ValueError("producer-bound derived reference receipt required")
            derived = json.loads(derived_receipt.read_text())
            if (
                derived.get("status") != "RETARGETED_TRAIN_RECEIVING_REFERENCE_NOT_PHYSICS"
                or derived.get("artifact_hash") != digest(motion_path)
                or derived.get("physics_steps") != 0
                or derived.get("promotion_authorized") is not False
            ):
                raise ValueError("exact unpromoted derived reference required")
        motion = load_cftrack_motion(motion_path)
    elif reference_source != "CFgen" or report.get("motion_source") is not None:
        raise ValueError("explicit qualified CFtrack reference source required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            maximum_new_neural_calls=controls,
            constructor_warmups=0,
            new_physical_steps=0,
            neural_optimizer_updates=0,
            reference_source=reference_source,
            hardware_authorized=False,
        ),
    )
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import label_first_touch

    with np.load(artifact, allow_pickle=False) as saved:
        raw = {k: saved[k].copy() for k in saved.files}
    for key, shape in (
        ("qpos", (controls * 10, 43)),
        ("qvel", (controls * 10, 41)),
        ("torque", (controls * 10, 29)),
        ("ball_forces", (controls * 10, 1, 3)),
        ("joint_targets", (controls, 29)),
        ("neural_obs", (controls, 1, 1244)),
        ("neural_actions", (controls, 1, 29)),
    ):
        if raw[key].shape != shape or not np.isfinite(raw[key]).all():
            raise ValueError("full finite actual trace required: " + key)
    if causal and (
        raw.get("causal_ankle_offset_xy_m", np.empty(0)).shape != (controls, 2)
        or not np.isfinite(raw["causal_ankle_offset_xy_m"]).all()
    ):
        raise ValueError("complete actual causal reference updates required")
    scene = [Path(path) for path in pins if path.endswith("/cached-standard/declared-scene.xml")]
    if len(scene) != 1:
        raise ValueError("unique pinned native scene required")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scene[0].read_text(), prefixes=("a/",))
    )
    if (
        compiled_model_hash(model)
        != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("compiled native body mismatch")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    if (abs(raw["torque"][:, lo.ai]) > 0.85 * np.array(G1_HARD_TORQUE_LIMITS) + 1e-12).any():
        raise ValueError("native torque caps violated")
    sys.path.insert(0, str(external))
    from common.ctrlcomp import PolicyOutput, StateAndCmd
    from policy.omnicontact.CFgen_meta1_loco import DEFAULT_JOINT_POS_MJ
    from policy.omnicontact.OmniContact import OmniContact

    policy = OmniContact(
        StateAndCmd(29),
        PolicyOutput(29),
        onnx_path=str(external / "policy/omnicontact/model/kick_50k.onnx"),
    )
    policy.task, policy.reference_source, policy.active_object_name = "kickball", "CFgen", "ball"
    if motion is not None:
        policy.reference_source = "NPZmotion"
        policy.npz_dir = str(motion_path)
        policy.tracking_start_frame, policy.tracking_end_frame = 0, -1
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
        if a.derived_moving_partial:
            from rosclaw_soccer.rsi.omnicontact_initial_velocity import reference_initial_velocities

            initial = reference_initial_velocities(motion, np.asarray(policy.lab2mj))
            data.qvel[lo.rv : lo.rv + 3] = initial["root_linear"]
            data.qvel[lo.rv + 3 : lo.rv + 6] = initial["root_angular_local"]
            data.qvel[lo.vi] = initial["joints"]
            data.qvel[lo.bv : lo.bv + 3] = initial["ball_linear"]
            data.qvel[lo.bv + 3 : lo.bv + 6] = initial["ball_angular_local"]
            if factorial:
                from rosclaw_soccer.rsi.initial_ball_factorial import initial_ball_factorial

                position, spin = initial_ball_factorial(
                    data.qpos[lo.bq : lo.bq + 3],
                    data.qvel[lo.bv + 3 : lo.bv + 6],
                    ground_ball=ground,
                    zero_spin=no_spin,
                )
                data.qpos[lo.bq : lo.bq + 3] = position
                data.qvel[lo.bv + 3 : lo.bv + 6] = spin
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    # A view of scratch measurements, NOT a SharedNativeMotorWorld. Assigning
    # saved states here never mutates an execution world or advances its clock.
    view = SimpleNamespace(
        world=SimpleNamespace(data=data, control_steps=0), policy=policy, layout=lo
    )
    SharedOmniContactPlayer._observe(view)
    policy.goal_pos_override = np.array([5.0, 0.0, 0.11], np.float32)
    policy.enter()
    causal_reference = None
    if causal:
        from rosclaw_soccer.rsi.causal_ball_reference import CausalBallReference

        causal_reference = CausalBallReference(policy)
    if report.get("reference_conditioning") is not None:
        from rosclaw_soccer.rsi.cftrack_direct_approach import condition_direct_kick_approach
        from rosclaw_soccer.sim.contracts import hash_json

        if hash_json(condition_direct_kick_approach(policy)) != hash_json(
            report["reference_conditioning"]
        ):
            raise ValueError("current-state direct reference does not regenerate")
    original_session = policy.ort_session
    if report.get("foot_frame_conditioning") is not None:
        from rosclaw_soccer.rsi.cftrack_foot_frame_reference import condition_cftrack_roll_reference
        from rosclaw_soccer.sim.contracts import hash_json

        if hash_json(condition_cftrack_roll_reference(policy)) != hash_json(
            report["foot_frame_conditioning"]
        ):
            raise ValueError("full-motion ankle frame binding does not regenerate")
    motor_plan = None
    if report.get("motor_plan") is not None:
        from rosclaw_soccer.rsi.cftrack_motor_residual_plan import CFtrackMotorResidualPlan
        from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash

        if (
            reference_source != "CFgen"
            or report.get("foot_frame_conditioning") is not None
            or report.get("reference_conditioning") is None
        ):
            raise ValueError("motor audit requires the original direct-CFgen reference")
        value = report["motor_plan"]
        motor_plan = CFtrackMotorResidualPlan(
            value["origin_counter"],
            value["start_control"],
            value["peak_control"],
            tuple(value["delta_hg"]),
            value["donor_policy_hash"],
        )
        motor_plan.require_donor(
            policy_hash=cftrack_state_hash(policy), counter=policy.counter_step
        )
        if raw.get("nn_decoded_targets", np.empty(0)).shape != (600, 29):
            raise ValueError("separately recorded original NN targets required")
    calls = []

    class VerifySession:
        def run(self, outputs, inputs):
            tick = len(calls)
            if (
                tick >= controls
                or not np.array_equal(inputs["obs"], raw["neural_obs"][tick])
                or not np.array_equal(inputs["time_step"], np.zeros((1, 1), np.float32))
            ):
                raise ValueError("actual proprioception/reference/history1244 does not rebuild")
            result = original_session.run(outputs, inputs)
            calls.append(tick)
            if not np.array_equal(result[0], raw["neural_actions"][tick]):
                raise ValueError("original CFtrack NN action differs")
            return result

    policy.ort_session = VerifySession()
    previous_target = data.qpos[lo.qi].astype(np.float64, copy=True)
    physical_limits = model.jnt_range[lo.joints].astype(np.float64)
    if projected and (
        (previous_target < physical_limits[:, 0]).any()
        or (previous_target > physical_limits[:, 1]).any()
    ):
        raise ValueError("projection initial measured position violates physical range")
    for tick in range(controls):
        if tick:
            # Preserve author's mixed-time object cache, as in native mj_step.
            data.qpos[:] = raw["qpos"][tick * 10 - 2]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            data.qpos[:], data.qvel[:] = raw["qpos"][tick * 10 - 1], raw["qvel"][tick * 10 - 1]
        view.world.control_steps = tick
        SharedOmniContactPlayer._observe(view)
        if causal_reference is not None and not np.array_equal(
            causal_reference.apply(data.xpos[lo.ball].copy()), raw["causal_ankle_offset_xy_m"][tick]
        ):
            raise ValueError("actual causal reference update differs")
        policy.run()
        decoded = policy.policy_output.actions.astype(np.float64)
        if "nn_decoded_targets" in raw and not np.array_equal(
            decoded, raw["nn_decoded_targets"][tick]
        ):
            raise ValueError("original NN decoded target differs")
        applied = (
            motor_plan.apply(
                decoded, counter=tick, limits_hg=model.jnt_range[lo.joints].astype(np.float64)
            )
            if motor_plan is not None
            else decoded
        )
        if projected:
            applied = rebuild_projected_target(decoded, previous_target, physical_limits)
        if (
            policy.counter_step != tick + 1
            or policy.async_stage_thread is not None
            or not np.array_equal(applied, raw["joint_targets"][tick])
        ):
            raise ValueError("private CFtrack tick/decoded native target differs")
        if projected:
            previous_target = applied.copy()
    forces = raw["ball_forces"][:, 0]
    foot = np.flatnonzero((forces[:, :2] > 1).any(axis=1))
    nf = np.flatnonzero(forces[:, 2] > 1)
    if (
        (int(foot[0]) if len(foot) else None) != report["first_foot_native_index"]
        or (int(nf[0]) if len(nf) else None) != report["first_nonfoot_native_index"]
        or float(forces[:, 2].max()) != report["maximum_nonfoot_force_n"]
    ):
        raise ValueError("native contact summary differs")
    exam = None
    if controls >= 300:
        grouped = np.zeros((300, 1, 6))
        grouped[:, 0, :3] = forces[:3000].reshape(300, 10, 3).max(axis=1)
        q = raw["qpos"].reshape(controls, 10, 43)
        exam = label_first_touch(
            dict(
                force_n=grouped,
                pelvis_z_per_substep_m=q[:300, :, lo.rq + 2][:, None],
                ball_position_after_step_m=q[:300, -1, lo.bq : lo.bq + 3][:, None],
            )
        ).to_dict()
    if data.time != 0:
        raise ValueError("readback unexpectedly advanced native time")
    check_pins(pins)
    if factorial and digest(a.factorial_protocol) != report.get("factorial_protocol_hash"):
        raise ValueError("factorial protocol changed during neural readback")
    if causal and digest(a.causal_protocol) != report.get("causal_protocol_hash"):
        raise ValueError("causal protocol changed during neural readback")
    if projected and digest(a.projection_protocol) != report.get("projection_protocol_hash"):
        raise ValueError("projection protocol changed during neural readback")
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_PARTIAL_DERIVED_NN_HISTORY_AND_TARGETS_VERIFIED"
            if a.derived_moving_partial
            else "CFTRACK_ACTUAL_NN1244_HISTORY_AND_TARGETS_VERIFIED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            qualification_hash=a.qualification_hash,
            actual_neural_readbacks=len(calls),
            reference_source=reference_source,
            foot_frame_conditioning=report.get("foot_frame_conditioning"),
            motor_plan=report.get("motor_plan"),
            motor_nomination_hash=report.get("motor_nomination_hash"),
            projected_target=projected,
            projection_protocol_hash=report.get("projection_protocol_hash"),
            motion_source=str(motion_path) if motion_path else None,
            new_initial_condition_not_an_old_exam_certificate=True,
            constructor_warmups=0,
            new_physical_steps=0,
            neural_optimizer_updates=0,
            original_policy_math_reused=True,
            h300_scoring_of_new_cold_entry_protocol=exam,
            first_foot_control=int(foot[0] // 10) if len(foot) else None,
            commercial_publicity_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
