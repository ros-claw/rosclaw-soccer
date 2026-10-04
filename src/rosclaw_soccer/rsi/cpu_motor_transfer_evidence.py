"""Replay actual CPU MuJoCo dynamics and causally reconstruct neural targets.

The compiled model and actuator controls are replayed, not Isaac trajectories.
This detects corrupt trace/report pairs but does not prove sim-to-real safety.
The initial implementation deliberately requires the modern 43/41 state model.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.contact_motor_primitive import JOINT_NAMES
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.independent_first_touch_bank import post_contact_displacement
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.sim.current_kinematic_observation import (
    CurrentKinematicObserver,
    snapshot_from_contract,
)
from rosclaw_soccer.sim.root_velocity_reference import root_velocity_world
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality

if TYPE_CHECKING:
    pass


def audit_cpu_transfer(
    root: Path,
    source_path: Path,
    *,
    sampling_decoder_factory: Any = None,
    mean_decoder_factory: Any = None,
) -> dict[str, Any]:
    import mujoco

    report = _sealed(root / "report.json")
    if mean_decoder_factory is not None:
        from rosclaw_soccer.rsi.proposal_episode_decoder_factory import (
            ProposalEpisodeDecoderFactory,
        )

        policy = report.get("executed_motor_policy")
        expected_factory: Any = ProposalEpisodeDecoderFactory
        proof_key = "proposal_memory_motor_proof"
        if type(policy) is dict and "extended_proposal_motor_proof" in policy:
            from rosclaw_soccer.rsi.extended_proposal_episode_factory import (
                ExtendedProposalEpisodeFactory,
            )

            expected_factory = ExtendedProposalEpisodeFactory
            proof_key = "extended_proposal_motor_proof"
        elif type(policy) is dict and "verified_success_imitation_motor_proof" in policy:
            from rosclaw_soccer.rsi.imitation_proposal_episode_factory import (
                ImitationProposalEpisodeFactory,
            )

            expected_factory = ImitationProposalEpisodeFactory
            proof_key = "verified_success_imitation_motor_proof"
        if (
            type(mean_decoder_factory) is not expected_factory
            or sampling_decoder_factory is not None
            or type(policy) is not dict
            or proof_key not in policy
            or policy.get("policy_hash") != mean_decoder_factory.policy_hash
            or policy.get("policy_hash")
            != hash_json({k: v for k, v in policy.items() if k != "policy_hash"})
        ):
            raise ValueError("source-bound private fixed proposal mean factory required")
    if sampling_decoder_factory is not None:
        from rosclaw_soccer.rsi.owned_proposal_sampling_factory import (
            OwnedProposalSamplingEpisodeFactory,
        )
        from rosclaw_soccer.rsi.proposal_sampling_episode_factory import (
            ProposalSamplingEpisodeFactory,
        )
        from rosclaw_soccer.rsi.smooth_sampling_decoder_factory import SmoothSamplingDecoderFactory

        family = (
            "proposal_sampling_motor_proof"
            if type(sampling_decoder_factory)
            in (ProposalSamplingEpisodeFactory, OwnedProposalSamplingEpisodeFactory)
            else "smooth_memory_motor_proof"
        )
        if type(sampling_decoder_factory) not in (
            SmoothSamplingDecoderFactory,
            ProposalSamplingEpisodeFactory,
            OwnedProposalSamplingEpisodeFactory,
        ) or family not in report.get("executed_motor_policy", {}):
            raise ValueError("only the verified smooth sampling factory is accepted")
    snapshot, trace_path = root / "compiled_model.mjb", root / "physical_trace.npz"
    if (
        report.get("source_hash") != hash_bytes(source_path.read_bytes())
        or report.get("physical_trace_hash") != hash_bytes(trace_path.read_bytes())
        or report.get("compiled_model_hash") != hash_bytes(snapshot.read_bytes())
        or report.get("activation_ceiling") != "SIM_ONLY"
        or report.get("promotion_authorized") is not False
        or report.get("hardware_authorized") is not False
        or report["physics"]["engine"] != "MuJoCo"
        or report["physics"]["device"] != "cpu"
        or report["physics"]["version"] != mujoco.__version__
    ):
        raise ValueError("CPU dynamics source, model, trace or authority contract changed")
    import json

    commitment = json.loads((root / "commitment.json").read_text())
    if report["commitment_hash"] != hash_json(commitment) or any(
        report[k] != v for k, v in commitment.items()
    ):
        raise ValueError("CPU commitment differs from report")
    if ("numeric_compilation" in report) != ("numeric_compilation" in commitment):
        raise ValueError("CPU numerical compilation differs from commitment")
    if ("imitation_proposal_factory" in report) != ("imitation_proposal_factory" in commitment):
        raise ValueError("imitation factory differs from commitment")
    if "imitation_proposal_factory" in commitment:
        from rosclaw_soccer.rsi.imitation_proposal_episode_factory import (
            validate_compilation_contract as validate_imitation_factory,
        )

        if any(
            k in commitment
            for k in (
                "numeric_compilation",
                "numeric_sampling_compilation",
                "body_response_guidance",
                "fixed_proposal_factory",
                "extended_proposal_factory",
            )
        ):
            raise ValueError("imitation factory cannot mix another compilation or guidance claim")
        validate_imitation_factory(
            commitment["imitation_proposal_factory"], report.get("executed_motor_policy", {})
        )
    if ("fixed_proposal_factory" in report) != ("fixed_proposal_factory" in commitment):
        raise ValueError("fixed proposal factory differs from commitment")
    if "fixed_proposal_factory" in commitment:
        from rosclaw_soccer.rsi.proposal_episode_decoder_factory import (
            validate_compilation_contract as validate_fixed_proposal_factory,
        )

        numeric = commitment.get("numeric_compilation")
        if type(numeric) is not dict or numeric.get("implementation") != "owned_snapshot":
            raise ValueError("fixed proposal factory requires original owned snapshot compilation")
        validate_fixed_proposal_factory(
            commitment["fixed_proposal_factory"], report.get("executed_motor_policy", {})
        )
    if ("numeric_sampling_compilation" in report) != ("numeric_sampling_compilation" in commitment):
        raise ValueError("CPU sampling compilation differs from commitment")
    if ("body_response_guidance" in report) != ("body_response_guidance" in commitment):
        raise ValueError("body response guidance differs from commitment")
    if "body_response_guidance" in commitment and (
        "foundation_observation_capture" not in commitment or report.get("step_model_hash") is None
    ):
        raise ValueError(
            "body guidance requires actual foundation capture and explicit proposal parent"
        )
    if "numeric_sampling_compilation" in commitment:
        from rosclaw_soccer.rsi.smooth_decoder_selection import (
            validate_sampling_compilation_contract,
        )

        sampling_provenance = commitment["numeric_sampling_compilation"]
        if isinstance(sampling_provenance, dict) and sampling_provenance.get("schema") == (
            "soccer.rsi.owned_proposal_sampling_compilation.v1"
        ):
            from rosclaw_soccer.rsi.owned_proposal_sampling_factory import (
                validate_compilation_contract as validate_owned_proposal_sampling_contract,
            )

            validate_owned_proposal_sampling_contract(
                sampling_provenance, report.get("executed_motor_policy", {})
            )
        elif isinstance(sampling_provenance, dict) and sampling_provenance.get("schema") == (
            "soccer.rsi.offline_proposal_sampling_compilation.v1"
        ):
            from rosclaw_soccer.rsi.proposal_sampling_episode_factory import (
                validate_compilation_contract as validate_proposal_sampling_contract,
            )

            validate_proposal_sampling_contract(
                sampling_provenance, report.get("executed_motor_policy", {})
            )
        else:
            validate_sampling_compilation_contract(
                sampling_provenance, report.get("executed_motor_policy", {})
            )
    if "numeric_compilation" in commitment:
        from rosclaw_soccer.rsi.proposal_decoder_selection import validate_compilation_contract

        validate_compilation_contract(commitment["numeric_compilation"])
        if hash_json(report["numeric_compilation"]) != hash_json(commitment["numeric_compilation"]):
            raise ValueError("CPU numerical compilation differs from commitment")
        executed = report.get("executed_motor_policy", {})
        proof = executed.get("step_motor_proof") if isinstance(executed, dict) else None
        inner = proof.get("model") if isinstance(proof, dict) else None
        if (
            not isinstance(executed, dict)
            or "proposal_memory_motor_proof" not in executed
            or not isinstance(inner, dict)
            or inner.get("schema") != "soccer.rsi.proposal_memory_motor.v1"
        ):
            raise ValueError("owned compilation requires the sealed proposal motor family")
    if "observation_contract" in commitment and commitment["observation_contract"] is None:
        raise ValueError("explicit null CPU observation contract is ambiguous")
    reference, observation_snapshot = snapshot_from_contract(commitment.get("observation_contract"))
    if report.get("observation_contract") != commitment.get("observation_contract"):
        raise ValueError("CPU observation contract differs from commitment")
    model = mujoco.MjModel.from_binary_path(str(snapshot))
    if (model.nq, model.nv, model.nu) != (43, 41, 29) or model.opt.timestep != 0.002:
        raise ValueError("modern canonical CPU transfer state required")
    names = list(G1_DDS_JOINT_NAMES)
    if report["physics"]["canonical_joint_names"] != names:
        raise ValueError("canonical PD gain ordering changed")
    kp = np.asarray(report["physics"]["canonical_pd_kp"])
    kd = np.asarray(report["physics"]["canonical_pd_kd"])
    if (
        kp.shape != (29,)
        or kd.shape != (29,)
        or not np.isfinite(kp).all()
        or not np.isfinite(kd).all()
        or np.any(kp <= 0)
        or np.any(kd < 0)
    ):
        raise ValueError("finite physical PD controller required")

    def identifier(kind: Any, name: str) -> int:
        index = mujoco.mj_name2id(model, kind, name)
        if index < 0:
            raise ValueError(f"missing CPU model element {name}")
        return int(index)

    joints = [identifier(mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    qi = np.asarray([model.jnt_qposadr[j] for j in joints])
    vi = np.asarray([model.jnt_dofadr[j] for j in joints])
    ai = []
    for joint in joints:
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
        if len(matches) != 1:
            raise ValueError("one actuator for each canonical joint required")
        ai.append(int(matches[0]))
    pelvis = identifier(mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    ball = identifier(mujoco.mjtObj.mjOBJ_BODY, "ball")
    bg = identifier(mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    bj = model.body_jntadr[ball]
    bq, bv = model.jnt_qposadr[bj], model.jnt_dofadr[bj]
    geometry_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_knee_link",
        "right_knee_link",
    ]
    geometry = [identifier(mujoco.mjtObj.mjOBJ_BODY, n) for n in geometry_names]
    contacts = (
        geometry[:2]
        + [
            identifier(mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("left_ankle_pitch_link", "right_ankle_pitch_link")
        ]
        + geometry[2:]
    )
    if (
        model.geom_size[bg, 0] != 0.11
        or model.body_mass[ball] != 0.43
        or np.any(model.dof_damping[bv : bv + 6] != 0)
    ):
        raise ValueError("physical ball dimensions or damping changed")
    shapes = {
        "canonical_qpos": (300, 1, 43),
        "canonical_qvel": (300, 1, 41),
        "joint_target_rad": (300, 1, 29),
        "torque_nm": (300, 1, 10, 29),
        "actual_actuator_force_nm": (300, 1, 10, 29),
        "pelvis_z_per_substep_m": (300, 1, 10),
        "force_n": (300, 1, 6),
        "ball_position_after_step_m": (300, 1, 3),
        "motor_delta_rad": (300, 1, 12),
        "pre_motor_joint_target_rad": (300, 1, 29),
    }
    with np.load(trace_path, allow_pickle=False) as loaded:
        trace = {k: loaded[k] for k in loaded.files}
    from rosclaw_soccer.rsi.foundation_observation_capture import (
        PREFIX,
        validate_capture,
        validate_measured_history,
    )

    if ("foundation_observation_capture" in report) != (
        "foundation_observation_capture" in commitment
    ):
        raise ValueError("CPU foundation capture differs from commitment")
    if "foundation_observation_capture" in commitment:
        validate_capture(commitment["foundation_observation_capture"], trace, frames=300, lanes=1)
        validate_measured_history(trace)
    elif any(name.startswith(PREFIX) for name in trace):
        raise ValueError("CPU foundation capture requires an explicit commitment")
    if any(
        trace[k].shape != shape or not np.isfinite(trace[k]).all() for k, shape in shapes.items()
    ):
        raise ValueError("complete finite CPU trace required")
    data = mujoco.MjData(model)
    data.qpos[:7] = trace["canonical_qpos"][0, 0, :7]
    data.qpos[qi] = trace["canonical_qpos"][0, 0, 7:36]
    data.qpos[bq : bq + 7] = trace["canonical_qpos"][0, 0, 36:43]
    data.qvel[:6] = trace["canonical_qvel"][0, 0, :6]
    data.qvel[vi] = trace["canonical_qvel"][0, 0, 6:35]
    data.qvel[bv : bv + 6] = trace["canonical_qvel"][0, 0, 35:41]
    mujoco.mj_forward(model, data)
    replay_forces = []
    minimum = float("inf")

    def equal(actual: Any, expected: Any, name: str) -> None:
        if not np.allclose(actual, expected, atol=1e-8, rtol=0):
            raise ValueError(f"CPU dynamics replay differs: {name}")

    observer = (
        CurrentKinematicObserver(model) if observation_snapshot == "current-kinematic" else None
    )
    for frame in range(300):
        if observer is None:
            positions, quaternions = data.xpos, data.xquat
            root_velocity = root_velocity_world(model, data, pelvis, reference)
            ball_velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, ball, ball_velocity, 0)
            ball_linear_velocity = ball_velocity[3:]
        else:
            current = observer.sample(data)
            positions, quaternions = current.body_position_m, current.body_quaternion_wxyz
            root_velocity = current.body_origin_velocity_world[pelvis]
            ball_linear_velocity = current.body_origin_velocity_world[ball, :3]
        qpos = np.concatenate((data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7]))
        qvel = np.concatenate((data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6]))
        equal(qpos, trace["canonical_qpos"][frame, 0], "qpos")
        equal(qvel, trace["canonical_qvel"][frame, 0], "qvel")
        equal(data.qpos[qi], trace["joint_position_rad"][frame, 0], "joint observation")
        equal(data.qvel[vi], trace["joint_velocity_rad_s"][frame, 0], "joint velocity")
        equal(
            np.concatenate((positions[pelvis], quaternions[pelvis][[1, 2, 3, 0]])),
            trace["root_pose_xyzw_m"][frame, 0],
            "root observation",
        )
        equal(positions[ball], trace["ball_position_before_step_m"][frame, 0], "ball observation")
        equal(
            positions[geometry],
            trace["foot_geometry_position_before_step_m"][frame, 0],
            "geometry observation",
        )
        equal(
            root_velocity,
            trace["root_velocity_world"][frame, 0],
            "root velocity",
        )
        equal(
            ball_linear_velocity,
            trace["ball_linear_velocity_before_step_m_s"][frame, 0],
            "ball velocity",
        )
        target = trace["joint_target_rad"][frame, 0]
        forces = np.zeros(6)
        for substep in range(10):
            torque = np.clip(
                (target - data.qpos[qi]) * kp - data.qvel[vi] * kd,
                -np.asarray(G1_HARD_TORQUE_LIMITS),
                np.asarray(G1_HARD_TORQUE_LIMITS),
            )
            equal(torque, trace["torque_nm"][frame, 0, substep], "PD torque")
            data.ctrl[ai] = torque
            mujoco.mj_step(model, data)
            equal(
                data.actuator_force[ai],
                trace["actual_actuator_force_nm"][frame, 0, substep],
                "actual actuator force",
            )
            equal(
                data.xpos[pelvis, 2], trace["pelvis_z_per_substep_m"][frame, 0, substep], "pelvis"
            )
            minimum = min(minimum, float(data.xpos[pelvis, 2]))
            for c in range(data.ncon):
                contact = data.contact[c]
                g1, g2 = int(contact.geom1), int(contact.geom2)
                if bg not in (g1, g2):
                    continue
                other_body = int(model.geom_bodyid[g2 if g1 == bg else g1])
                if other_body in contacts:
                    contact_force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, c, contact_force)
                    index = contacts.index(other_body)
                    forces[index] = max(forces[index], float(np.linalg.norm(contact_force[:3])))
        equal(forces, trace["force_n"][frame, 0], "physical contacts")
        equal(data.xpos[ball], trace["ball_position_after_step_m"][frame, 0], "physical ball")
        replay_forces.append(forces)
    actual_forces = np.asarray(replay_forces)
    contact_frames = np.flatnonzero(np.any(actual_forces > 1, axis=1))
    first = int(contact_frames[0]) if len(contact_frames) else None
    bodies = np.flatnonzero(actual_forces[first] > 1).tolist() if first is not None else []
    clean = bool(bodies and set(bodies) <= {0, 1} and not np.any(actual_forces[:, 2:] > 1))
    if (
        report["first_contact_frame"] != first
        or report["first_contact_bodies"] != bodies
        or report["clean_foot_only"] != clean
    ):
        raise ValueError("CPU contact outcome differs from actual dynamics")
    equal(minimum, report["minimum_pelvis_z_m"], "reported minimum pelvis")
    equal(data.xpos[ball], report["final_ball_position_m"], "reported final ball")
    if report.get("step_model_hash") is not None:
        from rosclaw_soccer.rsi.compiled_step_inference import CompiledStepMotor

        policy = report["executed_motor_policy"]
        if "proposal_sampling_motor_proof" in policy:
            from rosclaw_soccer.rsi.proposal_sampling_motor import CompiledProposalSamplingMotor

            proposal_decoder = (
                CompiledProposalSamplingMotor(policy)
                if sampling_decoder_factory is None
                else sampling_decoder_factory.bind(policy)
            )
            delta_at_frame = proposal_decoder.delta_at_frame
        elif "verified_success_imitation_motor_proof" in policy:
            from rosclaw_soccer.rsi.imitation_proposal_motor import CompiledImitationProposalMotor

            imitation_decoder = (
                CompiledImitationProposalMotor(policy)
                if mean_decoder_factory is None
                else mean_decoder_factory.new_episode()
            )
            delta_at_frame = imitation_decoder.delta_at_frame
        elif "extended_proposal_motor_proof" in policy:
            from rosclaw_soccer.rsi.extended_proposal_motor import CompiledExtendedProposalMotor

            extended_decoder = (
                CompiledExtendedProposalMotor(policy)
                if mean_decoder_factory is None
                else mean_decoder_factory.new_episode()
            )
            delta_at_frame = extended_decoder.delta_at_frame
        elif "proposal_memory_motor_proof" in policy:
            from rosclaw_soccer.rsi.proposal_memory_motor import CompiledProposalMemoryMotor

            mean_proposal_decoder = (
                CompiledProposalMemoryMotor(policy)
                if mean_decoder_factory is None
                else mean_decoder_factory.new_episode()
            )
            delta_at_frame = mean_proposal_decoder.delta_at_frame
        elif "advantage_memory_motor_proof" in policy:
            from rosclaw_soccer.rsi.advantage_memory_motor import CompiledAdvantageMemoryMotor

            delta_at_frame = CompiledAdvantageMemoryMotor(policy).delta_at_frame
        elif "current_memory_motor_proof" in policy:
            from rosclaw_soccer.rsi.current_memory_motor import CompiledCurrentMemoryMotor

            delta_at_frame = CompiledCurrentMemoryMotor(policy).delta_at_frame
        elif "consolidated_smooth_motor_proof" in policy:
            from rosclaw_soccer.rsi.consolidated_smooth_motor import (
                CompiledConsolidatedSmoothMotor,
            )

            delta_at_frame = CompiledConsolidatedSmoothMotor(policy).delta_at_frame
        elif "smooth_memory_motor_proof" in policy:
            from rosclaw_soccer.rsi.smooth_memory_motor import CompiledSmoothMemoryMotor

            decoder = (
                CompiledSmoothMemoryMotor(policy)
                if sampling_decoder_factory is None
                else sampling_decoder_factory.bind(policy)
            )
            delta_at_frame = decoder.delta_at_frame
        elif "output_memory_motor_proof" in policy:
            from rosclaw_soccer.rsi.output_memory_step_motor import CompiledOutputMemoryMotor

            delta_at_frame = CompiledOutputMemoryMotor(policy).delta_at_frame
        elif "replay_motor_proof" in policy:
            from rosclaw_soccer.rsi.kernel_replay_motor import CompiledReplayStepMotor

            delta_at_frame = CompiledReplayStepMotor(policy).delta_at_frame
        elif "selective_memory_motor_proof" in policy:
            from rosclaw_soccer.rsi.selective_phase_memory import CompiledSelectivePhaseMemory

            delta_at_frame = CompiledSelectivePhaseMemory(policy).delta_at_frame
        elif "memory_phase_motor_proof" in policy:
            from rosclaw_soccer.rsi.memory_guarded_phase_transfer import CompiledMemoryPhaseMotor

            delta_at_frame = CompiledMemoryPhaseMotor(policy).delta_at_frame
        elif "kernel_motor_proof" in policy:
            from rosclaw_soccer.rsi.kernel_guarded_step_execution import CompiledKernelStepMotor

            delta_at_frame = CompiledKernelStepMotor(policy).delta_at_frame
        elif "protected_phase_motor_proof" in policy:
            from rosclaw_soccer.rsi.protected_phase_step_execution import (
                CompiledProtectedPhaseMotor,
            )

            delta_at_frame = CompiledProtectedPhaseMotor(policy).delta_at_frame
        elif "compiled_motor_proof" in policy:
            delta_at_frame = CompiledStepMotor(policy).delta_at_frame
        else:
            delta_at_frame = CompiledStepMotor.from_legacy_preview(policy).delta_at_frame
        if policy["step_motor_proof"]["model"]["model_hash"] != report["step_model_hash"]:
            raise ValueError("CPU neural policy identity changed")
        ids = [names.index(n) for n in JOINT_NAMES]
        body_guidance = None
        if "body_response_guidance" in commitment:
            from rosclaw_soccer.rsi.body_response_guidance_execution import (
                BodyResponseGuidanceExecution,
            )

            if "proposal_memory_motor_proof" not in policy or ids != list(range(12)):
                raise ValueError("guidance requires the canonical proposal mean parent")
            body_guidance = BodyResponseGuidanceExecution(
                commitment["body_response_guidance"], policy
            )
            for key, width in (
                ("body_response_applied_increment", body_guidance.action_dimensions),
                ("body_response_status", 4),
            ):
                if (
                    key not in trace
                    or trace[key].shape != (300, 1, width)
                    or not np.isfinite(trace[key]).all()
                ):
                    raise ValueError("complete actual body guidance trace required")
        previous = np.zeros(12)
        for frame in range(300):
            nominal = trace["pre_motor_joint_target_rad"][frame, 0]
            delta = delta_at_frame(
                policy,
                trace,
                frame=frame,
                nominal_target=nominal,
                baseline=nominal[ids],
                limits=model.jnt_range[joints][ids],
                previous=previous,
                previous_contact_forces=trace["force_n"][frame - 1, 0] if frame else np.zeros(6),
            )
            if body_guidance is not None:
                delta, status = body_guidance.advance(
                    mean_proposal_decoder,
                    trace,
                    frame=frame,
                    nominal_target=nominal,
                    parent_delta=delta,
                    previous_final=previous,
                    limits=(
                        model.jnt_range[joints]
                        if body_guidance.action_dimensions == 29
                        else model.jnt_range[joints][ids]
                    ),
                )
                equal(
                    status["applied_increment"],
                    trace["body_response_applied_increment"][frame, 0],
                    "causal body guidance",
                )
                expected_status = np.asarray(
                    [
                        status["contact_phase"],
                        status["protected"],
                        status["active"],
                        status["fallback"],
                    ],
                    dtype=np.int64,
                )
                if not np.array_equal(expected_status, trace["body_response_status"][frame, 0]):
                    raise ValueError("causal body guidance status differs")
            equal(delta, trace["motor_delta_rad"][frame, 0], "causal neural output")
            composed = nominal.copy()
            if body_guidance is not None and body_guidance.action_dimensions == 29:
                composed += np.asarray(status["final_target_increment"])
            else:
                composed[ids] += delta
            equal(composed, trace["joint_target_rad"][frame, 0], "causal composed target")
            previous = delta
    displacement = post_contact_displacement(trace["ball_position_after_step_m"][:, 0], first)
    maximum_lateral = float(np.max(np.abs(trace["ball_position_after_step_m"][:, 0, 1])))
    quality = high_quality(
        dict(
            clean_foot_only=clean,
            minimum_pelvis_z_m=minimum,
            maximum_lateral_excursion_m=maximum_lateral,
            **displacement,
        )
    )
    result = dict(
        schema="soccer.rsi.cpu_motor_transfer_review.v1",
        reviewed_report_hash=report["report_hash"],
        actual_mujoco_dynamics_replayed=True,
        actual_pd_torque_reconstructed=True,
        neural_target_reconstructed=report.get("step_model_hash") is not None,
        physical_substeps=3000,
        first_contact_frame=first,
        contact_body_indices=np.flatnonzero(np.max(actual_forces, axis=0) > 1).tolist(),
        clean_foot_only=clean,
        minimum_pelvis_z_m=minimum,
        maximum_lateral_excursion_m=maximum_lateral,
        high_quality=quality,
        **displacement,
        safety_passed=minimum >= 0.65,
        qualification="CPU_DIAGNOSTIC_ONLY_NOT_PROMOTION",
        promotion_authorized=False,
        hardware_authorized=False,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
    )
    if mean_decoder_factory is not None:
        result["decoder_construction"] = dict(
            kind="VERIFIED_PRIVATE_FIXED_MEAN_FRESH_EPISODE",
            factory_contract=mean_decoder_factory.contract(),
            weights_or_action_law_changed=False,
            original_reference_constructor_used=False,
            physics_parity_requires_external_evidence=True,
        )
    if sampling_decoder_factory is not None:
        from rosclaw_soccer.rsi import (
            owned_proposal_sampling_factory,
            proposal_sampling_episode_factory,
        )
        from rosclaw_soccer.rsi import smooth_sampling_decoder_factory as factory_module

        owned = type(sampling_decoder_factory) is (
            owned_proposal_sampling_factory.OwnedProposalSamplingEpisodeFactory
        )
        factory_source = (
            owned_proposal_sampling_factory
            if owned
            else proposal_sampling_episode_factory
            if type(sampling_decoder_factory)
            is proposal_sampling_episode_factory.ProposalSamplingEpisodeFactory
            else factory_module
        )
        factory_source_path = factory_source.__file__
        if not isinstance(factory_source_path, str):
            raise ValueError("sampling factory source file unavailable")

        result["decoder_construction"] = {
            "kind": (
                "SOURCE_PINNED_ORIGINAL_NUMERIC_FRESH_EPISODE"
                if owned
                else "VERIFIED_SHARED_MEAN_WITH_INDEPENDENT_EPISODE_STATE"
            ),
            "source_hash": hash_bytes(Path(factory_source_path).read_bytes()),
            "complete_preview_validation_retained": not owned,
            "weights_or_action_law_changed": False,
        }
        if owned:
            result["decoder_construction"]["canonical_preview_integrity_checked"] = True
    result["report_hash"] = hash_json(result)
    return result
