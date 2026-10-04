"""Actual CPU MuJoCo closed-loop transfer probe, not an Isaac evidence replay.

No robot transport or vendor SDK is imported. Asset differences remain explicit;
this diagnostic cannot authorize promotion or physical execution.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.sonic_navigation import G1SonicNavigation, SonicNavigationConfig
from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch
from rosclaw_soccer.providers.g1.sonic_vector import BatchedSonicTracker
from rosclaw_soccer.rsi.contact_motor_contract import load_policy, motor_delta
from rosclaw_soccer.rsi.contact_time_phase_features import (
    current_context,
    gait_phase_features,
    predict_contact_time,
)
from rosclaw_soccer.rsi.first_touch_course_catalog import sample_training_courses
from rosclaw_soccer.rsi.late_swing_memory import load_late_swing_actor
from rosclaw_soccer.rsi.online_motor_actor_critic import configure_preview, validate_model
from rosclaw_soccer.rsi.sampling_model_io import load_sampling_model
from rosclaw_soccer.rsi.taskspace_gate_memory import select_taskspace_gate
from rosclaw_soccer.rsi.taskspace_swing_evidence import LEG_NAMES
from rosclaw_soccer.rsi.taskspace_swing_probe import (
    choose_swing_side,
    release_joint_delta,
    swing_joint_delta,
)
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.sim.current_kinematic_observation import (
    CurrentKinematicObserver,
)
from rosclaw_soccer.sim.current_kinematic_observation import (
    observation_contract as make_observation_contract,
)
from rosclaw_soccer.sim.root_velocity_reference import root_velocity_world
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation


def main(argv: list[str] | None = None, *, sampling_factory: Any = None) -> None:
    import mujoco
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "model-root", "late-swing-policy", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--neural-model", type=Path)
    group.add_argument("--motor-policy", type=Path)
    group.add_argument("--step-model", type=Path)
    parser.add_argument("--consumed-bank", type=Path)
    parser.add_argument("--foundation-only", action="store_true")
    parser.add_argument("--compressed-report", action="store_true")
    parser.add_argument(
        "--record-foundation-observation",
        action="store_true",
        help="Record actual 994-input/29-action frozen foundation calls; not a trained policy",
    )
    parser.add_argument(
        "--shared-evidence",
        action="store_true",
        help="Opt-in lossless shared model/world storage for new compressed evidence only",
    )
    parser.add_argument(
        "--proposal-decoder",
        choices=("reference", "owned_snapshot", "bounded_snapshot"),
        default="reference",
        help="Opt-in owned numerical compilation; only sealed proposal models are accepted",
    )
    parser.add_argument(
        "--root-velocity-reference",
        choices=("body-com", "body-origin"),
        default="body-com",
        help="Explicit diagnostic input convention; default preserves historical COM observations",
    )
    parser.add_argument(
        "--observation-snapshot", choices=("cached", "current-kinematic"), default="cached"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lane", type=int, required=True, choices=range(16))
    args = parser.parse_args(argv)
    make_observation_contract(args.root_velocity_reference, args.observation_snapshot)
    if args.foundation_only and (args.neural_model or args.motor_policy or args.step_model):
        parser.error("foundation-only cannot include a motor learning model")
    if args.proposal_decoder != "reference" and args.step_model is None:
        parser.error("owned proposal decoder requires an explicit sealed step-model")
    if args.shared_evidence and not args.compressed_report:
        parser.error("shared evidence requires compressed-report")
    if sampling_factory is not None and (
        args.step_model is None or args.proposal_decoder != "reference"
    ):
        parser.error("shared smooth factory requires only an explicit reference sampling model")
    # This first diagnostic only accepts already consumed courses.
    consumed = {
        (20261177, 0),
        (20261227, 4),
        (20261282, 0),
        (20261360, 0),
        (20261378, 0),
        (20261440, 2),
        (20261446, 0),
        (20261095, 2),
        (20261097, 2),
        (20261146, 4),
        (20261148, 2),
        (20260975, 4),
    }
    consumed_bank_hash = None
    if args.consumed_bank is not None:
        # Artifact inspection must not import historical experiment scripts.
        bank = json.loads(args.consumed_bank.read_text())
        if (
            bank.get("report_hash")
            != hash_json({k: v for k, v in bank.items() if k != "report_hash"})
            or bank.get("schema") != "soccer.rsi.progressive_motor_learning_bank.v312"
            or bank.get("partition") != "TRAIN_CONSUMED"
            or bank.get("fresh_holdout_open_authorized") is not False
        ):
            parser.error("sealed consumed physical bank required for additional transfer cases")
        consumed |= {(r["seed"], r["lane"]) for r in bank["courses"]}
        consumed_bank_hash = bank["report_hash"]
    if (args.seed, args.lane) not in consumed:
        parser.error("transfer diagnostic cannot open a fresh course")
    torch.set_num_threads(1)
    scene = args.scene.resolve()
    spec = mujoco.MjSpec.from_file(str(scene))
    model = spec.compile()
    native_ball_added = False
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball") < 0:
        native_ball_added = True
        body = spec.worldbody.add_body(name="ball", pos=(2.5, 0.0, 0.13))
        body.add_freejoint(name="ball_free")
        body.add_geom(
            name="ball_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.11, 0.0, 0.0),
            mass=0.43,
            friction=(0.6, 0.005, 0.0001),
        )
        if not np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE):
            spec.worldbody.add_geom(
                name="transfer_floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=(0.0, 0.0, 0.01),
                friction=(1.0, 0.005, 0.0001),
            )
        model = spec.compile()
    data = mujoco.MjData(model)
    model.opt.timestep = 0.002

    def identifier(kind: Any, name: str) -> int:
        value = int(mujoco.mj_name2id(model, kind, name))
        if value < 0:
            raise ValueError(f"missing simulation element {name}")
        return value

    names = list(G1_DDS_JOINT_NAMES)
    joints = [identifier(mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    qi = np.asarray([model.jnt_qposadr[j] for j in joints], dtype=int)
    vi = np.asarray([model.jnt_dofadr[j] for j in joints], dtype=int)
    actuator_ids = []
    for joint in joints:
        matching = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
        if len(matching) != 1:
            raise ValueError("exactly one simulated actuator per canonical joint required")
        actuator_ids.append(int(matching[0]))
    ai = np.asarray(actuator_ids, dtype=int)
    if model.nu != 29 or not np.array_equal(model.actuator_trnid[ai, 0], joints):
        raise ValueError("canonical named 29-joint torque contract required")
    pelvis = identifier(mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    body_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_knee_link",
        "right_knee_link",
    ]
    bodies = [identifier(mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names]
    contact_bodies = (
        bodies[:2]
        + [
            identifier(mujoco.mjtObj.mjOBJ_BODY, "left_ankle_pitch_link"),
            identifier(mujoco.mjtObj.mjOBJ_BODY, "right_ankle_pitch_link"),
        ]
        + bodies[2:]
    )
    ball = identifier(mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_geom = identifier(mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    bj = int(model.body_jntadr[ball])
    bq, bv = int(model.jnt_qposadr[bj]), int(model.jnt_dofadr[bj])
    # Explicit physics adaptation, not an assertion that the XML equals USD.
    model.geom_size[ball_geom, 0] = 0.11
    model.body_mass[ball] = 0.43
    model.body_inertia[ball] = 0.4 * 0.43 * 0.11**2
    model.dof_damping[bv : bv + 6] = 0
    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    if box >= 0:
        mask = model.geom_bodyid == box
        model.geom_contype[mask] = 0
        model.geom_conaffinity[mask] = 0
    mujoco.mj_setConst(model, data)
    navigation = G1SonicNavigation(
        args.model_root,
        "cpu.first_touch.0",
        SonicNavigationConfig(
            maximum_frames=300,
            planner_seed=30300,
            model_variant="low_latency",
            experimental_maximum_speed_mps=1.5,
            inference_threads=1,
        ),
    )
    navigation.backend.qualification.require_eligible()
    foundation = FrozenSonicG1Torch(args.model_root, variant="low_latency", device="cpu")
    late = load_late_swing_actor(args.late_swing_policy)
    neural = json.loads(args.neural_model.read_text()) if args.neural_model else None
    if neural:
        validate_model(neural)
    policy, knots = load_policy(args.motor_policy) if args.motor_policy else (None, None)
    # Restore the WHOLE original numerical model before preview validation.
    # Ordinary JSON/gzip policies remain unchanged; no distribution is changed.
    step_model = load_sampling_model(args.step_model) if args.step_model else None
    sampling_compilation = None
    if sampling_factory is not None:
        from rosclaw_soccer.rsi.owned_proposal_sampling_factory import (
            OwnedProposalSamplingEpisodeFactory,
        )
        from rosclaw_soccer.rsi.proposal_sampling_episode_factory import (
            ProposalSamplingEpisodeFactory,
        )

        expected_sampling_schema = (
            "soccer.rsi.proposal_memory_sampling.v1"
            if type(sampling_factory)
            in (ProposalSamplingEpisodeFactory, OwnedProposalSamplingEpisodeFactory)
            else "soccer.rsi.smooth_memory_sampling.v1"
        )
        if step_model is None or step_model.get("schema") != expected_sampling_schema:
            raise ValueError("shared sampling factory cannot compile other motor families")
    if args.proposal_decoder != "reference" and (
        step_model is None or step_model.get("schema") != "soccer.rsi.proposal_memory_motor.v1"
    ):
        raise ValueError("owned proposal decoder only accepts the sealed proposal family")
    if step_model is not None:
        from rosclaw_soccer.rsi.step_motor_execution import delta_at_frame
        from rosclaw_soccer.rsi.step_motor_execution import make_preview as legacy_preview

        make_preview: Callable[[dict[str, Any]], dict[str, Any]] = legacy_preview

        if step_model.get("schema") == "soccer.rsi.proposal_memory_sampling.v1":
            from rosclaw_soccer.rsi.owned_proposal_sampling_factory import (
                OwnedProposalSamplingEpisodeFactory,
            )
            from rosclaw_soccer.rsi.proposal_sampling_motor import CompiledProposalSamplingMotor
            from rosclaw_soccer.rsi.proposal_sampling_motor import make_preview as sampling_preview

            make_preview = sampling_preview
            prepared_proposal_policy = (
                sampling_factory.preview(step_model)
                if type(sampling_factory) is OwnedProposalSamplingEpisodeFactory
                else make_preview(step_model)
            )
            if sampling_factory is None:
                proposal_sampling_decoder = CompiledProposalSamplingMotor(prepared_proposal_policy)
            else:
                from rosclaw_soccer.rsi.proposal_sampling_episode_factory import (
                    ProposalSamplingEpisodeFactory,
                )
                from rosclaw_soccer.rsi.proposal_sampling_episode_factory import (
                    compilation_contract as proposal_sampling_contract,
                )

                if type(sampling_factory) not in (
                    ProposalSamplingEpisodeFactory,
                    OwnedProposalSamplingEpisodeFactory,
                ):
                    raise ValueError("only the fixed proposal sampling factory is accepted")
                proposal_sampling_decoder = sampling_factory.bind(prepared_proposal_policy)
                if type(sampling_factory) is OwnedProposalSamplingEpisodeFactory:
                    from rosclaw_soccer.rsi.owned_proposal_sampling_factory import (
                        compilation_contract as owned_proposal_sampling_contract,
                    )

                    sampling_compilation = owned_proposal_sampling_contract(
                        prepared_proposal_policy
                    )
                else:
                    sampling_compilation = proposal_sampling_contract(prepared_proposal_policy)
            delta_at_frame = proposal_sampling_decoder.delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.proposal_memory_motor.v1":
            from rosclaw_soccer.rsi.proposal_decoder_selection import select_proposal_decoder
            from rosclaw_soccer.rsi.proposal_memory_motor import make_preview as proposal_preview

            make_preview = proposal_preview
            policy = make_preview(step_model)
            delta_at_frame = select_proposal_decoder(
                policy, implementation=args.proposal_decoder
            ).delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.advantage_memory_motor.v1":
            from rosclaw_soccer.rsi.advantage_memory_motor import CompiledAdvantageMemoryMotor
            from rosclaw_soccer.rsi.advantage_memory_motor import make_preview as advantage_preview

            make_preview = advantage_preview
            delta_at_frame = CompiledAdvantageMemoryMotor(make_preview(step_model)).delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.current_memory_guarded_motor.v1":
            from rosclaw_soccer.rsi.current_memory_motor import CompiledCurrentMemoryMotor
            from rosclaw_soccer.rsi.current_memory_motor import make_preview as current_preview

            make_preview = current_preview
            delta_at_frame = CompiledCurrentMemoryMotor(make_preview(step_model)).delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.consolidated_smooth_motor.v1":
            from rosclaw_soccer.rsi.consolidated_smooth_motor import CompiledConsolidatedSmoothMotor
            from rosclaw_soccer.rsi.consolidated_smooth_motor import (
                make_preview as consolidated_preview,
            )

            make_preview = consolidated_preview
            delta_at_frame = CompiledConsolidatedSmoothMotor(
                make_preview(step_model)
            ).delta_at_frame
        elif step_model.get("schema") in (
            "soccer.rsi.smooth_memory_motor.v1",
            "soccer.rsi.smooth_memory_sampling.v1",
        ):
            from rosclaw_soccer.rsi.owned_smooth_sampling_factory import OwnedSmoothSamplingFactory
            from rosclaw_soccer.rsi.smooth_decoder_selection import select_smooth_decoder
            from rosclaw_soccer.rsi.smooth_memory_motor import make_preview as smooth_preview

            make_preview = smooth_preview
            prepared_smooth_policy = (
                sampling_factory.preview(step_model)
                if type(sampling_factory) is OwnedSmoothSamplingFactory
                else make_preview(step_model)
            )
            decoder, sampling_compilation = select_smooth_decoder(
                prepared_smooth_policy, sampling_factory=sampling_factory
            )
            delta_at_frame = decoder.delta_at_frame
        elif step_model.get("schema") in (
            "soccer.rsi.output_memory_step_motor.v1",
            "soccer.rsi.output_memory_step_sampling.v1",
        ):
            from rosclaw_soccer.rsi.output_memory_step_motor import CompiledOutputMemoryMotor
            from rosclaw_soccer.rsi.output_memory_step_motor import make_preview as output_preview

            make_preview = output_preview
            delta_at_frame = CompiledOutputMemoryMotor(make_preview(step_model)).delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.kernel_replay_motor.v1":
            from rosclaw_soccer.rsi.kernel_replay_motor import CompiledReplayStepMotor
            from rosclaw_soccer.rsi.kernel_replay_motor import make_preview as replay_preview

            make_preview = replay_preview
            delta_at_frame = CompiledReplayStepMotor(make_preview(step_model)).delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.selective_phase_memory.v1":
            from rosclaw_soccer.rsi.selective_phase_memory import CompiledSelectivePhaseMemory
            from rosclaw_soccer.rsi.selective_phase_memory import make_preview as selective_preview

            make_preview = selective_preview
            delta_at_frame = CompiledSelectivePhaseMemory(make_preview(step_model)).delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.memory_guarded_phase_transfer.v1":
            from rosclaw_soccer.rsi.memory_guarded_phase_transfer import CompiledMemoryPhaseMotor
            from rosclaw_soccer.rsi.memory_guarded_phase_transfer import (
                make_preview as memory_preview,
            )

            make_preview = memory_preview
            delta_at_frame = CompiledMemoryPhaseMotor(make_preview(step_model)).delta_at_frame
        elif step_model.get("schema") in (
            "soccer.rsi.kernel_guarded_step_actor_critic.v1",
            "soccer.rsi.kernel_guarded_step_sampling.v1",
        ):
            from rosclaw_soccer.rsi.kernel_guarded_step_execution import (
                CompiledKernelStepMotor,
            )
            from rosclaw_soccer.rsi.kernel_guarded_step_execution import (
                make_preview as kernel_preview,
            )

            make_preview = kernel_preview
            delta_at_frame = CompiledKernelStepMotor(make_preview(step_model)).delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.protected_phase_step_actor_critic.v1":
            from rosclaw_soccer.rsi.protected_phase_step_execution import (
                CompiledProtectedPhaseMotor,
            )
            from rosclaw_soccer.rsi.protected_phase_step_execution import (
                make_preview as protected_phase_preview,
            )

            make_preview = protected_phase_preview
            delta_at_frame = CompiledProtectedPhaseMotor(make_preview(step_model)).delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.compiled_step_motor_decoder.v1":
            from rosclaw_soccer.rsi.compiled_step_inference import CompiledStepMotor
            from rosclaw_soccer.rsi.compiled_step_inference import make_preview as compiled_preview

            make_preview = compiled_preview
            delta_at_frame = CompiledStepMotor(make_preview(step_model)).delta_at_frame
        elif step_model.get("schema") == "soccer.rsi.online_step_motor_mc_ppo.v1":
            from rosclaw_soccer.rsi.online_step_execution import (
                delta_at_frame as online_delta_at_frame,
            )
            from rosclaw_soccer.rsi.online_step_execution import make_preview as online_preview

            delta_at_frame = online_delta_at_frame
            make_preview = online_preview
        if step_model.get("schema") in (
            "soccer.rsi.smooth_memory_motor.v1",
            "soccer.rsi.smooth_memory_sampling.v1",
        ):
            policy = prepared_smooth_policy
        elif step_model.get("schema") != "soccer.rsi.proposal_memory_motor.v1":
            policy = make_preview(step_model)
    x, y, vx = sample_training_courses(args.seed, 16)[args.lane]
    data.qpos[:7] = (0, 0, 0.793, 1, 0, 0, 0)
    data.qpos[qi] = navigation.backend.default_angles
    data.qpos[bq : bq + 7] = (x, y, 0.13, 1, 0, 0, 0)
    data.qvel[bv : bv + 6] = (vx, 0, 0, 0, vx / 0.11, 0)
    mujoco.mj_forward(model, data)
    args.output_root.mkdir(parents=True, exist_ok=False)
    model_snapshot = args.output_root / "compiled_model.mjb"
    if args.shared_evidence:
        from rosclaw_soccer.rsi.shared_cpu_evidence import save_shared_compiled_model

        save_shared_compiled_model(model, model_snapshot)
    else:
        mujoco.mj_saveModel(model, str(model_snapshot), None)
    assets = {
        str(p.relative_to(scene.parent)): hash_bytes(p.read_bytes())
        for p in sorted(scene.parent.rglob("*"))
        if p.is_file() and p.suffix.lower() in {".xml", ".stl", ".obj", ".png", ".jpg"}
    }
    commitment = dict(
        schema="soccer.rsi.cpu_motor_transfer.v1",
        seed=args.seed,
        lane=args.lane,
        course=[x, y, vx],
        partition="CONSUMED_TRANSFER_DIAGNOSTIC",
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        assets=assets,
        foundation_hash=navigation.backend.qualification.qualification_hash,
        late_actor_hash=late["actor_hash"],
        execution_profile="foundation_only" if args.foundation_only else "taskspace_plus_motor",
        model_hash=neural["model_hash"] if neural else None,
        step_model_hash=step_model["model_hash"] if step_model else None,
        consumed_bank_hash=consumed_bank_hash,
        motor_policy_hash=policy["policy_hash"] if policy else None,
        taskspace_contract=dict(
            forward_cap_m=0.08,
            lateral_cap_m=0.15,
            acquisition_max_gap_m=0.95,
            revalidate_swing_side=True,
            vertical_offset_m=0.04,
        ),
        compiled_model_hash=hash_bytes(model_snapshot.read_bytes()),
        physics=dict(
            engine="MuJoCo",
            version=mujoco.__version__,
            device="cpu",
            dt_s=0.002,
            control_dt_s=0.02,
            ball_radius_m=0.11,
            ball_mass_kg=0.43,
            ball_free_damping=0.0,
            auxiliary_box_collision_disabled=True,
            robot_xml_not_usd=True,
            contact_settings="original XML pairs retained",
            native_ball_added=native_ball_added,
            canonical_joint_names=names,
            canonical_pd_kp=navigation.backend.kp.tolist(),
            canonical_pd_kd=navigation.backend.kd.tolist(),
        ),
        activation_ceiling="SIM_ONLY",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    observation_contract = make_observation_contract(
        args.root_velocity_reference, args.observation_snapshot
    )
    if args.proposal_decoder != "reference":
        from rosclaw_soccer.rsi.proposal_decoder_selection import compilation_contract

        commitment["numeric_compilation"] = compilation_contract(args.proposal_decoder)
    if sampling_compilation is not None:
        commitment["numeric_sampling_compilation"] = sampling_compilation
    if observation_contract is not None:
        commitment["observation_contract"] = observation_contract
    if args.record_foundation_observation:
        from rosclaw_soccer.rsi.foundation_observation_capture import capture_contract

        commitment["foundation_observation_capture"] = capture_contract()
    (args.output_root / "commitment.json").write_text(json.dumps(commitment, indent=2))
    history: dict[str, list[Any]] = {
        k: []
        for k in (
            "root_pose_xyzw_m",
            "root_velocity_world",
            "ball_position_before_step_m",
            "ball_linear_velocity_before_step_m_s",
            "foot_geometry_position_before_step_m",
            "joint_position_rad",
            "joint_velocity_rad_s",
            "joint_target_rad",
            "force_n",
            "ball_position_after_step_m",
            "torque_nm",
            "pre_motor_joint_target_rad",
            "motor_delta_rad",
            "navigation_command",
            "actual_actuator_force_nm",
            "pelvis_z_per_substep_m",
            "canonical_qpos",
            "canonical_qvel",
        )
    }
    if args.record_foundation_observation:
        from rosclaw_soccer.rsi.foundation_observation_capture import FIELDS, PREFIX

        history.update({PREFIX + name: [] for name in FIELDS})
    tracker = None
    tracking = y < 0
    gate = False
    side, contact_frame = -1, None
    swing_contact_delta = np.zeros(6)
    previous, motor_contact_delta = np.zeros(12), np.zeros(12)
    leg_ids = [np.asarray([names.index(n) for n in row]) for row in LEG_NAMES]
    motor_ids = np.concatenate(leg_ids)
    limits = model.jnt_range[joints].copy()
    minimum_pelvis = float("inf")
    first_contact_frame = None
    first_contact_bodies: list[int] = []
    observer = (
        CurrentKinematicObserver(model)
        if args.observation_snapshot == "current-kinematic"
        else None
    )
    for frame in range(300):
        if observer is None:
            positions, quaternions = data.xpos, data.xquat
            root_vel = root_velocity_world(model, data, pelvis, args.root_velocity_reference)
            ball_vel = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, ball, ball_vel, 0)
            ball_linear_vel = ball_vel[3:]
        else:
            current = observer.sample(data)
            positions, quaternions = current.body_position_m, current.body_quaternion_wxyz
            root_vel = current.body_origin_velocity_world[pelvis]
            ball_linear_vel = current.body_origin_velocity_world[ball, :3]
        root_pose = np.concatenate((positions[pelvis], quaternions[pelvis][[1, 2, 3, 0]]))
        for key, value in (
            ("root_pose_xyzw_m", root_pose),
            ("root_velocity_world", root_vel),
            ("ball_position_before_step_m", positions[ball]),
            ("ball_linear_velocity_before_step_m_s", ball_linear_vel),
            ("foot_geometry_position_before_step_m", positions[bodies]),
            ("joint_position_rad", data.qpos[qi]),
            ("joint_velocity_rad_s", data.qvel[vi]),
        ):
            history[key].append(value.copy()[None])
        gap = positions[ball] - positions[pelvis]
        lateral = (
            float(np.clip(1.2 * gap[1], -0.2, 0.2))
            if tracking and gap[0] > 0.95 and contact_frame is None
            else 0.0
        )
        qpos = np.concatenate((data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7]))
        qvel = np.concatenate((data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6]))
        history["canonical_qpos"].append(qpos.copy()[None])
        history["canonical_qvel"].append(qvel.copy()[None])
        observation = TeamMotorObservation(
            agent_id=navigation.agent_id,
            frame=frame,
            time_sec=frame * 0.02,
            intent="other",
            prospective_owner=False,
            qpos=tuple(float(v) for v in qpos),
            qvel=tuple(float(v) for v in qvel),
            target_position_m=(0.0, 0.0, 0.0),
            navigation_command=(1.4, lateral, 0.0),
            navigation_envelope=navigation.navigation_envelope,
        )
        if frame == 0:
            navigation.start_from_observation(observation)
        navigation.prepare_batched_proposal(observation)
        if tracker is None:
            tracker = BatchedSonicTracker(
                foundation,
                navigation.backend.reference[None],
                low_latency_legacy_encoder_layout=True,
                capture_neural_observation=args.record_foundation_observation,
            )
            tracker.reset(qpos[None], qvel[None])
        else:
            tracker.observe(qpos[None], qvel[None])
            if frame % 20 == 0:
                tracker.refresh_unexecuted_reference(
                    frame, navigation.backend.reference[None], unchanged_lookahead_frames=10
                )
        tracker.update(frame, qpos[None], qvel[None])
        if args.record_foundation_observation:
            for name, value in tracker.neural_observation().items():
                history[PREFIX + name].append(value.cpu().numpy().copy())
        proposal = navigation.commit_batched_action(tracker.action[0].numpy())
        target = np.asarray(proposal.target_rad).copy()
        if frame == 30:
            raw = current_context(
                root_pose[None], root_vel[None], positions[ball][None], ball_linear_vel[None]
            )
            features = gait_phase_features(
                raw, predict_contact_time(raw, np.asarray(late["contact_time_weights"]))
            )
            gate = bool(
                select_taskspace_gate(
                    features,
                    np.asarray(late["memory_features"]),
                    np.asarray(late["memory_clean"]),
                    np.asarray(late["memory_reward"]),
                    np.asarray(late["memory_groups"]),
                    neighbors=late["neighbors"],
                    confidence=late["confidence"],
                    baseline_clean_ceiling=late["baseline_clean_ceiling"],
                )[0]
                and vx < 0
                and not args.foundation_only
            )
            if neural:
                policy = configure_preview(neural, history, contact_frame)
                knots = np.asarray(policy["knots_rad"])
        swing_delta = np.zeros(6)
        if gate:
            if contact_frame is None:
                side = choose_swing_side(
                    positions[bodies[:2]],
                    positions[ball],
                    side,
                    acquisition_max_gap_m=0.95,
                    revalidate_swing_side=True,
                )
            if side >= 0:
                ids = leg_ids[side]
                if contact_frame is not None:
                    swing_delta = release_joint_delta(swing_contact_delta, frame - contact_frame)
                else:
                    jac = np.zeros((3, model.nv))
                    mujoco.mj_jacBody(model, data, jac, None, bodies[side])
                    swing_delta = swing_joint_delta(
                        positions[bodies[side]],
                        positions[ball],
                        jac[:, vi[ids]],
                        target[ids],
                        limits[ids],
                        forward_cap_m=0.08,
                        lateral_cap_m=0.15,
                        lateral_lead_m=0.0,
                        vertical_offset_m=0.04,
                    )
                target[ids] += swing_delta
        history["pre_motor_joint_target_rad"].append(target.copy()[None])
        if step_model is not None:
            if policy is None:
                raise ValueError("per-frame actor requires a sealed preview")
            previous = delta_at_frame(
                policy,
                history,
                frame=frame,
                nominal_target=target,
                baseline=target[motor_ids],
                limits=limits[motor_ids],
                previous=previous,
                previous_contact_forces=history["force_n"][frame - 1][0] if frame else np.zeros(6),
            )
            target[motor_ids] += previous
        elif policy is not None:
            if knots is None:
                raise ValueError("motor policy requires authenticated knots")
            previous = motor_delta(
                policy,
                knots,
                float(gap[0]),
                target[motor_ids],
                limits[motor_ids],
                previous,
                motor_contact_delta,
                frame - contact_frame if contact_frame is not None else None,
            )
            target[motor_ids] += previous
        history["motor_delta_rad"].append(previous.copy()[None])
        history["navigation_command"].append(np.asarray((1.4, lateral, 0.0))[None])
        history["joint_target_rad"].append(target[None].copy())
        forces = np.zeros(6)
        torques = []
        actual_forces = []
        pelvis_samples = []
        for _ in range(10):
            torque = np.clip(
                (target - data.qpos[qi]) * navigation.backend.kp
                - data.qvel[vi] * navigation.backend.kd,
                -np.asarray(G1_HARD_TORQUE_LIMITS),
                np.asarray(G1_HARD_TORQUE_LIMITS),
            )
            data.ctrl[ai] = torque
            torques.append(torque.copy())
            mujoco.mj_step(model, data)
            actual_forces.append(data.actuator_force[ai].copy())
            for c in range(data.ncon):
                contact = data.contact[c]
                g1, g2 = int(contact.geom1), int(contact.geom2)
                if ball_geom not in (g1, g2):
                    continue
                other = g2 if g1 == ball_geom else g1
                body = int(model.geom_bodyid[other])
                if body not in contact_bodies:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, c, force)
                index = contact_bodies.index(body)
                forces[index] = max(forces[index], float(np.linalg.norm(force[:3])))
            minimum_pelvis = min(minimum_pelvis, float(data.xpos[pelvis, 2]))
            pelvis_samples.append(float(data.xpos[pelvis, 2]))
        if contact_frame is None and np.any(forces > 1):
            contact_frame = frame
            first_contact_frame = frame
            first_contact_bodies = np.flatnonzero(forces > 1).tolist()
            swing_contact_delta = swing_delta.copy()
            motor_contact_delta = previous.copy()
        history["force_n"].append(forces[None])
        history["ball_position_after_step_m"].append(data.xpos[ball].copy()[None])
        history["torque_nm"].append(np.asarray(torques)[None])
        history["actual_actuator_force_nm"].append(np.asarray(actual_forces)[None])
        history["pelvis_z_per_substep_m"].append(np.asarray(pelvis_samples)[None])
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("nonfinite physical MuJoCo state")
    trace = args.output_root / "physical_trace.npz"
    arrays: dict[str, Any] = {k: np.asarray(v) for k, v in history.items()}
    if args.record_foundation_observation:
        from rosclaw_soccer.rsi.foundation_observation_capture import validate_capture

        validate_capture(commitment["foundation_observation_capture"], arrays, frames=300, lanes=1)
    np.savez_compressed(trace, **arrays)
    result = dict(
        commitment,
        commitment_hash=hash_json(commitment),
        physical_trace_hash=hash_bytes(trace.read_bytes()),
        first_contact_frame=first_contact_frame,
        first_contact_bodies=first_contact_bodies,
        clean_foot_only=bool(
            first_contact_bodies
            and all(i < 2 for i in first_contact_bodies)
            and not np.any(np.asarray(history["force_n"])[:, :, 2:] > 1)
        ),
        minimum_pelvis_z_m=minimum_pelvis,
        final_ball_position_m=data.xpos[ball].tolist(),
        taskspace_gate_selected=gate,
        executed_motor_policy=policy,
        qualification="UNQUALIFIED_CPU_TRANSFER_DIAGNOSTIC",
    )
    result["report_hash"] = hash_json(result)
    from scripts.rsi_atomic_artifacts import write_once, write_shared_physical_report

    report_name = "report.json.gz" if args.compressed_report else "report.json"
    if args.shared_evidence:
        write_shared_physical_report(args.output_root / report_name, result)
    else:
        write_once(args.output_root / report_name, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "report_hash",
                    "first_contact_frame",
                    "clean_foot_only",
                    "minimum_pelvis_z_m",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
