"""Reference-free direct-motor physics exam for the S52 recovery student."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.opentrack_recovery_bridge_exam import (
    OpenTrackRecoveryBridgeExamConfig,
    _atomic_json,
    _file_hash,
)
from rosclaw_soccer.training.opentrack_recovery_bridge_holdout import (
    _verified_development_report,
    _wilson_lower_bound,
)
from rosclaw_soccer.training.opentrack_recovery_student_collect import (
    _teacher_body_hash,
    _verified_holdout_report,
)
from rosclaw_soccer.training.recovery_snapshot import (
    RecoverySnapshot,
    load_recovery_snapshot_corpus,
)
from rosclaw_soccer.training.recovery_student import (
    RecoveryDistillationCorpus,
    build_recovery_proprioception,
    denormalize_absolute_motor_targets,
    load_recovery_distillation_corpus,
)
from rosclaw_soccer.training.recovery_teacher_bridge import (
    RecoveryPerturbationConfig,
    body_gravity_vector,
    build_recovery_perturbation_holdout,
)


@dataclass(frozen=True)
class RecoveryStudentPhysicsTrial:
    initial_snapshot_hash: str
    base_snapshot_hash: str
    suite: Literal["DEVELOPMENT_BASE", "SEALED_LOCAL_HOLDOUT"]
    student_onnx_hash: str
    succeeded: bool
    finite_state: bool
    ready_handoff_triggered: bool
    executed_sec: float
    final_stable_sec: float
    final_pelvis_height_m: float
    final_upright_projection: float
    final_root_linear_speed_mps: float
    final_root_angular_speed_rad_s: float
    peak_root_angular_speed_rad_s: float
    minimum_pelvis_height_m: float
    maximum_target_delta_rad: float
    torque_saturation_fraction: float
    joint_limit_clip_fraction: float
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.recovery_student_physics_trial.v1"

    def __post_init__(self) -> None:
        hashes = (
            self.initial_snapshot_hash,
            self.base_snapshot_hash,
            self.student_onnx_hash,
        )
        scalars = (
            self.executed_sec,
            self.final_stable_sec,
            self.final_pelvis_height_m,
            self.final_upright_projection,
            self.final_root_linear_speed_mps,
            self.final_root_angular_speed_rad_s,
            self.peak_root_angular_speed_rad_s,
            self.minimum_pelvis_height_m,
            self.maximum_target_delta_rad,
            self.torque_saturation_fraction,
            self.joint_limit_clip_fraction,
        )
        if (
            any(not value.startswith("sha256:") or len(value) != 71 for value in hashes)
            or self.suite not in {"DEVELOPMENT_BASE", "SEALED_LOCAL_HOLDOUT"}
            or any(not math.isfinite(value) for value in scalars)
            or min(
                self.executed_sec,
                self.final_stable_sec,
                self.final_pelvis_height_m,
                self.final_root_linear_speed_mps,
                self.final_root_angular_speed_rad_s,
                self.peak_root_angular_speed_rad_s,
                self.minimum_pelvis_height_m,
                self.maximum_target_delta_rad,
                self.torque_saturation_fraction,
                self.joint_limit_clip_fraction,
            )
            < 0.0
            or not 0.0 <= self.torque_saturation_fraction <= 1.0
            or not 0.0 <= self.joint_limit_clip_fraction <= 1.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_command_sent
        ):
            raise ValueError("recovery student physics trial is invalid")

    @property
    def trial_hash(self) -> str:
        return str(hash_json(asdict(self)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _verified_json(path: Path, *, schema_version: str, hash_key: str) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery student evidence must be an object")
    declared = payload.pop(hash_key, None)
    if payload.get("schema_version") != schema_version or declared != hash_json(payload):
        raise ValueError("recovery student evidence integrity failed")
    payload[hash_key] = declared
    return payload


def _run_student_trial(
    *,
    env: Any,
    session: Any,
    snapshot: RecoverySnapshot,
    base_snapshot_hash: str,
    suite: Literal["DEVELOPMENT_BASE", "SEALED_LOCAL_HOLDOUT"],
    student_onnx_hash: str,
    corpus: RecoveryDistillationCorpus,
    exam_config: OpenTrackRecoveryBridgeExamConfig,
    constants: Any,
    mujoco: Any,
    student_memory_size: int | None = None,
    absolute_target_sequence: NDArray[np.float32] | None = None,
    absolute_target_provider: Callable[
        [int, NDArray[np.float32]], NDArray[np.float32]
    ]
    | None = None,
    preserve_sequence_target_authority: bool = False,
    motor_target_lower_rad: NDArray[np.float32] | None = None,
    motor_target_upper_rad: NDArray[np.float32] | None = None,
    preserve_model_target_authority: bool = False,
) -> tuple[RecoveryStudentPhysicsTrial, list[dict[str, Any]]]:
    """Run direct PD control without env.step or any trajectory-handler read."""

    if absolute_target_sequence is not None and absolute_target_provider is not None:
        raise ValueError("recovery student accepts only one absolute-target source")
    env.reset()
    target_lower = (
        corpus.joint_lower_rad
        if motor_target_lower_rad is None
        else np.asarray(motor_target_lower_rad, dtype=np.float32)
    )
    target_upper = (
        corpus.joint_upper_rad
        if motor_target_upper_rad is None
        else np.asarray(motor_target_upper_rad, dtype=np.float32)
    )
    if (
        target_lower.shape != (29,)
        or target_upper.shape != (29,)
        or not np.all(np.isfinite(target_lower))
        or not np.all(np.isfinite(target_upper))
        or np.any(target_upper <= target_lower)
    ):
        raise ValueError("recovery student motor target authority is invalid")
    qpos = np.asarray(snapshot.qpos, dtype=np.float64).copy()
    qvel = np.asarray(snapshot.qvel, dtype=np.float64).copy()
    qpos[:2] = np.asarray(env.mj_data.qpos[:2], dtype=np.float64)
    env.mj_data.qpos[:] = qpos
    env.mj_data.qvel[:] = qvel
    mujoco.mj_forward(env.mj_model, env.mj_data)
    last_target = np.asarray(qpos[7:], dtype=np.float32).copy()
    history: deque[NDArray[np.float32]] = deque(
        maxlen=corpus.proprioception_spec.history_steps
    )
    student_memory: NDArray[np.float32] | None = (
        None
        if student_memory_size is None
        else np.zeros((1, 1, student_memory_size), dtype=np.float32)
    )
    ready_pose_streak = 0
    stable_streak = 0
    maximum_stable_streak = 0
    ready_trigger_step: int | None = None
    finite_state = True
    peak_angular_speed = 0.0
    minimum_pelvis_height = float(qpos[2])
    maximum_target_delta = 0.0
    saturation_count = 0
    torque_count = 0
    clip_count = 0
    target_count = 0
    executed_steps = 0
    final_linear_speed = float(np.linalg.norm(qvel[:3]))
    final_angular_speed = float(np.linalg.norm(qvel[3:6]))
    final_upright = float(-body_gravity_vector(qpos[3:7])[2])
    maximum_steps = int(round(exam_config.maximum_duration_sec / env.dt))
    trace: list[dict[str, Any]] = []
    render_stride = max(1, int(round(0.10 / env.dt)))
    for step in range(maximum_steps):
        gravity = (
            env.mj_data.site_xmat[env._pelvis_imu_site_id].reshape(3, 3).T
            @ np.asarray((0.0, 0.0, -1.0))
        )
        frame = build_recovery_proprioception(
            projected_gravity_body=gravity,
            pelvis_gyro_rad_s=env.get_gyro("pelvis"),
            joint_position_rad=env.mj_data.qpos[7:],
            joint_velocity_rad_s=env.mj_data.qvel[6:],
            last_motor_target_rad=last_target,
            default_joint_position_rad=corpus.default_joint_position_rad,
            spec=corpus.proprioception_spec,
        )
        if not history:
            history.extend(
                frame.copy() for _ in range(corpus.proprioception_spec.history_steps)
            )
        else:
            history.append(frame)
        if absolute_target_provider is not None:
            unclipped_target = np.asarray(
                absolute_target_provider(step, frame),
                dtype=np.float32,
            )
            if unclipped_target.shape != (29,) or not np.all(
                np.isfinite(unclipped_target)
            ):
                raise ValueError("recovery target provider returned an invalid target")
        elif absolute_target_sequence is not None:
            unclipped_target = np.asarray(
                absolute_target_sequence[
                    min(step, absolute_target_sequence.shape[0] - 1)
                ],
                dtype=np.float32,
            )
        elif student_memory is None:
            normalized_target = session.run(
                ["normalized_absolute_motor_target"],
                {"proprio_history": np.stack(history)[None, :, :]},
            )[0][0]
            unclipped_target = denormalize_absolute_motor_targets(
                np.asarray(normalized_target, dtype=np.float32),
                joint_lower_rad=target_lower,
                joint_upper_rad=target_upper,
            )
        else:
            stateful_outputs = session.run(
                ["normalized_absolute_motor_target", "memory_out"],
                {
                    "proprio_sequence": frame[None, None, :],
                    "memory_in": student_memory,
                },
            )
            normalized_target = stateful_outputs[0][0, 0]
            student_memory = np.asarray(stateful_outputs[1], dtype=np.float32)
            unclipped_target = denormalize_absolute_motor_targets(
                np.asarray(normalized_target, dtype=np.float32),
                joint_lower_rad=target_lower,
                joint_upper_rad=target_upper,
            )
        if (
            (absolute_target_sequence is not None or absolute_target_provider is not None)
            and preserve_sequence_target_authority
        ) or preserve_model_target_authority:
            motor_target = np.asarray(unclipped_target, dtype=np.float32)
        else:
            motor_target = np.clip(
                unclipped_target, corpus.joint_lower_rad, corpus.joint_upper_rad
            ).astype(np.float32)
            clip_count += int(np.count_nonzero(motor_target != unclipped_target))
        target_count += motor_target.size
        maximum_target_delta = max(
            maximum_target_delta,
            float(np.max(np.abs(motor_target - last_target))),
        )
        for _ in range(int(env.dt / env.sim_dt)):
            raw_torque = constants.KPs * (
                motor_target - env.mj_data.qpos[7:]
            ) + constants.KDs * (-env.mj_data.qvel[6:])
            saturation_count += int(
                np.count_nonzero(np.abs(raw_torque) > constants.TORQUE_LIMIT)
            )
            torque_count += raw_torque.size
            env.mj_data.ctrl[:] = np.clip(
                raw_torque, -constants.TORQUE_LIMIT, constants.TORQUE_LIMIT
            )
            mujoco.mj_step(env.mj_model, env.mj_data)
        last_target = motor_target
        qpos = np.asarray(env.mj_data.qpos, dtype=np.float64)
        qvel = np.asarray(env.mj_data.qvel, dtype=np.float64)
        finite_state = bool(np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel)))
        executed_steps = step + 1
        if not finite_state:
            break
        gravity_world = body_gravity_vector(qpos[3:7])
        final_upright = float(-gravity_world[2])
        final_linear_speed = float(np.linalg.norm(qvel[:3]))
        final_angular_speed = float(np.linalg.norm(qvel[3:6]))
        peak_angular_speed = max(peak_angular_speed, final_angular_speed)
        minimum_pelvis_height = min(minimum_pelvis_height, float(qpos[2]))
        ready_pose = bool(
            qpos[2] >= exam_config.ready_pelvis_height_m
            and final_upright >= exam_config.ready_upright_projection
        )
        ready_pose_streak = ready_pose_streak + 1 if ready_pose else 0
        if (
            ready_trigger_step is None
            and ready_pose_streak >= exam_config.ready_pose_hold_frames
        ):
            ready_trigger_step = step
        stable = bool(
            ready_pose
            and final_linear_speed <= exam_config.maximum_stable_linear_speed_mps
            and final_angular_speed <= exam_config.maximum_stable_angular_speed_rad_s
        )
        stable_streak = stable_streak + 1 if stable else 0
        maximum_stable_streak = max(maximum_stable_streak, stable_streak)
        if step % render_stride == 0 or stable_streak >= exam_config.final_stable_frames:
            trace.append(
                {
                    "step": step,
                    "time_sec": (step + 1) * env.dt,
                    "pelvis_height_m": float(qpos[2]),
                    "upright_projection": final_upright,
                    "root_linear_speed_mps": final_linear_speed,
                    "root_angular_speed_rad_s": final_angular_speed,
                    "stable_streak": stable_streak,
                }
            )
        if stable_streak >= exam_config.final_stable_frames:
            break
    succeeded = bool(
        finite_state and stable_streak >= exam_config.final_stable_frames
    )
    trial = RecoveryStudentPhysicsTrial(
        initial_snapshot_hash=snapshot.snapshot_hash,
        base_snapshot_hash=base_snapshot_hash,
        suite=suite,
        student_onnx_hash=student_onnx_hash,
        succeeded=succeeded,
        finite_state=finite_state,
        ready_handoff_triggered=ready_trigger_step is not None,
        executed_sec=executed_steps * env.dt,
        final_stable_sec=maximum_stable_streak * env.dt,
        final_pelvis_height_m=max(0.0, float(env.mj_data.qpos[2])),
        final_upright_projection=final_upright,
        final_root_linear_speed_mps=final_linear_speed,
        final_root_angular_speed_rad_s=final_angular_speed,
        peak_root_angular_speed_rad_s=peak_angular_speed,
        minimum_pelvis_height_m=max(0.0, minimum_pelvis_height),
        maximum_target_delta_rad=maximum_target_delta,
        torque_saturation_fraction=(saturation_count / max(1, torque_count)),
        joint_limit_clip_fraction=(clip_count / max(1, target_count)),
    )
    return trial, trace


def run_opentrack_recovery_student_exam(
    *,
    opentrack_root: Path,
    environment_config_path: Path,
    motion_dataset_id: str,
    snapshot_manifest_path: Path,
    development_report_path: Path,
    sealed_holdout_report_path: Path,
    corpus_manifest_path: Path,
    artifact_manifest_path: Path,
    training_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run base and sealed local states with one fixed proprio-only student."""

    root = opentrack_root.expanduser().resolve()
    environment_path = environment_config_path.expanduser().resolve()
    snapshot_path = snapshot_manifest_path.expanduser().resolve()
    development_path = development_report_path.expanduser().resolve()
    holdout_path = sealed_holdout_report_path.expanduser().resolve()
    corpus_path = corpus_manifest_path.expanduser().resolve()
    artifact_path = artifact_manifest_path.expanduser().resolve()
    training_path = training_report_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    files = (
        environment_path,
        snapshot_path,
        development_path,
        holdout_path,
        corpus_path,
        artifact_path,
        training_path,
    )
    if not root.is_dir() or any(not path.is_file() for path in files):
        raise FileNotFoundError("recovery student exam inputs are incomplete")
    if target.exists() or target == root or root in target.parents:
        raise ValueError("recovery student exam output must be new and external")
    development = _verified_development_report(development_path)
    sealed = _verified_holdout_report(holdout_path)
    artifact = _verified_json(
        artifact_path,
        schema_version="rosclaw_soccer.recovery_student_artifact.v1",
        hash_key="manifest_hash",
    )
    training = _verified_json(
        training_path,
        schema_version="rosclaw_soccer.recovery_student_training_report.v1",
        hash_key="report_hash",
    )
    corpus = load_recovery_distillation_corpus(corpus_path)
    corpus_payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    onnx_path = artifact_path.parent / str(artifact.get("onnx"))
    if (
        sealed.get("development_report_hash") != development["report_hash"]
        or development.get("snapshot_manifest_hash") != _file_hash(snapshot_path)
        or development.get("teacher_config_hash") != _file_hash(environment_path)
        or artifact.get("corpus_manifest_hash") != corpus.manifest_hash
        or artifact.get("proprioception_spec_hash")
        != corpus.proprioception_spec.spec_hash
        or artifact.get("contains_reference_features") is not False
        or artifact.get("activation_ceiling") != "SIM_ONLY"
        or artifact.get("hardware_authorized") is not False
        or artifact.get("onnx_hash") != _file_hash(onnx_path)
        or training.get("artifact_manifest_hash") != artifact["manifest_hash"]
        or training.get("student_reads_reference_phase") is not False
        or training.get("student_reads_teacher_identity") is not False
        or corpus_payload.get("body_hash") is None
        or corpus_payload.get("physics_scene_hash") is None
    ):
        raise ValueError("recovery student exam evidence bindings differ")

    base_snapshots = load_recovery_snapshot_corpus(snapshot_path)
    perturbation_payload = dict(sealed["perturbation_config"])
    perturbation = RecoveryPerturbationConfig(**perturbation_payload)
    sealed_snapshots = build_recovery_perturbation_holdout(
        base_snapshots, config=perturbation
    )
    recorded = sealed.get("perturbations")
    if not isinstance(recorded, list) or len(recorded) != len(sealed_snapshots):
        raise ValueError("sealed student exam state count differs")
    for (snapshot, identity), expected in zip(sealed_snapshots, recorded, strict=True):
        if (
            snapshot.snapshot_hash != expected.get("perturbed_snapshot_hash")
            or identity.perturbation_hash != expected.get("perturbation_hash")
        ):
            raise ValueError("sealed student exam state identity differs")

    os.environ.setdefault("GLI_PATH", str(root))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")
    mujoco = importlib.import_module("mujoco")
    tmj = importlib.import_module("track_mj")
    importlib.import_module(
        "track_mj.envs.g1_tracking.play.play_g1_env_tracking_general"
    )
    constants = importlib.import_module(
        "track_mj.envs.g1_tracking.g1_tracking_constants"
    )
    ort = importlib.import_module("onnxruntime")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    session_inputs = tuple(item.name for item in session.get_inputs())
    session_outputs = tuple(item.name for item in session.get_outputs())
    stateful = artifact.get("architecture") == (
        "STATEFUL_FRAME_MLP_GRU_ABSOLUTE_JOINT_TARGET_V1"
    )
    memory_size = int(artifact.get("memory_size", -1)) if stateful else None
    stateless_io_valid = bool(
        not stateful
        and session_inputs == ("proprio_history",)
        and session_outputs == ("normalized_absolute_motor_target",)
        and session.get_inputs()[0].shape[1:]
        == [
            corpus.proprioception_spec.history_steps,
            corpus.proprioception_spec.observation_dim,
        ]
    )
    stateful_io_valid = bool(
        stateful
        and memory_size is not None
        and memory_size >= 32
        and session_inputs == ("proprio_sequence", "memory_in")
        and session_outputs
        == ("normalized_absolute_motor_target", "memory_out")
        and session.get_inputs()[0].shape[2:] == [
            corpus.proprioception_spec.observation_dim
        ]
        and session.get_inputs()[1].shape[0] == 1
        and session.get_inputs()[1].shape[2] == memory_size
    )
    if not stateless_io_valid and not stateful_io_valid:
        raise ValueError("recovery student ONNX IO is incompatible")
    extended_target_authority = artifact.get("motor_target_authority") == (
        "TRAINING_TEACHER_PD_TARGET_ENVELOPE"
    )
    target_lower = (
        np.asarray(artifact.get("motor_target_lower_rad"), dtype=np.float32)
        if extended_target_authority
        else corpus.joint_lower_rad
    )
    target_upper = (
        np.asarray(artifact.get("motor_target_upper_rad"), dtype=np.float32)
        if extended_target_authority
        else corpus.joint_upper_rad
    )
    if (
        target_lower.shape != (29,)
        or target_upper.shape != (29,)
        or np.any(target_lower > corpus.joint_lower_rad + 1.0e-7)
        or np.any(target_upper < corpus.joint_upper_rad - 1.0e-7)
        or not np.all(np.isfinite(target_lower))
        or not np.all(np.isfinite(target_upper))
        or np.any(target_upper <= target_lower)
        or (
            extended_target_authority
            and artifact.get("torque_limit_required") is not True
        )
    ):
        raise ValueError("recovery student motor target authority differs")

    environment_payload = json.loads(environment_path.read_text(encoding="utf-8"))
    if not isinstance(environment_payload.get("env_config"), dict):
        raise ValueError("recovery student environment config is invalid")
    first_route = development["post_skill_transfer"]["development_schedule"][
        "selected_trials"
    ][0]
    bootstrap_match = first_route["match"]
    environment_config = copy.deepcopy(
        tmj.registry.get("G1TrackingGeneral", "tracking_config").env_config
    )
    environment_config.update(environment_payload["env_config"])
    environment_config.reference_traj_config.name = {
        motion_dataset_id: [bootstrap_match["motion_id"]]
    }
    environment_config.reference_traj_config.random_start = False
    environment_config.reference_traj_config.fixed_start_frame = bootstrap_match[
        "entry_frame"
    ]
    environment_class = tmj.registry.get(
        "G1TrackingGeneral", "tracking_play_env_class"
    )
    previous_directory = Path.cwd()
    try:
        os.chdir(root)
        environment = environment_class(
            config=environment_config,
            play_ref_motion=False,
            use_viewer=False,
            use_renderer=False,
            exp_name="rosclaw-s52-recovery-student-exam",
        )
    finally:
        os.chdir(previous_directory)
    try:
        observed_body_hash = _teacher_body_hash(environment, mujoco)
        observed_scene_hash = _file_hash(
            Path(constants.task_to_xml("flat_terrain")).expanduser().resolve()
        )
        default_error = float(
            np.max(
                np.abs(
                    np.asarray(environment._default_qpos)
                    - corpus.default_joint_position_rad
                )
            )
        )
        lower_error = float(
            np.max(np.abs(np.asarray(environment._lowers) - corpus.joint_lower_rad))
        )
        upper_error = float(
            np.max(np.abs(np.asarray(environment._uppers) - corpus.joint_upper_rad))
        )
        if (
            observed_body_hash != corpus_payload["body_hash"]
            or observed_scene_hash != corpus_payload["physics_scene_hash"]
            or max(default_error, lower_error, upper_error) > 1.0e-6
        ):
            raise ValueError(
                "recovery student body or scene contract differs: "
                + json.dumps(
                    {
                        "body_hash_equal": (
                            observed_body_hash == corpus_payload["body_hash"]
                        ),
                        "scene_hash_equal": (
                            observed_scene_hash == corpus_payload["physics_scene_hash"]
                        ),
                        "default_error_rad": default_error,
                        "lower_error_rad": lower_error,
                        "upper_error_rad": upper_error,
                    },
                    sort_keys=True,
                )
            )
        exam_payload = dict(development["exam_config"])
        exam_payload["time_dilations"] = tuple(exam_payload["time_dilations"])
        exam_config = OpenTrackRecoveryBridgeExamConfig(**exam_payload)
        onnx_hash = _file_hash(onnx_path)
        trials: list[RecoveryStudentPhysicsTrial] = []
        traces: list[dict[str, Any]] = []
        for base in base_snapshots:
            trial, trace = _run_student_trial(
                env=environment,
                session=session,
                snapshot=base,
                base_snapshot_hash=base.snapshot_hash,
                suite="DEVELOPMENT_BASE",
                student_onnx_hash=onnx_hash,
                corpus=corpus,
                exam_config=exam_config,
                constants=constants,
                mujoco=mujoco,
                student_memory_size=memory_size,
                motor_target_lower_rad=target_lower,
                motor_target_upper_rad=target_upper,
                preserve_model_target_authority=extended_target_authority,
            )
            trials.append(trial)
            traces.append({"trial_hash": trial.trial_hash, "samples": trace})
        for perturbed_snapshot, identity in sealed_snapshots:
            trial, trace = _run_student_trial(
                env=environment,
                session=session,
                snapshot=perturbed_snapshot,
                base_snapshot_hash=identity.base_snapshot_hash,
                suite="SEALED_LOCAL_HOLDOUT",
                student_onnx_hash=onnx_hash,
                corpus=corpus,
                exam_config=exam_config,
                constants=constants,
                mujoco=mujoco,
                student_memory_size=memory_size,
                motor_target_lower_rad=target_lower,
                motor_target_upper_rad=target_upper,
                preserve_model_target_authority=extended_target_authority,
            )
            trials.append(trial)
            traces.append({"trial_hash": trial.trial_hash, "samples": trace})
    finally:
        environment.close()

    base_trials = [item for item in trials if item.suite == "DEVELOPMENT_BASE"]
    holdout_trials = [item for item in trials if item.suite == "SEALED_LOCAL_HOLDOUT"]
    holdout_passed = sum(item.succeeded for item in holdout_trials)
    per_base: list[dict[str, Any]] = []
    for base_hash in sorted(item.snapshot_hash for item in base_snapshots):
        selection = [
            item for item in holdout_trials if item.base_snapshot_hash == base_hash
        ]
        passed = sum(item.succeeded for item in selection)
        per_base.append(
            {
                "base_snapshot_hash": base_hash,
                "passed_count": passed,
                "trial_count": len(selection),
                "pass_rate": passed / len(selection),
            }
        )
    local_holdout_passed = bool(
        holdout_passed / len(holdout_trials) >= 0.80
        and all(item["pass_rate"] >= 2.0 / 3.0 for item in per_base)
        and all(item.finite_state for item in holdout_trials)
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_student_physics_exam.v1",
        "development_report_hash": development["report_hash"],
        "sealed_holdout_report_hash": sealed["report_hash"],
        "corpus_manifest_hash": corpus.manifest_hash,
        "artifact_manifest_hash": artifact["manifest_hash"],
        "training_report_hash": training["report_hash"],
        "student_onnx_hash": _file_hash(onnx_path),
        "proprioception_spec_hash": corpus.proprioception_spec.spec_hash,
        "exam_config": asdict(exam_config),
        "exam_config_hash": exam_config.config_hash,
        "control_path": (
            "ONNX_PROPRIO_TO_INTERNAL_MEMORY_TO_ABSOLUTE_TARGET_TO_PD"
            if stateful
            else "ONNX_PROPRIO_HISTORY_TO_ABSOLUTE_TARGET_TO_PD"
        ),
        "student_uses_internal_memory": stateful,
        "student_memory_size": memory_size,
        "motor_target_authority": artifact.get(
            "motor_target_authority", "PHYSICAL_JOINT_RANGE"
        ),
        "physical_joint_limits_enforced_by_mujoco": True,
        "torque_limits_enforced_each_substep": True,
        "reference_phase_reads_during_control": 0,
        "teacher_identity_reads_during_control": 0,
        "environment_step_calls_during_control": 0,
        "trajectory_handler_reads_during_control": 0,
        "bootstrap_motion_role": "ENVIRONMENT_MODEL_CONSTRUCTION_ONLY",
        "base_trial_count": len(base_trials),
        "base_passed_count": sum(item.succeeded for item in base_trials),
        "base_pass_rate": sum(item.succeeded for item in base_trials)
        / len(base_trials),
        "sealed_holdout_trial_count": len(holdout_trials),
        "sealed_holdout_passed_count": holdout_passed,
        "sealed_holdout_pass_rate": holdout_passed / len(holdout_trials),
        "sealed_holdout_wilson_95_lower_bound": _wilson_lower_bound(
            passed=holdout_passed, count=len(holdout_trials)
        ),
        "per_base_results": per_base,
        "local_holdout_passed": local_holdout_passed,
        "trials": [item.to_dict() | {"trial_hash": item.trial_hash} for item in trials],
        "traces": traces,
        "maximum_peak_root_angular_speed_rad_s": max(
            item.peak_root_angular_speed_rad_s for item in trials
        ),
        "maximum_target_delta_rad": max(item.maximum_target_delta_rad for item in trials),
        "mean_torque_saturation_fraction": float(
            np.mean([item.torque_saturation_fraction for item in trials])
        ),
        "mean_joint_limit_clip_fraction": float(
            np.mean([item.joint_limit_clip_fraction for item in trials])
        ),
        "student_contains_reference_features": False,
        "physical_truth": True,
        "physics_backend": "opentrack_mujoco_cpu_direct_pd",
        "promotion_eligible": False,
        "promotion_blockers": [
            *([] if local_holdout_passed else ["SEALED_LOCAL_HOLDOUT_FAILED"]),
            "LOCAL_PERTURBATION_HOLDOUT_IS_NOT_NEW_POST_SKILL_EPISODES",
            "NO_SOURCE_SCENE_FULL_CHAIN_ROLLOUT",
            "NO_INDEPENDENT_CROSS_SCENE_STUDENT_QUALIFICATION",
        ],
        "claim_boundary": "PROPRIO_ONLY_LOCAL_PHYSICS_EXAM_NOT_DEPLOYMENT_PROMOTION",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(target, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opentrack-root", required=True, type=Path)
    parser.add_argument("--environment-config", required=True, type=Path)
    parser.add_argument("--motion-dataset-id", required=True)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--development-report", required=True, type=Path)
    parser.add_argument("--sealed-holdout-report", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--artifact-manifest", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()
    report = run_opentrack_recovery_student_exam(
        opentrack_root=args.opentrack_root,
        environment_config_path=args.environment_config,
        motion_dataset_id=args.motion_dataset_id,
        snapshot_manifest_path=args.snapshot_manifest,
        development_report_path=args.development_report,
        sealed_holdout_report_path=args.sealed_holdout_report,
        corpus_manifest_path=args.corpus_manifest,
        artifact_manifest_path=args.artifact_manifest,
        training_report_path=args.training_report,
        output_path=args.output_path,
    )
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "base_pass_rate": report["base_pass_rate"],
                "sealed_holdout_pass_rate": report["sealed_holdout_pass_rate"],
                "local_holdout_passed": report["local_holdout_passed"],
                "promotion_eligible": report["promotion_eligible"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RecoveryStudentPhysicsTrial",
    "run_opentrack_recovery_student_exam",
]
