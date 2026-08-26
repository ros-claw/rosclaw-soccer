"""Paired OpenTrack recovery-teacher exam on source and post-skill states.

The external policy is reference-conditioned and the injected snapshots may
come from a materially different MuJoCo scene.  This exam therefore measures a
teacher bridge for development; it never promotes the result as a deployable
proprioceptive recovery controller or as proof in the source physics scene.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.recovery_snapshot import load_recovery_snapshot_corpus
from rosclaw_soccer.training.recovery_teacher_bridge import (
    RecoveryBridgeTrial,
    RecoveryEntryMatch,
    RecoveryEntryMatcher,
    RecoveryEntrySearchConfig,
    body_gravity_vector,
    build_recovery_bridge_schedule,
)

_DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class OpenTrackRecoveryBridgeExamConfig:
    """Bounded failure-driven search for a recovery-teacher bridge."""

    maximum_snapshots: int = 64
    maximum_match_candidates: int = 3
    time_dilations: tuple[int, ...] = (1, 2)
    maximum_duration_sec: float = 40.0
    ready_pose_hold_frames: int = 10
    final_stable_frames: int = 100
    ready_pelvis_height_m: float = 0.62
    ready_upright_projection: float = 0.75
    maximum_stable_linear_speed_mps: float = 0.50
    maximum_stable_angular_speed_rad_s: float = 1.50
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.opentrack_recovery_bridge_exam_config.v1"

    def __post_init__(self) -> None:
        scalars = (
            self.maximum_duration_sec,
            self.ready_pelvis_height_m,
            self.ready_upright_projection,
            self.maximum_stable_linear_speed_mps,
            self.maximum_stable_angular_speed_rad_s,
        )
        if (
            not 1 <= self.maximum_snapshots <= 64
            or not 1 <= self.maximum_match_candidates <= 8
            or not self.time_dilations
            or len(set(self.time_dilations)) != len(self.time_dilations)
            or any(not 1 <= value <= 4 for value in self.time_dilations)
            or tuple(sorted(self.time_dilations)) != self.time_dilations
            or not all(math.isfinite(value) for value in scalars)
            or not 10.0 <= self.maximum_duration_sec <= 60.0
            or not 1 <= self.ready_pose_hold_frames <= 100
            or not 50 <= self.final_stable_frames <= 250
            or not 0.50 <= self.ready_pelvis_height_m <= 0.90
            or not 0.50 <= self.ready_upright_projection <= 1.0
            or min(
                self.maximum_stable_linear_speed_mps,
                self.maximum_stable_angular_speed_rad_s,
            )
            <= 0.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("OpenTrack recovery bridge exam config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _trial_key(
    *, kind: str, snapshot_hash: str, match_hash: str, time_dilation: int
) -> str:
    return str(
        hash_json(
            {
                "kind": kind,
                "snapshot_hash": snapshot_hash,
                "match_hash": match_hash,
                "time_dilation": time_dilation,
            }
        )
    )


def _append_trial_journal(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_trial_journal(
    path: Path, *, expected_bindings: dict[str, str]
) -> dict[str, tuple[RecoveryBridgeTrial, dict[str, Any]]]:
    if not path.exists():
        return {}
    recovered: dict[str, tuple[RecoveryBridgeTrial, dict[str, Any]]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"recovery trial journal line {line_number} is invalid") from exc
        if (
            not isinstance(row, dict)
            or set(row) != {"bindings", "key", "trace", "trial"}
            or row["bindings"] != expected_bindings
            or not isinstance(row["trial"], dict)
            or not isinstance(row["trace"], dict)
        ):
            raise ValueError("recovery trial journal binding is invalid")
        raw_trial = dict(row["trial"])
        raw_match = raw_trial.pop("match", None)
        recorded_hash = raw_trial.pop("trial_hash", None)
        if not isinstance(raw_match, dict):
            raise ValueError("recovery trial journal match is invalid")
        trial = RecoveryBridgeTrial(
            match=RecoveryEntryMatch(**raw_match),
            **raw_trial,
        )
        if recorded_hash != trial.trial_hash or row["key"] in recovered:
            raise ValueError("recovery trial journal integrity check failed")
        recovered[str(row["key"])] = (trial, dict(row["trace"]))
    return recovered


def _git_head(root: Path) -> str:
    head = root / ".git" / "HEAD"
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        value = (root / ".git" / value.removeprefix("ref: ")).read_text(
            encoding="utf-8"
        ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("OpenTrack checkout must have a pinned readable commit")
    return value


def _actuator_joint_names(model: Any, mujoco: Any) -> tuple[str, ...]:
    return tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index))
        for index in range(model.nu)
    )


def _scene_compatibility(
    *, source_scene_path: Path, teacher_scene_path: Path, mujoco: Any
) -> dict[str, Any]:
    source_path = source_scene_path.expanduser().resolve()
    teacher_path = teacher_scene_path.expanduser().resolve()
    source = mujoco.MjModel.from_xml_path(str(source_path))
    teacher = mujoco.MjModel.from_xml_path(str(teacher_path))
    source_names = _actuator_joint_names(source, mujoco)
    teacher_names = _actuator_joint_names(teacher, mujoco)
    joint_order_equal = source_names == teacher_names and len(source_names) == 29
    range_error = (
        float(np.max(np.abs(source.jnt_range[1:30] - teacher.jnt_range[1:30])))
        if joint_order_equal
        else None
    )
    common_bodies: list[tuple[int, int]] = []
    missing_bodies: list[str] = []
    for teacher_id in range(1, teacher.nbody):
        name = mujoco.mj_id2name(teacher, mujoco.mjtObj.mjOBJ_BODY, teacher_id)
        source_id = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_BODY, name)
        if source_id < 0:
            missing_bodies.append(str(name))
        else:
            common_bodies.append((source_id, teacher_id))
    mass_error = max(
        (abs(float(source.body_mass[a] - teacher.body_mass[b])) for a, b in common_bodies),
        default=math.inf,
    )
    inertia_error = max(
        (
            float(np.max(np.abs(source.body_inertia[a] - teacher.body_inertia[b])))
            for a, b in common_bodies
        ),
        default=math.inf,
    )
    # Contact topology is intentionally conservative.  Different counts or
    # collision masks are enough to reject scene equivalence.
    source_contact = sorted(
        (int(source.geom_type[i]), int(source.geom_contype[i]), int(source.geom_conaffinity[i]))
        for i in range(source.ngeom)
        if int(source.geom_contype[i]) or int(source.geom_conaffinity[i])
    )
    teacher_contact = sorted(
        (
            int(teacher.geom_type[i]),
            int(teacher.geom_contype[i]),
            int(teacher.geom_conaffinity[i]),
        )
        for i in range(teacher.ngeom)
        if int(teacher.geom_contype[i]) or int(teacher.geom_conaffinity[i])
    )
    contact_topology_equal = source_contact == teacher_contact
    equivalent = bool(
        joint_order_equal
        and range_error is not None
        and range_error <= 1e-12
        and not missing_bodies
        and mass_error <= 1e-9
        and inertia_error <= 1e-9
        and contact_topology_equal
    )
    return {
        "source_scene_hash": _file_hash(source_path),
        "teacher_scene_hash": _file_hash(teacher_path),
        "joint_order_equal": joint_order_equal,
        "maximum_joint_range_error_rad": range_error,
        "missing_teacher_body_names": missing_bodies,
        "maximum_common_body_mass_error_kg": mass_error,
        "maximum_common_body_inertia_error": inertia_error,
        "contact_topology_equal": contact_topology_equal,
        "scene_equivalent": equivalent,
    }


def _restore_reference(
    *, env: Any, state: Any, carry: Any, mujoco: Any
) -> Any:
    state.info["traj_info"] = carry
    env.current_traj_info = carry
    trajectory = env.th.get_current_traj_data(carry)
    env.ref_mj_data.qpos[:] = trajectory.qpos
    env.ref_mj_data.qvel[:] = trajectory.qvel
    mujoco.mj_forward(env.mj_model, env.ref_mj_data)
    observation, _ = env.get_obs(trajectory, state.info)
    return type(state)(state.info, observation)


def _run_bridge_trial(
    *,
    env: Any,
    session: Any,
    snapshot: Any | None,
    snapshot_hash: str,
    match: RecoveryEntryMatch,
    teacher_policy_hash: str,
    time_dilation: int,
    config: OpenTrackRecoveryBridgeExamConfig,
    mujoco: Any,
    frame_callback: Callable[[Any, int, bool], None] | None = None,
    transition_callback: Callable[[Any, Any, NDArray[np.float64], int], None]
    | None = None,
) -> tuple[RecoveryBridgeTrial, dict[str, Any]]:
    state = env.reset()
    if snapshot is not None:
        qpos = np.asarray(snapshot.qpos, dtype=np.float64).copy()
        qvel = np.asarray(snapshot.qvel, dtype=np.float64).copy()
        # Root x/y is an irrelevant world-frame offset but can be outside a
        # teacher terrain tile.  Keep the reference location without changing
        # the physical posture, momentum, or 29 joint values.
        qpos[:2] = np.asarray(env.mj_data.qpos[:2], dtype=np.float64)
        env.mj_data.qpos[:] = qpos
        env.mj_data.qvel[:] = qvel
        mujoco.mj_forward(env.mj_model, env.mj_data)
        trajectory = env.th.get_current_traj_data(state.info["traj_info"])
        env.ref_mj_data.qpos[:] = trajectory.qpos
        env.ref_mj_data.qvel[:] = trajectory.qvel
        mujoco.mj_forward(env.mj_model, env.ref_mj_data)
        state.info["last_motor_targets"] = qpos[7:].copy()
        observation, _ = env.get_obs(trajectory, state.info)
        state = type(state)(state.info, observation)

    ready_pose_streak = 0
    final_stable_streak = 0
    maximum_final_stable_streak = 0
    ready_carry: Any | None = None
    ready_trigger_step: int | None = None
    peak_angular_speed = 0.0
    minimum_pelvis_height = float(env.mj_data.qpos[2])
    finite_state = True
    maximum_steps = int(round(config.maximum_duration_sec / env.dt))
    executed_steps = 0
    final_linear_speed = math.inf
    final_angular_speed = math.inf
    final_upright = -1.0
    for step in range(maximum_steps):
        carry = state.info["traj_info"]
        inputs = {"obs": np.asarray(state.obs["state"], dtype=np.float32).reshape(1, -1)}
        action = session.run(["continuous_actions"], inputs)[0][0]
        if transition_callback is not None:
            transition_callback(
                env,
                state,
                np.asarray(action, dtype=np.float64),
                step,
            )
        state = env.step(state, action)
        should_hold = ready_carry is not None or step % time_dilation != time_dilation - 1
        if should_hold:
            state = _restore_reference(
                env=env,
                state=state,
                carry=ready_carry if ready_carry is not None else carry,
                mujoco=mujoco,
            )
        qpos = np.asarray(env.mj_data.qpos, dtype=np.float64)
        qvel = np.asarray(env.mj_data.qvel, dtype=np.float64)
        finite_state = bool(np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel)))
        executed_steps = step + 1
        if not finite_state:
            break
        gravity = body_gravity_vector(qpos[3:7])
        final_upright = float(-gravity[2])
        final_linear_speed = float(np.linalg.norm(qvel[:3]))
        final_angular_speed = float(np.linalg.norm(qvel[3:6]))
        peak_angular_speed = max(peak_angular_speed, final_angular_speed)
        minimum_pelvis_height = min(minimum_pelvis_height, float(qpos[2]))
        ready_pose = bool(
            qpos[2] >= config.ready_pelvis_height_m
            and final_upright >= config.ready_upright_projection
        )
        ready_pose_streak = ready_pose_streak + 1 if ready_pose else 0
        if ready_carry is None and ready_pose_streak >= config.ready_pose_hold_frames:
            ready_carry = state.info["traj_info"]
            ready_trigger_step = step
        stable = bool(
            ready_pose
            and final_linear_speed <= config.maximum_stable_linear_speed_mps
            and final_angular_speed <= config.maximum_stable_angular_speed_rad_s
        )
        final_stable_streak = final_stable_streak + 1 if stable else 0
        maximum_final_stable_streak = max(
            maximum_final_stable_streak, final_stable_streak
        )
        if frame_callback is not None:
            frame_callback(env, step, ready_carry is not None)
        if final_stable_streak >= config.final_stable_frames:
            break
    succeeded = bool(
        finite_state and final_stable_streak >= config.final_stable_frames
    )
    trial = RecoveryBridgeTrial(
        snapshot_hash=snapshot_hash,
        match=match,
        teacher_policy_hash=teacher_policy_hash,
        time_dilation=time_dilation,
        succeeded=succeeded,
        final_stable_sec=maximum_final_stable_streak * env.dt,
        executed_sec=executed_steps * env.dt,
        peak_root_angular_speed_rad_s=peak_angular_speed,
        final_pelvis_height_m=max(0.0, float(env.mj_data.qpos[2])),
        finite_state=finite_state,
        ready_handoff_triggered=ready_trigger_step is not None,
    )
    trace_summary = {
        "trial_hash": trial.trial_hash,
        "ready_trigger_step": ready_trigger_step,
        "minimum_pelvis_height_m": minimum_pelvis_height,
        "final_upright_projection": final_upright,
        "final_root_linear_speed_mps": final_linear_speed,
        "final_root_angular_speed_rad_s": final_angular_speed,
    }
    return trial, trace_summary


def run_opentrack_recovery_bridge_exam(
    *,
    opentrack_root: Path,
    teacher_policy_path: Path,
    teacher_config_path: Path,
    motion_paths: tuple[Path, ...],
    motion_dataset_id: str,
    snapshot_manifest_path: Path,
    source_scene_path: Path,
    output_path: Path,
    search_config: RecoveryEntrySearchConfig | None = None,
    exam_config: OpenTrackRecoveryBridgeExamConfig | None = None,
) -> dict[str, Any]:
    """Run paired source-neighborhood and snapshot-transfer physics trials."""

    active = exam_config or OpenTrackRecoveryBridgeExamConfig()
    root = opentrack_root.expanduser().resolve()
    policy_path = teacher_policy_path.expanduser().resolve()
    config_path = teacher_config_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if (
        not root.is_dir()
        or not policy_path.is_file()
        or not config_path.is_file()
        or not source_scene_path.expanduser().resolve().is_file()
    ):
        raise FileNotFoundError("OpenTrack recovery bridge inputs are incomplete")
    if not _DATASET_ID.fullmatch(motion_dataset_id):
        raise ValueError("OpenTrack motion dataset id is invalid")
    if target.exists() or target == root or root in target.parents:
        raise ValueError("bridge evidence must be new and outside the OpenTrack checkout")
    expected_motion_root = (
        root / "storage" / "data" / "mocap" / motion_dataset_id / "UnitreeG1"
    )
    resolved_motions = tuple(path.expanduser().resolve() for path in motion_paths)
    if not resolved_motions or any(
        path.parent != expected_motion_root for path in resolved_motions
    ):
        raise ValueError("motion paths must belong to the declared OpenTrack dataset")

    os.environ.setdefault("GLI_PATH", str(root))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")
    mujoco = importlib.import_module("mujoco")
    tmj = importlib.import_module("track_mj")
    importlib.import_module("track_mj.envs.g1_tracking.play.play_g1_env_tracking_general")
    constants = importlib.import_module("track_mj.envs.g1_tracking.g1_tracking_constants")
    ort = importlib.import_module("onnxruntime")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("env_config"), dict):
        raise ValueError("OpenTrack teacher config has no environment contract")
    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    inputs = tuple(item.name for item in session.get_inputs())
    outputs = tuple(item.name for item in session.get_outputs())
    if inputs != ("obs",) or outputs != ("continuous_actions",):
        raise ValueError("OpenTrack recovery teacher IO is incompatible")

    corpus = load_recovery_snapshot_corpus(snapshot_manifest_path)[
        : active.maximum_snapshots
    ]
    if not corpus:
        raise ValueError("OpenTrack recovery bridge snapshot selection is empty")
    matcher = RecoveryEntryMatcher.from_paths(resolved_motions, config=search_config)
    teacher_hash = _file_hash(policy_path)
    snapshot_manifest_hash = _file_hash(snapshot_manifest_path.expanduser().resolve())
    journal_path = target.with_suffix(target.suffix + ".trials.jsonl")
    journal_bindings = {
        "exam_config_hash": active.config_hash,
        "search_config_hash": matcher.config.config_hash,
        "reference_library_hash": matcher.library_hash,
        "snapshot_manifest_hash": snapshot_manifest_hash,
        "teacher_policy_hash": teacher_hash,
    }
    recovered_trials = _load_trial_journal(
        journal_path, expected_bindings=journal_bindings
    )
    scene_path = Path(constants.task_to_xml("flat_terrain")).expanduser().resolve()
    compatibility = _scene_compatibility(
        source_scene_path=source_scene_path,
        teacher_scene_path=scene_path,
        mujoco=mujoco,
    )

    def make_env(match: RecoveryEntryMatch) -> Any:
        environment_config = copy.deepcopy(
            tmj.registry.get("G1TrackingGeneral", "tracking_config").env_config
        )
        environment_config.update(payload["env_config"])
        environment_config.reference_traj_config.name = {
            motion_dataset_id: [match.motion_id]
        }
        environment_config.reference_traj_config.random_start = False
        environment_config.reference_traj_config.fixed_start_frame = match.entry_frame
        environment_class = tmj.registry.get(
            "G1TrackingGeneral", "tracking_play_env_class"
        )
        # OpenTrack currently resolves motion archives against process cwd.
        # Restrict that compatibility shim to construction and always restore
        # the caller's directory, including when upstream initialization fails.
        previous_directory = Path.cwd()
        try:
            os.chdir(root)
            return environment_class(
                config=environment_config,
                play_ref_motion=False,
                use_viewer=False,
                use_renderer=False,
                exp_name="rosclaw-s51-recovery-bridge",
            )
        finally:
            os.chdir(previous_directory)

    source_trials: list[RecoveryBridgeTrial] = []
    source_trace: list[dict[str, Any]] = []
    transfer_trials: list[RecoveryBridgeTrial] = []
    transfer_trace: list[dict[str, Any]] = []
    for snapshot in corpus:
        matches = matcher.match(
            snapshot, maximum_matches=active.maximum_match_candidates
        )
        source_key = _trial_key(
            kind="source",
            snapshot_hash=snapshot.snapshot_hash,
            match_hash=matches[0].match_hash,
            time_dilation=1,
        )
        if source_key in recovered_trials:
            source_trial, trace = recovered_trials[source_key]
        else:
            source_env = make_env(matches[0])
            try:
                source_trial, trace = _run_bridge_trial(
                    env=source_env,
                    session=session,
                    snapshot=None,
                    snapshot_hash=snapshot.snapshot_hash,
                    match=matches[0],
                    teacher_policy_hash=teacher_hash,
                    time_dilation=1,
                    config=active,
                    mujoco=mujoco,
                )
            finally:
                source_env.close()
            _append_trial_journal(
                journal_path,
                {
                    "bindings": journal_bindings,
                    "key": source_key,
                    "trace": trace,
                    "trial": source_trial.to_dict()
                    | {"trial_hash": source_trial.trial_hash},
                },
            )
        source_trials.append(source_trial)
        source_trace.append(trace)
        for match in matches:
            for dilation in active.time_dilations:
                transfer_key = _trial_key(
                    kind="transfer",
                    snapshot_hash=snapshot.snapshot_hash,
                    match_hash=match.match_hash,
                    time_dilation=dilation,
                )
                if transfer_key in recovered_trials:
                    trial, trace = recovered_trials[transfer_key]
                else:
                    transfer_env = make_env(match)
                    try:
                        trial, trace = _run_bridge_trial(
                            env=transfer_env,
                            session=session,
                            snapshot=snapshot,
                            snapshot_hash=snapshot.snapshot_hash,
                            match=match,
                            teacher_policy_hash=teacher_hash,
                            time_dilation=dilation,
                            config=active,
                            mujoco=mujoco,
                        )
                    finally:
                        transfer_env.close()
                    _append_trial_journal(
                        journal_path,
                        {
                            "bindings": journal_bindings,
                            "key": transfer_key,
                            "trace": trace,
                            "trial": trial.to_dict() | {"trial_hash": trial.trial_hash},
                        },
                    )
                transfer_trials.append(trial)
                transfer_trace.append(trace)

    schedule = build_recovery_bridge_schedule(transfer_trials)
    source_passes = sum(item.succeeded for item in source_trials)
    selected = schedule["selected_trials"]
    selected_peaks = [
        float(item["peak_root_angular_speed_rad_s"])
        for item in selected
        if bool(item["succeeded"])
    ]
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.opentrack_recovery_bridge_exam.v1",
        "exam_config": asdict(active),
        "exam_config_hash": active.config_hash,
        "search_config": asdict(matcher.config),
        "search_config_hash": matcher.config.config_hash,
        "reference_library_hash": matcher.library_hash,
        "snapshot_manifest_hash": snapshot_manifest_hash,
        "snapshot_count": len(corpus),
        "teacher_policy_hash": teacher_hash,
        "teacher_config_hash": _file_hash(config_path),
        "teacher_policy_io": {"inputs": list(inputs), "outputs": list(outputs)},
        "opentrack_commit": _git_head(root),
        "physics_backend": "opentrack_mujoco_cpu",
        "physical_truth": True,
        "scene_compatibility": compatibility,
        "source_neighborhood": {
            "passed_count": source_passes,
            "trial_count": len(source_trials),
            "pass_rate": source_passes / len(source_trials),
            "trials": [item.to_dict() | {"trial_hash": item.trial_hash} for item in source_trials],
            "trace_summaries": source_trace,
        },
        "post_skill_transfer": {
            "development_schedule": schedule,
            "trial_count": len(transfer_trials),
            "trials": [
                item.to_dict() | {"trial_hash": item.trial_hash}
                for item in transfer_trials
            ],
            "trace_summaries": transfer_trace,
            "selected_success_peak_angular_speed_maximum_rad_s": (
                max(selected_peaks) if selected_peaks else None
            ),
        },
        "trial_journal": {
            "path": journal_path.name,
            "hash": _file_hash(journal_path),
            "recovered_trial_count": len(recovered_trials),
            "completed_trial_count": len(source_trials) + len(transfer_trials),
        },
        "teacher_role": "PRIVILEGED_REFERENCE_CONDITIONED_TRAINING_TEACHER",
        "promotion_eligible": False,
        "promotion_blockers": [
            "DEVELOPMENT_SNAPSHOTS_USED_FOR_ROUTE_SELECTION",
            "REFERENCE_PHASE_AND_TEACHER_ID_ARE_PRIVILEGED",
            *([] if compatibility["scene_equivalent"] else ["PHYSICS_SCENE_NOT_EQUIVALENT"]),
            "NO_INDEPENDENT_PERTURBATION_HOLDOUT",
            "NO_SOURCE_SCENE_FULL_CHAIN_ROLLOUT",
        ],
        "claim_boundary": (
            "CROSS_SCENE_TEACHER_BRIDGE_DEVELOPMENT_NOT_SOURCE_SCENE_PROMOTION"
        ),
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(target, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opentrack-root", required=True, type=Path)
    parser.add_argument("--teacher-policy", required=True, type=Path)
    parser.add_argument("--teacher-config", required=True, type=Path)
    parser.add_argument("--motion-path", required=True, action="append", type=Path)
    parser.add_argument("--motion-dataset-id", required=True)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--source-scene", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--maximum-match-candidates", default=3, type=int)
    parser.add_argument("--maximum-duration-sec", default=40.0, type=float)
    args = parser.parse_args()
    report = run_opentrack_recovery_bridge_exam(
        opentrack_root=args.opentrack_root,
        teacher_policy_path=args.teacher_policy,
        teacher_config_path=args.teacher_config,
        motion_paths=tuple(args.motion_path),
        motion_dataset_id=args.motion_dataset_id,
        snapshot_manifest_path=args.snapshot_manifest,
        source_scene_path=args.source_scene,
        output_path=args.output_path,
        exam_config=OpenTrackRecoveryBridgeExamConfig(
            maximum_match_candidates=args.maximum_match_candidates,
            maximum_duration_sec=args.maximum_duration_sec,
        ),
    )
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "development_pass_rate": report["post_skill_transfer"][
                    "development_schedule"
                ]["development_pass_rate"],
                "promotion_eligible": report["promotion_eligible"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OpenTrackRecoveryBridgeExamConfig",
    "run_opentrack_recovery_bridge_exam",
]
