"""Bounded CPU-MuJoCo exam for external OpenTrack policies.

The OpenTrack checkout and weights remain external.  This module owns only the
matched, content-addressed exam contract and a deliberately optional runtime
bridge.  Importing ROSClaw Soccer never imports JAX, OpenTrack, or ONNX Runtime.
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.athlete_foundation.foundation_shootout import (
    FoundationMetrics,
    FoundationThresholds,
)

_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class OpenTrackEpisodeSpec:
    """One deterministic motion window in a matched foundation exam."""

    episode_id: str
    suite_id: str
    dataset_id: str
    motion_id: str
    source_hash: str
    license_id: str
    start_frame: int = 0
    max_steps: int = 600
    critical: bool = False

    def __post_init__(self) -> None:
        for label, value in (
            ("episode_id", self.episode_id),
            ("suite_id", self.suite_id),
            ("dataset_id", self.dataset_id),
            ("motion_id", self.motion_id),
            ("license_id", self.license_id),
        ):
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"{label} is not a normalized identifier")
        if not _SHA256.fullmatch(self.source_hash):
            raise ValueError("source_hash must be a sha256: content hash")
        if self.start_frame < 0 or not 50 <= self.max_steps <= 5000:
            raise ValueError("episode window is outside the bounded exam range")


@dataclass(frozen=True)
class OpenTrackFoundationExamPlan:
    """Stable/plastic suites evaluated with the same body, physics, and policy."""

    episodes: tuple[OpenTrackEpisodeSpec, ...]
    thresholds: FoundationThresholds = field(default_factory=FoundationThresholds)
    body_id: str = "unitree.g1.29dof"
    physics_backend: str = "mujoco_cpu"
    control_dt_sec: float = 0.02
    sim_dt_sec: float = 0.002
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.opentrack_foundation_exam_plan.v1"

    def __post_init__(self) -> None:
        if len(self.episodes) < 8:
            raise ValueError("OpenTrack foundation exam requires at least eight episodes")
        episode_ids = tuple(item.episode_id for item in self.episodes)
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("OpenTrack foundation episode ids must be unique")
        suite_ids = {item.suite_id for item in self.episodes}
        if not {"retention", "acquisition"}.issubset(suite_ids):
            raise ValueError("exam requires both retention and acquisition suites")
        if self.physics_backend != "mujoco_cpu":
            raise ValueError("OpenTrack physical truth must use CPU MuJoCo")
        if self.control_dt_sec != 0.02 or self.sim_dt_sec != 0.002:
            raise ValueError("OpenTrack exam timing must match the sealed 50 Hz controller")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_command_sent:
            raise ValueError("OpenTrack foundation exam must remain SIM_ONLY")

    @property
    def plan_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episodes": [asdict(item) for item in self.episodes],
            "thresholds": asdict(self.thresholds),
            "body_id": self.body_id,
            "physics_backend": self.physics_backend,
            "control_dt_sec": self.control_dt_sec,
            "sim_dt_sec": self.sim_dt_sec,
            "activation_ceiling": self.activation_ceiling,
            "hardware_command_sent": self.hardware_command_sent,
        }


def load_opentrack_exam_plan(path: Path) -> OpenTrackFoundationExamPlan:
    """Load a strict JSON plan without accepting runtime/code payloads."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"episodes"}:
        raise ValueError("OpenTrack exam plan must contain only episodes")
    raw_episodes = payload["episodes"]
    if not isinstance(raw_episodes, list):
        raise ValueError("OpenTrack exam episodes must be a list")
    episodes: list[OpenTrackEpisodeSpec] = []
    expected = {item.name for item in OpenTrackEpisodeSpec.__dataclass_fields__.values()}
    for raw in raw_episodes:
        if not isinstance(raw, dict) or not set(raw).issubset(expected):
            raise ValueError("OpenTrack episode contains unexpected fields")
        episodes.append(OpenTrackEpisodeSpec(**raw))
    return OpenTrackFoundationExamPlan(episodes=tuple(episodes))


def summarize_opentrack_episodes(
    reports: tuple[dict[str, Any], ...],
    *,
    thresholds: FoundationThresholds,
) -> tuple[FoundationMetrics, tuple[str, ...], dict[str, Any]]:
    """Aggregate raw episode traces and fail closed on missing/non-finite data."""

    if len(reports) < 8:
        raise ValueError("foundation summary requires at least eight episode reports")
    required = {
        "suite_id",
        "success",
        "fell",
        "finite_state",
        "joint_squared_error_sum",
        "joint_error_count",
        "keypoint_squared_error_sum",
        "keypoint_error_count",
        "foot_slip_sum_mps",
        "foot_slip_count",
        "minimum_pelvis_height_m",
        "peak_torque_fraction",
        "saturated_control_steps",
        "control_steps",
        "root_angular_speeds_rad_s",
        "joint_jerk_squared_sum",
        "joint_jerk_count",
        "transition_error_rad",
        "recovered_upright",
    }
    if any(set(report) < required for report in reports):
        raise ValueError("OpenTrack episode report is incomplete")
    suites = {str(report["suite_id"]) for report in reports}
    if not {"retention", "acquisition"}.issubset(suites):
        raise ValueError("foundation summary is missing a matched suite")

    numeric_values: list[float] = []
    for report in reports:
        for key in required - {
            "suite_id",
            "success",
            "fell",
            "finite_state",
            "root_angular_speeds_rad_s",
            "recovered_upright",
        }:
            numeric_values.append(float(report[key]))
        numeric_values.extend(float(item) for item in report["root_angular_speeds_rad_s"])
    finite_state = all(bool(report["finite_state"]) for report in reports) and all(
        math.isfinite(value) for value in numeric_values
    )
    joint_sq = sum(float(report["joint_squared_error_sum"]) for report in reports)
    joint_count = sum(int(report["joint_error_count"]) for report in reports)
    keypoint_sq = sum(float(report["keypoint_squared_error_sum"]) for report in reports)
    keypoint_count = sum(int(report["keypoint_error_count"]) for report in reports)
    slip_sum = sum(float(report["foot_slip_sum_mps"]) for report in reports)
    slip_count = sum(int(report["foot_slip_count"]) for report in reports)
    jerk_sq = sum(float(report["joint_jerk_squared_sum"]) for report in reports)
    jerk_count = sum(int(report["joint_jerk_count"]) for report in reports)
    control_steps = sum(int(report["control_steps"]) for report in reports)
    if min(joint_count, keypoint_count, jerk_count, control_steps) <= 0:
        raise ValueError("OpenTrack episode report has empty physical traces")
    angular = np.asarray(
        [float(value) for report in reports for value in report["root_angular_speeds_rad_s"]],
        dtype=np.float64,
    )
    if angular.size == 0:
        raise ValueError("OpenTrack episode report has no root angular trace")

    metrics = FoundationMetrics(
        tracking_success_rate=sum(bool(report["success"]) for report in reports) / len(reports),
        joint_rmse_rad=math.sqrt(joint_sq / joint_count),
        keypoint_mpjpe_m=math.sqrt(keypoint_sq / keypoint_count),
        foot_slip_mps=slip_sum / max(1, slip_count),
        minimum_pelvis_height_m=min(float(report["minimum_pelvis_height_m"]) for report in reports),
        peak_torque_fraction=max(float(report["peak_torque_fraction"]) for report in reports),
        torque_saturation_rate=(
            sum(int(report["saturated_control_steps"]) for report in reports) / control_steps
        ),
        p95_root_angular_speed_rad_s=float(np.percentile(angular, 95)),
        joint_jerk_rms_rad_s3=math.sqrt(jerk_sq / jerk_count),
        transition_error_rad=max(float(report["transition_error_rad"]) for report in reports),
        recovery_rate=sum(bool(report["recovered_upright"]) for report in reports) / len(reports),
        finite_state=finite_state,
    )
    reasons = thresholds.reasons(metrics)
    suite_summary = {
        suite: {
            "episode_count": len(items),
            "success_rate": sum(bool(item["success"]) for item in items) / len(items),
            "fall_count": sum(bool(item["fell"]) for item in items),
            "recovery_rate": sum(bool(item["recovered_upright"]) for item in items) / len(items),
        }
        for suite in sorted(suites)
        if (items := tuple(item for item in reports if item["suite_id"] == suite))
    }
    return metrics, reasons, suite_summary


def run_opentrack_foundation_exam(
    *,
    opentrack_root: Path,
    policy_path: Path,
    source_config_path: Path,
    plan: OpenTrackFoundationExamPlan,
    output_path: Path,
    reference_policy_path: Path | None = None,
    residual_scale: float = 1.0,
) -> dict[str, Any]:
    """Execute a sealed plan in OpenTrack's CPU-MuJoCo play environment."""

    root = opentrack_root.expanduser().resolve()
    policy = policy_path.expanduser().resolve()
    source_config = source_config_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if not root.is_dir() or not policy.is_file() or not source_config.is_file():
        raise FileNotFoundError("OpenTrack checkout, policy, and source config must exist")
    if target == root or root in target.parents:
        raise ValueError("physical evidence must remain outside the OpenTrack source checkout")
    if target.exists():
        raise ValueError("OpenTrack exam refuses to overwrite existing evidence")
    if not math.isfinite(residual_scale) or not 0.0 <= residual_scale <= 1.0:
        raise ValueError("OpenTrack residual scale must be finite and in [0, 1]")
    if reference_policy_path is None and residual_scale != 1.0:
        raise ValueError("OpenTrack residual scaling requires a reference policy")
    os.environ.setdefault("GLI_PATH", str(root))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")

    tmj = importlib.import_module("track_mj")
    importlib.import_module("track_mj.envs.g1_tracking.play.play_g1_env_tracking_general")
    ort = importlib.import_module("onnxruntime")
    constants = importlib.import_module("track_mj.envs.g1_tracking.g1_tracking_constants")
    source_payload = json.loads(source_config.read_text(encoding="utf-8"))
    if not isinstance(source_payload, dict) or not isinstance(
        source_payload.get("env_config"), dict
    ):
        raise ValueError("OpenTrack source config has no environment contract")
    policy_session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
    input_names = tuple(item.name for item in policy_session.get_inputs())
    output_names = tuple(item.name for item in policy_session.get_outputs())
    if set(input_names) not in ({"obs"}, {"obs", "history"}) or output_names != (
        "continuous_actions",
    ):
        raise ValueError("OpenTrack policy IO contract is incompatible")
    uses_history = "history" in input_names
    if uses_history:
        importlib.import_module(
            "track_mj.envs.g1_tracking_adapter.play.play_g1_env_tracking_general"
        )
    reference_policy: Path | None = None
    reference_session: Any | None = None
    if reference_policy_path is not None:
        reference_policy = reference_policy_path.expanduser().resolve()
        if not reference_policy.is_file():
            raise FileNotFoundError("OpenTrack reference policy does not exist")
        reference_session = ort.InferenceSession(
            str(reference_policy), providers=["CPUExecutionProvider"]
        )
        reference_inputs = tuple(item.name for item in reference_session.get_inputs())
        reference_outputs = tuple(item.name for item in reference_session.get_outputs())
        if reference_inputs != ("obs",) or reference_outputs != ("continuous_actions",):
            raise ValueError("OpenTrack reference policy IO contract is incompatible")

    old_cwd = Path.cwd()
    reports: list[dict[str, Any]] = []
    try:
        os.chdir(root)
        for episode in plan.episodes:
            config_key = "tracking_adapter_config" if uses_history else "tracking_config"
            env_cfg = copy.deepcopy(tmj.registry.get("G1TrackingGeneral", config_key).env_config)
            env_cfg.update(source_payload["env_config"])
            env_cfg.reference_traj_config.name = {episode.dataset_id: [episode.motion_id]}
            env_cfg.reference_traj_config.random_start = False
            env_cfg.reference_traj_config.fixed_start_frame = episode.start_frame
            env_key = (
                "tracking_adapter_play_env_class" if uses_history else "tracking_play_env_class"
            )
            env_class = tmj.registry.get("G1TrackingGeneral", env_key)
            env = env_class(
                config=env_cfg,
                play_ref_motion=False,
                use_viewer=False,
                use_renderer=False,
                exp_name="sealed-foundation-exam",
            )
            try:
                reports.append(
                    _run_episode(
                        env=env,
                        policy_session=policy_session,
                        episode=episode,
                        torque_limits=np.asarray(constants.TORQUE_LIMIT, dtype=np.float64),
                        history_len=int(env_cfg.history_len) if uses_history else 0,
                        reference_policy_session=reference_session,
                        residual_scale=residual_scale,
                    )
                )
            finally:
                env.close()
    finally:
        os.chdir(old_cwd)

    metrics, reasons, suite_summary = summarize_opentrack_episodes(
        tuple(reports), thresholds=plan.thresholds
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.opentrack_foundation_exam.v1",
        "plan": plan.to_dict(),
        "plan_hash": plan.plan_hash,
        "policy_hash": _file_hash(policy),
        "source_config_hash": _file_hash(source_config),
        "opentrack_commit": _git_head(root),
        "policy_io": {"inputs": list(input_names), "outputs": list(output_names)},
        "runtime_kind": "adapter_history" if uses_history else "base_policy",
        "episode_reports": reports,
        "suite_summary": suite_summary,
        "aggregate": asdict(metrics),
        "status": "PHYSICS_QUALIFIED" if not reasons else "PHYSICS_UNQUALIFIED",
        "passed": not reasons,
        "reasons": list(reasons),
        "physical_truth": True,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    if reference_policy is not None:
        residual_sq = sum(float(item["residual_action_squared_sum"]) for item in reports)
        raw_residual_sq = sum(float(item["raw_residual_action_squared_sum"]) for item in reports)
        residual_count = sum(int(item["residual_action_count"]) for item in reports)
        if residual_count <= 0:
            raise ValueError("OpenTrack residual audit has no matched action samples")
        report["reference_policy_hash"] = _file_hash(reference_policy)
        report["residual_output_rms"] = math.sqrt(residual_sq / residual_count)
        report["raw_residual_output_rms"] = math.sqrt(raw_residual_sq / residual_count)
        report["residual_scale"] = residual_scale
    report["report_hash"] = hash_json(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return report


def _run_episode(
    *,
    env: Any,
    policy_session: Any,
    episode: OpenTrackEpisodeSpec,
    torque_limits: np.ndarray,
    history_len: int = 0,
    reference_policy_session: Any | None = None,
    residual_scale: float = 1.0,
) -> dict[str, Any]:
    state = env.reset()
    available = int(env.th.len_trajectory(0)) - episode.start_frame - 2
    steps = min(episode.max_steps, available)
    if steps < 50:
        raise ValueError(f"episode {episode.episode_id} has too few available frames")
    joint_sq = 0.0
    joint_count = 0
    keypoint_sq = 0.0
    keypoint_count = 0
    slip_sum = 0.0
    slip_count = 0
    minimum_pelvis = math.inf
    peak_torque_fraction = 0.0
    saturated = 0
    angular: list[float] = []
    jerk_sq = 0.0
    jerk_count = 0
    previous_velocity = np.asarray(env.mj_data.qvel[6:], dtype=np.float64).copy()
    previous_acceleration = np.zeros_like(previous_velocity)
    fell = False
    finite_state = True
    final_error = math.inf
    residual_sq = 0.0
    raw_residual_sq = 0.0
    residual_count = 0
    for _ in range(steps):
        onnx_input: dict[str, np.ndarray] = {
            "obs": np.asarray(state.obs["state"], dtype=np.float32).reshape(1, -1)
        }
        if history_len:
            history = np.asarray(state.obs["history_state"], dtype=np.float32)
            if history.size % history_len:
                raise ValueError("OpenTrack history observation is not divisible by history_len")
            history_matrix = history.reshape(history_len, -1).swapaxes(-1, -2)
            onnx_input["history"] = history_matrix.reshape(
                1, history_matrix.shape[0], history_matrix.shape[1]
            )
        action = policy_session.run(["continuous_actions"], onnx_input)[0][0]
        if reference_policy_session is not None:
            reference_action = reference_policy_session.run(
                ["continuous_actions"], {"obs": onnx_input["obs"]}
            )[0][0]
            residual = np.asarray(action, dtype=np.float64) - np.asarray(
                reference_action, dtype=np.float64
            )
            raw_residual_sq += float(np.dot(residual, residual))
            applied_residual = residual_scale * residual
            residual_sq += float(np.dot(applied_residual, applied_residual))
            residual_count += int(residual.size)
            action = np.asarray(reference_action, dtype=np.float64) + applied_residual
        state = env.step(state, action)
        qpos = np.asarray(env.mj_data.qpos, dtype=np.float64)
        qvel = np.asarray(env.mj_data.qvel, dtype=np.float64)
        finite = bool(np.isfinite(qpos).all() and np.isfinite(qvel).all())
        finite_state = finite_state and finite
        if not finite:
            break
        reference_qpos = np.asarray(env.ref_mj_data.qpos, dtype=np.float64)
        error = qpos[7:] - reference_qpos[7:]
        joint_sq += float(np.dot(error, error))
        joint_count += int(error.size)
        final_error = float(np.sqrt(np.mean(np.square(error))))
        simulated_keypoints = np.asarray(
            env.mj_data.xpos[env.valid_body_ids], dtype=np.float64
        ) - np.asarray(env.mj_data.xpos[env.body_id_pelvis], dtype=np.float64)
        reference_keypoints = np.asarray(
            env.ref_mj_data.xpos[env.valid_body_ids], dtype=np.float64
        ) - np.asarray(env.ref_mj_data.xpos[env.body_id_pelvis], dtype=np.float64)
        body_error = simulated_keypoints - reference_keypoints
        keypoint_sq += float(np.sum(np.square(body_error)))
        keypoint_count += int(body_error.shape[0])
        for body_id in env.feet_ids:
            if env.mj_data.xpos[body_id, 2] < 0.14:
                slip_sum += float(np.linalg.norm(env.mj_data.cvel[body_id, 3:5]))
                slip_count += 1
        fractions = np.abs(np.asarray(env.mj_data.ctrl, dtype=np.float64)) / torque_limits
        peak_torque_fraction = max(peak_torque_fraction, float(np.max(fractions)))
        saturated += int(bool(np.any(fractions >= 0.999)))
        angular.append(float(np.linalg.norm(qvel[3:6])))
        velocity = qvel[6:].copy()
        acceleration = (velocity - previous_velocity) / env.dt
        jerk = (acceleration - previous_acceleration) / env.dt
        jerk_sq += float(np.dot(jerk, jerk))
        jerk_count += int(jerk.size)
        previous_velocity = velocity
        previous_acceleration = acceleration
        minimum_pelvis = min(minimum_pelvis, float(qpos[2]))
        quaternion = qpos[3:7]
        upright = 2.0 * (quaternion[0] ** 2 + quaternion[3] ** 2) - 1.0
        fell = bool(fell or qpos[2] < 0.45 or upright < 0.35)
    qpos = np.asarray(env.mj_data.qpos, dtype=np.float64)
    quaternion = qpos[3:7]
    final_upright = 2.0 * (quaternion[0] ** 2 + quaternion[3] ** 2) - 1.0
    recovered = bool(
        finite_state
        and not fell
        and qpos[2] >= 0.60
        and final_upright >= 0.75
        and final_error <= 0.35
    )
    joint_rmse = math.sqrt(joint_sq / max(1, joint_count))
    keypoint_mpjpe = math.sqrt(keypoint_sq / max(1, keypoint_count))
    success = bool(finite_state and not fell and joint_rmse <= 0.35 and keypoint_mpjpe <= 0.12)
    return {
        "episode_id": episode.episode_id,
        "suite_id": episode.suite_id,
        "dataset_id": episode.dataset_id,
        "motion_id": episode.motion_id,
        "source_hash": episode.source_hash,
        "critical": episode.critical,
        "executed_steps": min(steps, len(angular)),
        "success": success,
        "fell": fell,
        "finite_state": finite_state,
        "joint_rmse_rad": joint_rmse,
        "joint_squared_error_sum": joint_sq,
        "joint_error_count": joint_count,
        "keypoint_mpjpe_m": keypoint_mpjpe,
        "keypoint_squared_error_sum": keypoint_sq,
        "keypoint_error_count": keypoint_count,
        "foot_slip_sum_mps": slip_sum,
        "foot_slip_count": slip_count,
        "minimum_pelvis_height_m": minimum_pelvis,
        "peak_torque_fraction": peak_torque_fraction,
        "saturated_control_steps": saturated,
        "control_steps": len(angular),
        "root_angular_speeds_rad_s": angular,
        "joint_jerk_squared_sum": jerk_sq,
        "joint_jerk_count": jerk_count,
        "transition_error_rad": final_error,
        "recovered_upright": recovered,
        "residual_action_squared_sum": residual_sq,
        "raw_residual_action_squared_sum": raw_residual_sq,
        "residual_action_count": residual_count,
    }


def sanitize_opentrack_adapter_export_config(
    *, source_path: Path, output_path: Path
) -> dict[str, Any]:
    """Create a content-addressed export view for OpenTrack adapter checkpoints.

    OpenTrack serializes Python callbacks into strings in ``config.json``.  Its
    adapter exporter then writes those strings back into strongly typed
    callback fields, which fails before loading the network.  The callbacks do
    not describe network structure and are already removed by OpenTrack's own
    training restore helper.  This bridge removes only that allow-listed set,
    preserves every numerical/network field, and never overwrites the source.
    """

    source = source_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("OpenTrack adapter config does not exist")
    if target.exists() or target == source:
        raise ValueError("sanitized OpenTrack export config requires a new output path")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("policy_config"), dict):
        raise ValueError("OpenTrack adapter config has no policy_config")
    policy_config = payload["policy_config"]
    removed: dict[str, Any] = {}
    for key in ("progress_fn", "randomization_fn", "wrap_env_fn"):
        if key in policy_config:
            removed[key] = policy_config.pop(key)
    if not removed:
        raise ValueError("OpenTrack adapter config has no serialized callback fields")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return {
        "schema_version": "rosclaw_soccer.opentrack_export_sanitization.v1",
        "source_hash": _file_hash(source),
        "output_hash": _file_hash(target),
        "removed_fields": sorted(removed),
        "removed_value_types": {key: type(value).__name__ for key, value in removed.items()},
        "network_fields_changed": False,
    }


def preprocess_opentrack_reference_fast(
    *,
    opentrack_root: Path,
    source_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Complete one OpenTrack trajectory with direct CPU forward kinematics.

    OpenTrack's reference implementation routes every frame through a generic
    play callback.  The resulting labels are useful, but that path also asks
    JAX to stage scalar trajectory bookkeeping on every CPU frame.  This
    equivalent bounded bridge keeps interpolation and model filtering owned by
    OpenTrack, then batches only the deterministic MuJoCo forward-kinematics
    recording loop.  The source archive is never overwritten.
    """

    root = opentrack_root.expanduser().resolve()
    source = source_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if not root.is_dir() or not source.is_file():
        raise FileNotFoundError("OpenTrack checkout and source trajectory must exist")
    if target.exists() or target.suffix != ".npz":
        raise ValueError("fast OpenTrack preprocessing requires a new NPZ output")
    os.environ.setdefault("GLI_PATH", str(root))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    mujoco = importlib.import_module("mujoco")
    constants = importlib.import_module("track_mj.envs.g1_tracking.g1_tracking_constants")
    traj_class = importlib.import_module("track_mj.utils.dataset.traj_class")
    traj_handler = importlib.import_module("track_mj.utils.dataset.traj_handler")
    traj_process = importlib.import_module("track_mj.utils.dataset.traj_process")
    model = mujoco.MjModel.from_xml_path(str(constants.task_to_xml("flat_terrain")))
    data = mujoco.MjData(model)
    trajectory = traj_class.Trajectory.load(source, backend=np)
    if trajectory.data.n_trajectories != 1:
        raise ValueError("fast OpenTrack preprocessing accepts one trajectory per archive")
    if not math.isclose(float(trajectory.info.frequency), 50.0):
        data_50hz, info_50hz = traj_class.interpolate_trajectories(
            trajectory.data,
            trajectory.info,
            50.0,
            backend=np,
        )
        trajectory = traj_class.Trajectory(
            info=info_50hz,
            data=data_50hz.to_numpy(),
            transitions=trajectory.transitions,
        )
    handler = traj_handler.TrajectoryHandler(
        model=model,
        traj=trajectory,
        control_dt=0.02,
        random_start=False,
        fixed_start_conf=(0, 0),
        warn=False,
    )
    handler.to_numpy()
    trajectory = handler.traj
    sample_count = int(trajectory.data.qpos.shape[0])
    recorder = traj_process.ExtendTrajData(
        SimpleNamespace(_mj_model=model),
        n_samples=sample_count,
        model=model,
    )
    initial_xy = np.asarray(trajectory.data.qpos[0, :2], dtype=np.float64)
    for index in range(sample_count):
        qpos = np.asarray(trajectory.data.qpos[index], dtype=np.float64).copy()
        qpos[:2] -= initial_xy
        data.qpos[:] = qpos
        data.qvel[:] = np.asarray(trajectory.data.qvel[index], dtype=np.float64)
        mujoco.mj_forward(model, data)
        recorder.recorder["xpos"][index] = data.xpos[recorder.b_ids]
        recorder.recorder["xquat"][index] = data.xquat[recorder.b_ids]
        recorder.recorder["cvel"][index] = data.cvel[recorder.b_ids]
        recorder.recorder["subtree_com"][index] = data.subtree_com[recorder.b_ids]
        recorder.recorder["site_xpos"][index] = data.site_xpos[recorder.s_ids]
        recorder.recorder["site_xmat"][index] = data.site_xmat[recorder.s_ids]
        recorder.recorder["qpos"][index] = data.qpos
        recorder.recorder["qvel"][index] = data.qvel
    recorder.current_length = sample_count
    completed_data, completed_info = recorder.extend_trajectory_data(
        trajectory.data, trajectory.info
    )
    completed = traj_class.Trajectory(
        info=completed_info,
        data=completed_data.to_numpy(),
        transitions=trajectory.transitions,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    completed.save(str(target))
    return {
        "schema_version": "rosclaw_soccer.opentrack_fast_preprocess.v1",
        "source_hash": _file_hash(source),
        "output_hash": _file_hash(target),
        "frame_count": sample_count,
        "frequency_hz": float(completed.info.frequency),
        "completion_backend": "mujoco_cpu_forward_kinematics",
        "opentrack_commit": _git_head(root),
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _git_head(root: Path) -> str:
    head = root / ".git" / "HEAD"
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = root / ".git" / value.removeprefix("ref: ")
        value = ref.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("OpenTrack checkout must have a readable pinned commit")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sealed OpenTrack foundation exam")
    parser.add_argument("--opentrack-root", required=True, type=Path)
    parser.add_argument("--policy-path", required=True, type=Path)
    parser.add_argument("--source-config-path", required=True, type=Path)
    parser.add_argument("--plan-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--reference-policy-path", type=Path)
    parser.add_argument("--residual-scale", default=1.0, type=float)
    args = parser.parse_args()
    report = run_opentrack_foundation_exam(
        opentrack_root=args.opentrack_root,
        policy_path=args.policy_path,
        source_config_path=args.source_config_path,
        plan=load_opentrack_exam_plan(args.plan_path),
        output_path=args.output_path,
        reference_policy_path=args.reference_policy_path,
        residual_scale=args.residual_scale,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()


__all__ = [
    "OpenTrackEpisodeSpec",
    "OpenTrackFoundationExamPlan",
    "load_opentrack_exam_plan",
    "preprocess_opentrack_reference_fast",
    "run_opentrack_foundation_exam",
    "sanitize_opentrack_adapter_export_config",
    "summarize_opentrack_episodes",
]
