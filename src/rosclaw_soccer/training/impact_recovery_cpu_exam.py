"""Independent CPU-MuJoCo exam for the learned impact-recovery residual.

The four-GPU MJX result is only a candidate.  This module reconstructs the
same 29-DoF controller around its frozen motion memory in ordinary MuJoCo,
runs a content-bound paired suite, and compares the learned residual against
the zero-residual incumbent.  Passing this exam grants no deployment or
hardware authority; it only allows the candidate to enter a later team-level
full-chain simulation exam.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.training.agents.ppo import checkpoint
from rosclaw.continual.champion_registry import (
    DominanceMetricRole,
    PairedDominanceEvidence,
    PairedDominanceMetric,
)

from rosclaw_soccer.providers.g1.asset_qualification import g1_body_hash
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.impact_recovery_mjx import (
    ImpactRecoveryMJXConfig,
    _load_curriculum_arrays,
    _tree_hash,
    validate_impact_recovery_mjx_evaluation_report,
    validate_impact_recovery_mjx_report,
)
from rosclaw_soccer.training.opentrack_recovery_mjx_ppo import (
    _make_recovery_ppo_networks,
)
from rosclaw_soccer.training.recovery_mjx import (
    _KDS,
    _KPS,
    _TORQUE_LIMIT,
    _atomic_json,
    compiled_mujoco_model_contract,
)

Population = Literal["acquisition", "retention"]
Policy = Callable[[jax.Array, jax.Array], tuple[jax.Array, dict[str, Any]]]

_G1_QPOS_WIDTH = 36
_G1_QVEL_WIDTH = 35
_JOINT_COUNT = 29
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class ImpactRecoveryCPUExamConfig:
    """Fail-closed paired CPU physics qualification contract."""

    episodes_per_seed: int = 32
    seeds: tuple[int, ...] = (67_151, 67_152, 67_153, 67_154)
    minimum_acquisition_gain_count: int = 8
    maximum_retention_drop_count: int = 4
    simulation_dt_sec: float = 0.002
    substeps: int = 10
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.impact_recovery_cpu_exam_config.v1"

    def __post_init__(self) -> None:
        if (
            not 4 <= self.episodes_per_seed <= 256
            or not 2 <= len(self.seeds) <= 8
            or len(set(self.seeds)) != len(self.seeds)
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31
                for seed in self.seeds
            )
            or not 1 <= self.minimum_acquisition_gain_count <= self.episode_count
            or not 0 <= self.maximum_retention_drop_count <= self.episode_count
            or not math.isfinite(self.simulation_dt_sec)
            or not 0.0005 <= self.simulation_dt_sec <= 0.01
            or not 1 <= self.substeps <= 50
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("impact-recovery CPU exam config is invalid")

    @property
    def episode_count(self) -> int:
        return self.episodes_per_seed * len(self.seeds)

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _rotation_matrix(quaternion_wxyz: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    w, x, y, z = quaternion_wxyz
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _upright(quaternion_wxyz: np.ndarray[Any, Any]) -> float:
    _, x, y, _ = quaternion_wxyz
    return float(1.0 - 2.0 * (x * x + y * y))


def _heading_error(quaternion_wxyz: np.ndarray[Any, Any], desired: float) -> float:
    w, x, y, z = quaternion_wxyz
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.atan2(math.sin(desired - yaw), math.cos(desired - yaw))


def _memory_blend(step: int, acquisition: bool, config: ImpactRecoveryMJXConfig) -> float:
    blend = float(np.clip(step / config.memory_blend_steps, 0.0, 1.0))
    if config.retention_memory_mode == "DIRECT_REPLAY" and not acquisition:
        return 1.0
    return blend


def _teacher_novelty_gate(
    qpos: np.ndarray[Any, Any],
    qvel: np.ndarray[Any, Any],
    reference_qpos: np.ndarray[Any, Any],
    reference_qvel: np.ndarray[Any, Any],
    config: ImpactRecoveryMJXConfig,
) -> float:
    gravity = _rotation_matrix(qpos[3:7]).T @ np.asarray((0.0, 0.0, -1.0))
    reference_gravity = _rotation_matrix(reference_qpos[3:7]).T @ np.asarray((0.0, 0.0, -1.0))
    components = np.asarray(
        (
            abs(qpos[2] - reference_qpos[2]) / 0.10,
            np.linalg.norm(gravity - reference_gravity) / 0.25,
            np.linalg.norm(qvel[:3] - reference_qvel[:3]),
            np.linalg.norm(qvel[3:6] - reference_qvel[3:6]) / 2.0,
            np.sqrt(np.mean(np.square((qpos[7:36] - reference_qpos[7:36]) / 0.50))),
            np.sqrt(np.mean(np.square((qvel[6:35] - reference_qvel[6:35]) / 3.0))),
        ),
        dtype=np.float64,
    )
    novelty = float(np.sqrt(np.mean(np.square(components))))
    return float(
        np.clip(
            (novelty - config.novelty_gate_lower)
            / (config.novelty_gate_upper - config.novelty_gate_lower),
            0.0,
            1.0,
        )
    )


def _kinematics(
    data: mujoco.MjData,
    left_foot_site: int,
    right_foot_site: int,
    desired_heading: float,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], float, np.ndarray[Any, Any]]:
    rotation = _rotation_matrix(np.asarray(data.qpos[3:7]))
    return (
        rotation.T @ np.asarray(data.qvel[:3]),
        rotation.T @ np.asarray(data.qvel[3:6]),
        _heading_error(np.asarray(data.qpos[3:7]), desired_heading),
        np.asarray(
            (data.site_xpos[left_foot_site, 2], data.site_xpos[right_foot_site, 2]),
            dtype=np.float64,
        ),
    )


def _frame(
    *,
    data: mujoco.MjData,
    default_joint_position: np.ndarray[Any, Any],
    last_target: np.ndarray[Any, Any],
    memory_target: np.ndarray[Any, Any],
    kp: np.ndarray[Any, Any],
    kd: np.ndarray[Any, Any],
    left_foot_site: int,
    right_foot_site: int,
    desired_heading: float,
    config: ImpactRecoveryMJXConfig,
) -> np.ndarray[Any, Any]:
    rotation = _rotation_matrix(np.asarray(data.qpos[3:7]))
    body_linear, body_angular, heading, foot_heights = _kinematics(
        data, left_foot_site, right_foot_site, desired_heading
    )
    foot_features = np.concatenate(
        (
            np.clip((foot_heights - 0.035) / 0.10, 0.0, 1.0),
            (foot_heights <= config.ready_foot_height_m).astype(np.float64),
        )
    )
    values = [
        rotation.T @ np.asarray((0.0, 0.0, -1.0)),
        np.clip(body_linear / 2.0, -1.0, 1.0),
        np.clip(body_angular / 3.0, -1.0, 1.0),
        np.asarray((math.sin(heading), math.cos(heading))),
        np.asarray(data.qpos[7:36]) - default_joint_position,
        np.asarray(data.qvel[6:35]) * 0.05,
        last_target,
        memory_target - np.asarray(data.qpos[7:36]),
        foot_features,
    ]
    if config.gain_memory_mode == "DYNAMIC":
        values.extend((np.clip(kp / 300.0, 0.0, 1.5), np.clip(kd / 10.0, 0.0, 1.5)))
    return cast(
        np.ndarray[Any, Any],
        np.nan_to_num(
            np.concatenate(values).astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
    )


def _build_suite(
    *,
    model: mujoco.MjModel,
    arrays: dict[str, np.ndarray[Any, Any]],
    config: ImpactRecoveryMJXConfig,
    exam_config: ImpactRecoveryCPUExamConfig,
) -> dict[str, np.ndarray[Any, Any]]:
    source_succeeded = np.asarray(arrays["source_succeeded"], dtype=np.bool_)
    indexes = {
        "acquisition": np.flatnonzero(~source_succeeded),
        "retention": np.flatnonzero(source_succeeded),
    }
    if any(values.size == 0 for values in indexes.values()):
        raise ValueError("impact-recovery CPU exam requires both populations")
    rows: list[int] = []
    populations: list[int] = []
    seeds: list[int] = []
    episode_indexes: list[int] = []
    qpos_rows: list[np.ndarray[Any, Any]] = []
    qvel_rows: list[np.ndarray[Any, Any]] = []
    lower = np.asarray(model.jnt_range[1:30, 0])
    upper = np.asarray(model.jnt_range[1:30, 1])
    for population_id, population in enumerate(("acquisition", "retention")):
        population_rows = indexes[population]
        for seed in exam_config.seeds:
            rng = np.random.default_rng(seed + population_id * 1_000_003)
            for episode_index in range(exam_config.episodes_per_seed):
                row = int(rng.choice(population_rows))
                qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
                qpos[:_G1_QPOS_WIDTH] = arrays["qpos"][row]
                qpos[7:36] = np.clip(
                    qpos[7:36]
                    + rng.uniform(
                        -config.joint_position_noise_rad,
                        config.joint_position_noise_rad,
                        _JOINT_COUNT,
                    ),
                    lower,
                    upper,
                )
                qvel = np.zeros(model.nv, dtype=np.float64)
                qvel[:_G1_QVEL_WIDTH] = arrays["qvel"][row]
                velocity_noise = rng.uniform(-1.0, 1.0, _G1_QVEL_WIDTH)
                qvel[:3] += velocity_noise[:3] * config.root_linear_velocity_noise_mps
                qvel[3:6] += velocity_noise[3:6] * config.root_angular_velocity_noise_rad_s
                qvel[6:35] += velocity_noise[6:35] * config.joint_velocity_noise_rad_s
                rows.append(row)
                populations.append(population_id)
                seeds.append(seed)
                episode_indexes.append(episode_index)
                qpos_rows.append(qpos)
                qvel_rows.append(qvel)
    return {
        "curriculum_row": np.asarray(rows, dtype=np.int32),
        "population": np.asarray(populations, dtype=np.int8),
        "seed": np.asarray(seeds, dtype=np.int64),
        "episode_index": np.asarray(episode_indexes, dtype=np.int32),
        "initial_qpos": np.stack(qpos_rows),
        "initial_qvel": np.stack(qvel_rows),
    }


def _gain_targets(
    arrays: dict[str, np.ndarray[Any, Any]],
    row: int,
    step: int,
    acquisition: bool,
    initial_target: np.ndarray[Any, Any],
    initial_kp: np.ndarray[Any, Any],
    initial_kd: np.ndarray[Any, Any],
    config: ImpactRecoveryMJXConfig,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    memory_step = min(step, arrays["frozen_memory_target_rad"].shape[1] - 1)
    blend = _memory_blend(memory_step, acquisition, config)
    memory_target = arrays["frozen_memory_target_rad"][row, memory_step]
    baseline = (1.0 - blend) * initial_target + blend * memory_target
    if config.gain_memory_mode == "DYNAMIC":
        kp = (1.0 - blend) * initial_kp + blend * arrays["frozen_memory_kp"][row, memory_step]
        kd = (1.0 - blend) * initial_kd + blend * arrays["frozen_memory_kd"][row, memory_step]
    else:
        kp, kd = np.asarray(_KPS), np.asarray(_KDS)
    return baseline, kp, kd


def _run_population(
    *,
    population: Population,
    candidate: bool,
    suite: dict[str, np.ndarray[Any, Any]],
    model: mujoco.MjModel,
    arrays: dict[str, np.ndarray[Any, Any]],
    desired_heading: float,
    training_config: ImpactRecoveryMJXConfig,
    exam_config: ImpactRecoveryCPUExamConfig,
    policy: Policy,
) -> list[dict[str, Any]]:
    population_id = 0 if population == "acquisition" else 1
    selected = np.flatnonzero(suite["population"] == population_id)
    left_site = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot"))
    right_site = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot"))
    if left_site < 0 or right_site < 0:
        raise ValueError("impact-recovery CPU exam foot sites are unavailable")
    lower = np.asarray(model.jnt_range[1:30, 0])
    upper = np.asarray(model.jnt_range[1:30, 1])
    default = np.asarray(model.qpos0[7:36])
    residual_limits = training_config.residual_limits_rad.astype(np.float64)
    data_rows: list[mujoco.MjData] = []
    states: list[dict[str, Any]] = []
    for suite_index in selected:
        row = int(suite["curriculum_row"][suite_index])
        data = mujoco.MjData(model)
        data.qpos[:] = suite["initial_qpos"][suite_index]
        data.qvel[:] = suite["initial_qvel"][suite_index]
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        target = np.asarray(arrays["initial_motor_target_rad"][row], dtype=np.float64)
        initial_kp = (
            np.asarray(arrays["initial_kp"][row], dtype=np.float64)
            if training_config.gain_memory_mode == "DYNAMIC"
            else np.asarray(_KPS, dtype=np.float64)
        )
        initial_kd = (
            np.asarray(arrays["initial_kd"][row], dtype=np.float64)
            if training_config.gain_memory_mode == "DYNAMIC"
            else np.asarray(_KDS, dtype=np.float64)
        )
        initial_frame = _frame(
            data=data,
            default_joint_position=default,
            last_target=target,
            memory_target=target,
            kp=initial_kp,
            kd=initial_kd,
            left_foot_site=left_site,
            right_foot_site=right_site,
            desired_heading=desired_heading,
            config=training_config,
        )
        data_rows.append(data)
        states.append(
            {
                "suite_index": int(suite_index),
                "row": row,
                "done": False,
                "success": False,
                "terminal_reason": "TIME_LIMIT",
                "steps": 0,
                "stable_streak": 0,
                "maximum_stable_streak": 0,
                "last_target": target.copy(),
                "initial_target": target.copy(),
                "initial_kp": initial_kp,
                "initial_kd": initial_kd,
                "history": np.repeat(
                    initial_frame[None, :], training_config.history_frames, axis=0
                ),
                "initial_xy": np.asarray(data.qpos[:2]).copy(),
                "minimum_pelvis_height_m": float(data.qpos[2]),
                "maximum_drift_m": 0.0,
                "maximum_linear_speed_mps": 0.0,
                "maximum_angular_speed_rad_s": 0.0,
                "torque_saturation_sum": 0.0,
                "residual_rms_sum": 0.0,
                "residual_gate_sum": 0.0,
            }
        )
    policy_key = jax.random.PRNGKey(exam_config.seeds[0] + (1 if candidate else 0))
    for step in range(training_config.episode_length):
        if all(bool(state["done"]) for state in states):
            break
        observations = np.stack([state["history"].reshape(-1) for state in states])
        if candidate:
            actions, _ = policy(jnp.asarray(observations), policy_key)
            action_rows = np.asarray(actions, dtype=np.float64)
        else:
            action_rows = np.zeros((len(states), _JOINT_COUNT), dtype=np.float64)
        if action_rows.shape != (len(states), _JOINT_COUNT) or not np.all(np.isfinite(action_rows)):
            raise ValueError("impact-recovery CPU policy emitted an invalid action")
        for state_index, (state, data) in enumerate(zip(states, data_rows, strict=True)):
            if state["done"]:
                continue
            row = int(state["row"])
            acquisition = population == "acquisition"
            baseline, kp, kd = _gain_targets(
                arrays,
                row,
                step,
                acquisition,
                state["initial_target"],
                state["initial_kp"],
                state["initial_kd"],
                training_config,
            )
            memory_step = min(step, arrays["frozen_memory_target_rad"].shape[1] - 1)
            if training_config.residual_gate_mode == "TEACHER_NOVELTY":
                gate = _teacher_novelty_gate(
                    np.asarray(data.qpos),
                    np.asarray(data.qvel),
                    arrays["frozen_memory_qpos"][row, memory_step],
                    arrays["frozen_memory_qvel"][row, memory_step],
                    training_config,
                )
            else:
                gate = 1.0
            if (
                training_config.residual_authority_steps > 0
                and step >= training_config.residual_authority_steps
            ):
                gate = 0.0
            bounded_action = np.clip(action_rows[state_index], -1.0, 1.0) * gate
            desired_target = np.clip(baseline + bounded_action * residual_limits, lower, upper)
            target_delta = np.clip(
                desired_target - state["last_target"],
                -training_config.maximum_target_step_rad,
                training_config.maximum_target_step_rad,
            )
            motor_target = state["last_target"] + target_delta
            saturation = 0.0
            for _ in range(exam_config.substeps):
                raw_torque = kp * (motor_target - data.qpos[7:36]) - kd * data.qvel[6:35]
                saturation += float(np.mean(np.abs(raw_torque) > _TORQUE_LIMIT))
                data.ctrl[:] = np.clip(raw_torque, -_TORQUE_LIMIT, _TORQUE_LIMIT)
                mujoco.mj_step(model, data)
            finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
            body_linear, body_angular, heading, foot_heights = _kinematics(
                data, left_site, right_site, desired_heading
            )
            upright = _upright(np.asarray(data.qpos[3:7]))
            linear_speed = float(np.linalg.norm(body_linear))
            angular_speed = float(np.linalg.norm(body_angular))
            bilateral_support = bool(np.all(foot_heights <= training_config.ready_foot_height_m))
            balance_ready = bool(
                data.qpos[2] >= training_config.ready_pelvis_height_m
                and upright >= training_config.ready_upright_projection
                and linear_speed <= training_config.ready_linear_speed_mps
                and angular_speed <= training_config.ready_angular_speed_rad_s
                and bilateral_support
            )
            ready = bool(
                balance_ready
                and (
                    training_config.learning_stage != "GOAL_READY"
                    or abs(heading) <= training_config.ready_heading_error_rad
                )
            )
            state["stable_streak"] = int(state["stable_streak"]) + 1 if ready else 0
            state["maximum_stable_streak"] = max(
                int(state["maximum_stable_streak"]), int(state["stable_streak"])
            )
            success = int(state["stable_streak"]) >= training_config.success_stable_steps
            fallen = bool(data.qpos[2] < 0.42 or upright < 0.20)
            state["steps"] = step + 1
            state["last_target"] = motor_target
            state["minimum_pelvis_height_m"] = min(
                float(state["minimum_pelvis_height_m"]), float(data.qpos[2])
            )
            state["maximum_drift_m"] = max(
                float(state["maximum_drift_m"]),
                float(np.linalg.norm(data.qpos[:2] - state["initial_xy"])),
            )
            state["maximum_linear_speed_mps"] = max(
                float(state["maximum_linear_speed_mps"]), linear_speed
            )
            state["maximum_angular_speed_rad_s"] = max(
                float(state["maximum_angular_speed_rad_s"]), angular_speed
            )
            state["torque_saturation_sum"] = float(state["torque_saturation_sum"]) + (
                saturation / exam_config.substeps
            )
            state["residual_rms_sum"] = float(state["residual_rms_sum"]) + float(
                np.sqrt(np.mean(np.square(bounded_action)))
            )
            state["residual_gate_sum"] = float(state["residual_gate_sum"]) + gate
            if success or fallen or not finite:
                state["done"] = True
                state["success"] = success
                state["terminal_reason"] = (
                    "SUCCESS" if success else "NONFINITE" if not finite else "FALLEN"
                )
                continue
            next_step = min(step + 1, arrays["frozen_memory_target_rad"].shape[1] - 1)
            next_baseline, next_kp, next_kd = _gain_targets(
                arrays,
                row,
                next_step,
                acquisition,
                state["initial_target"],
                state["initial_kp"],
                state["initial_kd"],
                training_config,
            )
            next_frame = _frame(
                data=data,
                default_joint_position=default,
                last_target=motor_target,
                memory_target=next_baseline,
                kp=next_kp,
                kd=next_kd,
                left_foot_site=left_site,
                right_foot_site=right_site,
                desired_heading=desired_heading,
                config=training_config,
            )
            state["history"] = np.concatenate((state["history"][1:], next_frame[None, :]))
    rows: list[dict[str, Any]] = []
    for state, data in zip(states, data_rows, strict=True):
        steps = max(1, int(state["steps"]))
        suite_index = int(state["suite_index"])
        rows.append(
            {
                "seed": int(suite["seed"][suite_index]),
                "episode_index": int(suite["episode_index"][suite_index]),
                "curriculum_row": int(state["row"]),
                "success": bool(state["success"]),
                "terminal_reason": str(state["terminal_reason"]),
                "control_steps": int(state["steps"]),
                "maximum_stable_streak": int(state["maximum_stable_streak"]),
                "minimum_pelvis_height_m": float(state["minimum_pelvis_height_m"]),
                "maximum_drift_m": float(state["maximum_drift_m"]),
                "maximum_linear_speed_mps": float(state["maximum_linear_speed_mps"]),
                "maximum_angular_speed_rad_s": float(state["maximum_angular_speed_rad_s"]),
                "mean_torque_saturation_fraction": float(state["torque_saturation_sum"]) / steps,
                "mean_residual_rms": float(state["residual_rms_sum"]) / steps,
                "mean_residual_gate": float(state["residual_gate_sum"]) / steps,
                "final_pelvis_height_m": float(data.qpos[2]),
                "final_upright_projection": _upright(np.asarray(data.qpos[3:7])),
            }
        )
    return rows


def _population_summary(
    incumbent: list[dict[str, Any]], challenger: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(incumbent) != len(challenger) or not incumbent:
        raise ValueError("impact-recovery CPU paired results are incomplete")
    paired = {
        "both_success": 0,
        "challenger_only_success": 0,
        "incumbent_only_success": 0,
        "both_failure": 0,
    }
    for incumbent_row, challenger_row in zip(incumbent, challenger, strict=True):
        identity = ("seed", "episode_index", "curriculum_row")
        if any(incumbent_row[name] != challenger_row[name] for name in identity):
            raise ValueError("impact-recovery CPU scenarios are not paired")
        incumbent_success = bool(incumbent_row["success"])
        challenger_success = bool(challenger_row["success"])
        if incumbent_success and challenger_success:
            paired["both_success"] += 1
        elif challenger_success:
            paired["challenger_only_success"] += 1
        elif incumbent_success:
            paired["incumbent_only_success"] += 1
        else:
            paired["both_failure"] += 1
    incumbent_success_count = sum(bool(row["success"]) for row in incumbent)
    challenger_success_count = sum(bool(row["success"]) for row in challenger)
    return {
        "episode_count": len(incumbent),
        "incumbent_success_count": incumbent_success_count,
        "challenger_success_count": challenger_success_count,
        "incumbent_success_rate": incumbent_success_count / len(incumbent),
        "challenger_success_rate": challenger_success_count / len(challenger),
        "success_gain_count": challenger_success_count - incumbent_success_count,
        "paired_outcomes": paired,
        "incumbent_episodes": incumbent,
        "challenger_episodes": challenger,
    }


def run_impact_recovery_cpu_exam(
    *,
    asset_root: Path,
    curriculum_manifest_path: Path,
    training_report_path: Path,
    gpu_evaluation_report_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    config: ImpactRecoveryCPUExamConfig | None = None,
) -> dict[str, Any]:
    """Run the paired ordinary-MuJoCo exam and persist replayable evidence."""

    active = config or ImpactRecoveryCPUExamConfig()
    root = asset_root.expanduser().resolve()
    curriculum_path = curriculum_manifest_path.expanduser().resolve()
    training_path = training_report_path.expanduser().resolve()
    evaluation_path = gpu_evaluation_report_path.expanduser().resolve()
    selected_checkpoint = checkpoint_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    model_path = root / "g1_description" / "g1_liao.xml"
    if (
        any(
            not path.is_file()
            for path in (model_path, curriculum_path, training_path, evaluation_path)
        )
        or not selected_checkpoint.is_dir()
        or not (selected_checkpoint / "ppo_network_config.json").is_file()
    ):
        raise FileNotFoundError("impact-recovery CPU exam inputs are incomplete")
    if (
        destination.exists()
        or destination == checkout
        or checkout in destination.parents
        or destination in checkout.parents
    ):
        raise ValueError("impact-recovery CPU evidence must be new and external")
    devices = tuple(jax.devices())
    if not devices or any(getattr(device, "platform", "") != "cpu" for device in devices):
        raise RuntimeError("impact-recovery CPU exam requires CPU-only JAX visibility")
    training = validate_impact_recovery_mjx_report(training_path)
    evaluation = validate_impact_recovery_mjx_evaluation_report(evaluation_path)
    manifest, arrays = _load_curriculum_arrays(curriculum_path)
    selected_hash, selected_rows = _tree_hash(selected_checkpoint)
    checkpoint_root = (training_path.parent / "checkpoints").resolve()
    if checkpoint_root not in selected_checkpoint.parents:
        raise ValueError("impact-recovery CPU checkpoint escaped the training tree")
    selected_prefix = selected_checkpoint.relative_to(checkpoint_root).as_posix() + "/"
    declared_rows = [
        {**row, "path": str(row["path"])[len(selected_prefix) :]}
        for row in cast(list[dict[str, Any]], training["checkpoint_files"])
        if str(row.get("path", "")).startswith(selected_prefix)
    ]
    if (
        selected_rows != declared_rows
        or evaluation.get("selected_checkpoint_files") != selected_rows
        or evaluation.get("selected_checkpoint_hash") != selected_hash
        or evaluation.get("training_report_hash") != training.get("report_hash")
        or evaluation.get("curriculum_manifest_hash") != manifest.get("manifest_hash")
        or training.get("curriculum_manifest_hash") != manifest.get("manifest_hash")
        or training.get("body_hash") != g1_body_hash(root)
        or manifest.get("training_model_hash") != hash_bytes(model_path.read_bytes())
    ):
        raise ValueError("impact-recovery CPU evidence binding changed")
    training_config = ImpactRecoveryMJXConfig(**cast(dict[str, Any], training["config"]))
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = active.simulation_dt_sec
    if (
        model.nq < _G1_QPOS_WIDTH
        or model.nv < _G1_QVEL_WIDTH
        or model.nu != _JOINT_COUNT
        or compiled_mujoco_model_contract(model) != training.get("compiled_model_contract")
    ):
        raise ValueError("impact-recovery CPU compiled model contract changed")
    suite = _build_suite(model=model, arrays=arrays, config=training_config, exam_config=active)
    destination.mkdir(parents=True)
    suite_path = destination / "scenario-suite.npz"
    np.savez_compressed(suite_path, **suite)  # type: ignore[arg-type]
    desired_heading = float(cast(dict[str, Any], manifest["config"])["desired_heading_rad"])
    policy = checkpoint.load_policy(
        selected_checkpoint,
        network_factory=_make_recovery_ppo_networks,
        deterministic=True,
    )
    populations: dict[str, Any] = {}
    for population in ("acquisition", "retention"):
        incumbent = _run_population(
            population=population,
            candidate=False,
            suite=suite,
            model=model,
            arrays=arrays,
            desired_heading=desired_heading,
            training_config=training_config,
            exam_config=active,
            policy=policy,
        )
        challenger = _run_population(
            population=population,
            candidate=True,
            suite=suite,
            model=model,
            arrays=arrays,
            desired_heading=desired_heading,
            training_config=training_config,
            exam_config=active,
            policy=policy,
        )
        populations[population] = _population_summary(incumbent, challenger)
    dominance = PairedDominanceEvidence(
        incumbent_artifact_hash=str(manifest["archive_hash"]),
        challenger_artifact_hash=selected_hash,
        scenario_suite_hash=hash_bytes(suite_path.read_bytes()),
        metrics=(
            PairedDominanceMetric(
                metric_id="acquisition_success_count",
                incumbent_value=float(populations["acquisition"]["incumbent_success_count"]),
                challenger_value=float(populations["acquisition"]["challenger_success_count"]),
                higher_is_better=True,
                role=DominanceMetricRole.OBJECTIVE,
                minimum_improvement=float(active.minimum_acquisition_gain_count),
            ),
            PairedDominanceMetric(
                metric_id="retention_success_count",
                incumbent_value=float(populations["retention"]["incumbent_success_count"]),
                challenger_value=float(populations["retention"]["challenger_success_count"]),
                higher_is_better=True,
                role=DominanceMetricRole.GUARDRAIL,
                maximum_regression=float(active.maximum_retention_drop_count),
            ),
        ),
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_cpu_exam.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "training_report_hash": training["report_hash"],
        "training_report_file_hash": hash_bytes(training_path.read_bytes()),
        "gpu_evaluation_report_hash": evaluation["report_hash"],
        "gpu_evaluation_report_file_hash": hash_bytes(evaluation_path.read_bytes()),
        "curriculum_manifest_hash": manifest["manifest_hash"],
        "curriculum_archive_hash": manifest["archive_hash"],
        "body_hash": manifest["body_hash"],
        "compiled_model_contract": compiled_mujoco_model_contract(model),
        "selected_checkpoint_hash": selected_hash,
        "selected_checkpoint_files": selected_rows,
        "scenario_suite": suite_path.name,
        "scenario_suite_hash": hash_bytes(suite_path.read_bytes()),
        "scenario_count": int(suite["curriculum_row"].shape[0]),
        "scenario_pairing": "IDENTICAL_INITIAL_QPOS_QVEL_AND_CURRICULUM_ROW",
        "incumbent": "CONTENT_BOUND_DYNAMIC_GAIN_MEMORY_WITH_ZERO_RESIDUAL",
        "challenger": "CONTENT_BOUND_DYNAMIC_GAIN_MEMORY_WITH_LEARNED_29_DOF_RESIDUAL",
        "populations": populations,
        "dominance_evidence": dominance.to_dict(),
        "dominance_evidence_hash": dominance.evidence_hash,
        "decision": (
            "CANDIDATE_READY_FOR_TEAM_FULL_CHAIN_EXAM"
            if dominance.promotion_passed
            else "CANDIDATE_ARCHIVED_BY_CPU_MUJOCO"
        ),
        "physics_backend": "CPU_MUJOCO",
        "jax_inference_backend": "CPU",
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "claim_boundary": "Isolated CPU-MuJoCo recovery exam; team full-chain exam still required",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    report_path = destination / "cpu-exam.json"
    _atomic_json(report_path, report)
    return validate_impact_recovery_cpu_exam(report_path)


def validate_impact_recovery_cpu_exam(path: Path) -> dict[str, Any]:
    """Recompute all aggregate and authority fields without loading JAX."""

    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("impact-recovery CPU exam must be an object")
    report = cast(dict[str, Any], payload)
    declared = report.pop("report_hash", None)
    try:
        config_value = report.get("config")
        populations = report.get("populations")
        evidence_value = report.get("dominance_evidence")
        if (
            not isinstance(config_value, dict)
            or not isinstance(populations, dict)
            or not isinstance(evidence_value, dict)
        ):
            raise ValueError("impact-recovery CPU exam is incomplete")
        config = ImpactRecoveryCPUExamConfig(**config_value)
        suite_name = report.get("scenario_suite")
        suite_path = resolved.parent / str(suite_name)
        suite_valid = bool(
            isinstance(suite_name, str)
            and Path(suite_name).name == suite_name
            and suite_path.is_file()
            and hash_bytes(suite_path.read_bytes()) == report.get("scenario_suite_hash")
        )
        populations_valid = set(populations) == {"acquisition", "retention"}
        if populations_valid:
            for population in populations.values():
                if not isinstance(population, dict):
                    populations_valid = False
                    break
                incumbent = population.get("incumbent_episodes")
                challenger = population.get("challenger_episodes")
                if not isinstance(incumbent, list) or not isinstance(challenger, list):
                    populations_valid = False
                    break
                try:
                    expected = _population_summary(incumbent, challenger)
                except (KeyError, TypeError, ValueError):
                    populations_valid = False
                    break
                if population != expected or expected["episode_count"] != config.episode_count:
                    populations_valid = False
                    break
        metric_values = evidence_value.get("metrics")
        metrics = (
            tuple(
                PairedDominanceMetric(
                    metric_id=str(item["metric_id"]),
                    incumbent_value=float(item["incumbent_value"]),
                    challenger_value=float(item["challenger_value"]),
                    higher_is_better=bool(item["higher_is_better"]),
                    role=DominanceMetricRole(str(item["role"])),
                    minimum_improvement=float(item["minimum_improvement"]),
                    maximum_regression=float(item["maximum_regression"]),
                )
                for item in metric_values
                if isinstance(item, dict)
            )
            if isinstance(metric_values, list)
            else ()
        )
        dominance = PairedDominanceEvidence(
            incumbent_artifact_hash=str(evidence_value.get("incumbent_artifact_hash", "")),
            challenger_artifact_hash=str(evidence_value.get("challenger_artifact_hash", "")),
            scenario_suite_hash=str(evidence_value.get("scenario_suite_hash", "")),
            metrics=metrics,
            evidence_domain=str(evidence_value.get("evidence_domain", "")),
        )
        expected_decision = (
            "CANDIDATE_READY_FOR_TEAM_FULL_CHAIN_EXAM"
            if dominance.promotion_passed
            else "CANDIDATE_ARCHIVED_BY_CPU_MUJOCO"
        )
        checkpoint_files = report.get("selected_checkpoint_files")
        checkpoint_paths = (
            [str(item.get("path", "")) for item in checkpoint_files]
            if isinstance(checkpoint_files, list)
            and all(isinstance(item, dict) for item in checkpoint_files)
            else []
        )
        checkpoint_manifest_valid = bool(
            isinstance(checkpoint_files, list)
            and checkpoint_files
            and checkpoint_paths == sorted(checkpoint_paths)
            and len(checkpoint_paths) == len(set(checkpoint_paths))
            and report.get("selected_checkpoint_hash") == hash_json(checkpoint_files)
            and all(
                set(item) == {"path", "size_bytes", "hash"}
                and Path(str(item["path"])).as_posix() == str(item["path"])
                and not Path(str(item["path"])).is_absolute()
                and ".." not in Path(str(item["path"])).parts
                and isinstance(item["size_bytes"], int)
                and not isinstance(item["size_bytes"], bool)
                and item["size_bytes"] >= 0
                and _SHA256.fullmatch(str(item["hash"])) is not None
                for item in checkpoint_files
            )
        )
        if (
            report.get("schema_version") != "rosclaw_soccer.impact_recovery_cpu_exam.v1"
            or declared != hash_json(report)
            or report.get("config_hash") != config.config_hash
            or not suite_valid
            or report.get("scenario_count") != config.episode_count * 2
            or not populations_valid
            or len(metrics) != 2
            or evidence_value != dominance.to_dict()
            or dominance.scenario_suite_hash != report.get("scenario_suite_hash")
            or report.get("dominance_evidence_hash") != dominance.evidence_hash
            or report.get("decision") != expected_decision
            or not checkpoint_manifest_valid
            or report.get("physics_backend") != "CPU_MUJOCO"
            or report.get("jax_inference_backend") != "CPU"
            or report.get("promotion_eligible") is not False
            or report.get("promotion_authority") != "NONE"
            or report.get("activation_ceiling") != "SIM_ONLY"
            or report.get("hardware_authorized") is not False
            or report.get("hardware_command_sent") is not False
            or any(
                _SHA256.fullmatch(str(report.get(name, ""))) is None
                for name in (
                    "training_report_hash",
                    "training_report_file_hash",
                    "gpu_evaluation_report_hash",
                    "gpu_evaluation_report_file_hash",
                    "curriculum_manifest_hash",
                    "curriculum_archive_hash",
                    "body_hash",
                    "selected_checkpoint_hash",
                    "scenario_suite_hash",
                    "dominance_evidence_hash",
                )
            )
        ):
            raise ValueError("impact-recovery CPU exam authority or integrity changed")
        return report
    finally:
        if declared is not None:
            report["report_hash"] = declared


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the independent CPU-MuJoCo recovery exam")
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--curriculum-manifest", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--gpu-evaluation-report", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--episodes-per-seed", default=32, type=int)
    args = parser.parse_args()
    result = run_impact_recovery_cpu_exam(
        asset_root=args.asset_root,
        curriculum_manifest_path=args.curriculum_manifest,
        training_report_path=args.training_report,
        gpu_evaluation_report_path=args.gpu_evaluation_report,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        source_checkout_path=args.source_checkout,
        config=ImpactRecoveryCPUExamConfig(episodes_per_seed=args.episodes_per_seed),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    _main()


__all__ = [
    "ImpactRecoveryCPUExamConfig",
    "run_impact_recovery_cpu_exam",
    "validate_impact_recovery_cpu_exam",
]
