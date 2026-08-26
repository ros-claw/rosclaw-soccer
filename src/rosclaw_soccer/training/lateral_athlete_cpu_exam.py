"""Independent CPU MuJoCo qualification for the lateral athlete expert.

The neural model is evaluated in the native G1 stadium with 2 ms physics,
the frozen 29-DoF locomotion prior, hard torque clipping and paired left/right
routes.  Pixels never participate in scoring.  This exam grants no hardware
authority; it only qualifies a SIM_ONLY option for later dive-router work.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.goalkeeper_mjwarp import _LOCO_DEFAULT, _LOCO_TO_MOTOR
from rosclaw_soccer.training.goalkeeper_targeted_dive_exam import _ready_control
from rosclaw_soccer.training.lateral_athlete_expert import (
    decode_lateral_athlete_command,
    lateral_athlete_features_numpy,
    load_lateral_athlete_expert,
)
from rosclaw_soccer.world.field import build_g1_stadium_model, g1_stadium_scene_hash


@dataclass(frozen=True)
class LateralAthleteExamConfig:
    """Fail-closed physical gate for bilateral lateral locomotion."""

    distances_m: tuple[float, ...] = (0.50, 1.00, 1.50, 2.00)
    seeds_per_route: int = 2
    control_dt_sec: float = 0.02
    physics_substeps: int = 10
    episode_duration_sec: float = 10.0
    command_release_sec: float = 8.5
    maximum_lateral_command_mps: float = 0.40
    maximum_command_step: float = 0.25
    command_filter_fraction: float = 0.25
    settle_error_m: float = 0.10
    settle_speed_mps: float = 0.10
    settle_hold_sec: float = 0.30
    maximum_endpoint_error_m: float = 0.10
    maximum_successor_lateral_speed_mps: float = 0.04
    maximum_successor_root_angular_speed_rad_s: float = 0.20
    minimum_pelvis_height_m: float = 0.70
    minimum_upright_projection: float = 0.96
    maximum_root_angular_speed_rad_s: float = 2.50
    maximum_bilateral_endpoint_gap_m: float = 0.05
    random_seed: int = 101_901
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.lateral_athlete_exam_config.v1"

    def __post_init__(self) -> None:
        if (
            not self.distances_m
            or tuple(sorted(self.distances_m)) != self.distances_m
            or any(
                not math.isfinite(value) or not 0.25 <= value <= 2.25 for value in self.distances_m
            )
        ):
            raise ValueError("lateral athlete exam distances are invalid")
        if not 1 <= self.seeds_per_route <= 32:
            raise ValueError("lateral athlete exam seed count is invalid")
        if not math.isclose(
            self.control_dt_sec / self.physics_substeps, 0.002, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("lateral athlete exam must use qualified 2 ms physics")
        if not 8.0 <= self.episode_duration_sec <= 15.0:
            raise ValueError("lateral athlete exam duration is invalid")
        if not 6.0 <= self.command_release_sec < self.episode_duration_sec:
            raise ValueError("lateral athlete command release is invalid")
        if not 0.20 <= self.maximum_lateral_command_mps <= 0.40:
            raise ValueError("lateral athlete exam command exceeds prior authority")
        bounded = (
            self.maximum_command_step,
            self.command_filter_fraction,
            self.settle_error_m,
            self.settle_speed_mps,
            self.settle_hold_sec,
            self.maximum_endpoint_error_m,
            self.maximum_successor_lateral_speed_mps,
            self.maximum_successor_root_angular_speed_rad_s,
            self.minimum_pelvis_height_m,
            self.minimum_upright_projection,
            self.maximum_root_angular_speed_rad_s,
            self.maximum_bilateral_endpoint_gap_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in bounded):
            raise ValueError("lateral athlete exam thresholds must be finite and positive")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("lateral athlete exam is SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _load_bound_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    claimed = report.get("report_hash")
    payload = dict(report)
    payload.pop("report_hash", None)
    if claimed != hash_json(payload):
        raise ValueError("lateral athlete training report content hash is invalid")
    return cast(dict[str, Any], report)


def _gravity_projection(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = (float(value) for value in quaternion)
    projection: np.ndarray = np.asarray(
        (
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ),
        dtype=np.float64,
    )
    return projection


def _parent_command(*, error_m: float) -> float:
    """S100 sign-only full-drive parent retained as a paired baseline."""

    return -float(np.sign(error_m)) if abs(error_m) > 0.06 else 0.0


def _rollout(
    *,
    model: Any,
    locomotion: Any,
    actor: Any | None,
    checkpoint: dict[str, Any] | None,
    target_lateral_m: float,
    seed: int,
    config: LateralAthleteExamConfig,
) -> dict[str, Any]:
    import mujoco
    import torch

    data = mujoco.MjData(model)
    ready, kp, kd, torque_limits = _ready_control()
    mujoco.mj_resetData(model, data)
    rng = np.random.default_rng(seed)
    initial_lateral_velocity = float(rng.uniform(-0.035, 0.035))
    initial_root_angular = np.asarray(
        rng.uniform((-0.025, -0.015, -0.020), (0.025, 0.015, 0.020)), dtype=np.float64
    )
    data.qpos[:7] = (4.52, 0.0, 0.793, 0.0, 0.0, 0.0, 1.0)
    data.qpos[7:36] = ready
    data.qpos[36:43] = (-20.0, 0.0, 0.115, 1.0, 0.0, 0.0, 0.0)
    data.qvel[1] = initial_lateral_velocity
    data.qvel[3:6] = initial_root_angular
    mujoco.mj_forward(model, data)
    loco_order = np.asarray(_LOCO_TO_MOTOR, dtype=np.int64)
    loco_default = np.asarray(_LOCO_DEFAULT, dtype=np.float64)
    loco_action: np.ndarray = np.zeros(29, dtype=np.float64)
    hidden = torch.zeros((1, 1, 256), dtype=torch.float32)
    cell = torch.zeros_like(hidden)
    applied_command = 0.0
    settled_latched = False
    settle_steps = 0
    settled_time: float | None = None
    minimum_pelvis = math.inf
    minimum_upright = 1.0
    maximum_angular = 0.0
    maximum_torque_fraction = 0.0
    joint_violation = False
    nonfinite = False
    command_sign_reversals = 0
    previous_command_sign = 0.0
    final_lateral_speeds: list[float] = []
    final_angular_speeds: list[float] = []
    command_trace: list[float] = []
    root_trace: list[dict[str, Any]] = []
    steps = int(round(config.episode_duration_sec / config.control_dt_sec))
    settle_hold_steps = int(round(config.settle_hold_sec / config.control_dt_sec))
    final_window_steps = int(round(0.50 / config.control_dt_sec))
    for step in range(steps):
        time_sec = step * config.control_dt_sec
        error = target_lateral_m - float(data.qpos[1])
        if settled_latched or time_sec >= config.command_release_sec:
            requested_command = 0.0
        elif actor is None:
            requested_command = _parent_command(error_m=error)
        else:
            if checkpoint is None:
                raise RuntimeError("lateral athlete checkpoint disappeared")
            qw, _, _, qz = (float(data.qpos[index]) for index in range(3, 7))
            upright = 2.0 * (qw * qw + qz * qz) - 1.0
            features = lateral_athlete_features_numpy(
                lateral_error_m=np.asarray((error,), dtype=np.float64),
                lateral_velocity_mps=np.asarray((float(data.qvel[1]),), dtype=np.float64),
                time_remaining_sec=np.asarray(
                    (max(0.0, config.command_release_sec - time_sec),), dtype=np.float64
                ),
                pelvis_height_m=np.asarray((float(data.qpos[2]),), dtype=np.float64),
                upright_projection=np.asarray((upright,), dtype=np.float64),
                root_angular_velocity_rad_s=np.asarray(data.qvel[3:6], dtype=np.float64).reshape(
                    1, 3
                ),
                previous_command=np.asarray((applied_command,), dtype=np.float64),
            )
            with torch.inference_mode():
                requested_command = float(
                    decode_lateral_athlete_command(
                        torch=torch,
                        model=actor,
                        features=torch.as_tensor(features, dtype=torch.float32),
                    )[0]
                )
        delta = float(
            np.clip(
                requested_command - applied_command,
                -config.maximum_command_step,
                config.maximum_command_step,
            )
        )
        applied_command += config.command_filter_fraction * delta
        command_trace.append(applied_command)
        command_sign = float(np.sign(applied_command)) if abs(applied_command) >= 0.10 else 0.0
        if (
            command_sign != 0.0
            and previous_command_sign != 0.0
            and command_sign != previous_command_sign
        ):
            command_sign_reversals += 1
        if command_sign != 0.0:
            previous_command_sign = command_sign
        observation: np.ndarray = np.zeros(96, dtype=np.float32)
        observation[:3] = data.qvel[3:6]
        observation[3:6] = _gravity_projection(np.asarray(data.qpos[3:7]))
        observation[7] = applied_command * config.maximum_lateral_command_mps
        observation[9:38] = data.qpos[7:36][loco_order] - loco_default
        observation[38:67] = data.qvel[6:35][loco_order]
        observation[67:96] = loco_action
        with torch.inference_mode():
            encoded = locomotion.normalizer.forward(
                torch.from_numpy(np.clip(observation, -100.0, 100.0))[None, :]
            )
            sequence, (hidden, cell) = locomotion.rnn.forward__0(
                encoded.unsqueeze(0), (hidden, cell)
            )
            loco_action = np.asarray(
                torch.clamp(locomotion.actor.forward(sequence.squeeze(0)), -100.0, 100.0)[0]
            )
        target: np.ndarray = np.zeros(29, dtype=np.float64)
        target[loco_order] = 0.25 * loco_action + loco_default
        for _ in range(config.physics_substeps):
            torque = np.clip(
                kp * (target - data.qpos[7:36]) - kd * data.qvel[6:35],
                -torque_limits,
                torque_limits,
            )
            maximum_torque_fraction = max(
                maximum_torque_fraction,
                float(np.max(np.abs(torque) / torque_limits)),
            )
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
        finite = bool(
            np.all(np.isfinite(data.qpos))
            and np.all(np.isfinite(data.qvel))
            and np.all(np.isfinite(data.ctrl))
        )
        if not finite:
            nonfinite = True
            break
        joint_position = np.asarray(data.qpos[7:36], dtype=np.float64)
        limited = np.asarray(model.jnt_limited[1:30], dtype=bool)
        lower = np.asarray(model.jnt_range[1:30, 0], dtype=np.float64)
        upper = np.asarray(model.jnt_range[1:30, 1], dtype=np.float64)
        joint_violation |= bool(
            np.any(limited & ((joint_position < lower - 0.03) | (joint_position > upper + 0.03)))
        )
        qw, _, _, qz = (float(data.qpos[index]) for index in range(3, 7))
        upright = 2.0 * (qw * qw + qz * qz) - 1.0
        angular_speed = float(np.linalg.norm(data.qvel[3:6]))
        minimum_pelvis = min(minimum_pelvis, float(data.qpos[2]))
        minimum_upright = min(minimum_upright, upright)
        maximum_angular = max(maximum_angular, angular_speed)
        root_trace.append(
            {
                "time_sec": time_sec,
                "qpos": np.asarray(data.qpos, dtype=np.float64).tolist(),
                "lateral_velocity_mps": float(data.qvel[1]),
                "root_angular_speed_rad_s": angular_speed,
                "applied_command": applied_command,
            }
        )
        if (
            abs(target_lateral_m - float(data.qpos[1])) <= config.settle_error_m
            and abs(float(data.qvel[1])) <= config.settle_speed_mps
        ):
            settle_steps += 1
        else:
            settle_steps = 0
        if not settled_latched and settle_steps >= settle_hold_steps:
            settled_latched = True
            settled_time = time_sec
        if step >= steps - final_window_steps:
            final_lateral_speeds.append(abs(float(data.qvel[1])))
            final_angular_speeds.append(angular_speed)
    endpoint_error = abs(target_lateral_m - float(data.qpos[1])) if not nonfinite else math.inf
    successor_lateral = max(final_lateral_speeds, default=math.inf)
    successor_angular = max(final_angular_speeds, default=math.inf)
    safe = bool(
        not nonfinite
        and not joint_violation
        and minimum_pelvis >= config.minimum_pelvis_height_m
        and minimum_upright >= config.minimum_upright_projection
        and maximum_angular <= config.maximum_root_angular_speed_rad_s
        and maximum_torque_fraction <= 1.0 + 1.0e-9
    )
    passed = bool(
        safe
        and settled_time is not None
        and endpoint_error <= config.maximum_endpoint_error_m
        and successor_lateral <= config.maximum_successor_lateral_speed_mps
        and successor_angular <= config.maximum_successor_root_angular_speed_rad_s
    )
    return {
        "seed": seed,
        "target_lateral_m": target_lateral_m,
        "initial_lateral_velocity_mps": initial_lateral_velocity,
        "initial_root_angular_velocity_rad_s": initial_root_angular.tolist(),
        "settled_time_sec": settled_time,
        "endpoint_lateral_m": float(data.qpos[1]),
        "endpoint_error_m": endpoint_error,
        "maximum_successor_lateral_speed_mps": successor_lateral,
        "maximum_successor_root_angular_speed_rad_s": successor_angular,
        "minimum_pelvis_height_m": minimum_pelvis,
        "minimum_upright_projection": minimum_upright,
        "maximum_root_angular_speed_rad_s": maximum_angular,
        "maximum_torque_fraction": maximum_torque_fraction,
        "command_sign_reversals": command_sign_reversals,
        "mean_absolute_command_delta": float(np.mean(np.abs(np.diff(np.asarray(command_trace))))),
        "nonfinite": nonfinite,
        "joint_limit_violation": joint_violation,
        "safe": safe,
        "passed": passed,
        "trajectory": root_trace,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "route_count": len(rows),
        "pass_rate": float(np.mean([bool(row["passed"]) for row in rows])),
        "safe_rate": float(np.mean([bool(row["safe"]) for row in rows])),
        "maximum_endpoint_error_m": float(max(float(row["endpoint_error_m"]) for row in rows)),
        "mean_endpoint_error_m": float(np.mean([float(row["endpoint_error_m"]) for row in rows])),
        "maximum_successor_lateral_speed_mps": float(
            max(float(row["maximum_successor_lateral_speed_mps"]) for row in rows)
        ),
        "maximum_successor_root_angular_speed_rad_s": float(
            max(float(row["maximum_successor_root_angular_speed_rad_s"]) for row in rows)
        ),
        "minimum_pelvis_height_m": float(
            min(float(row["minimum_pelvis_height_m"]) for row in rows)
        ),
        "minimum_upright_projection": float(
            min(float(row["minimum_upright_projection"]) for row in rows)
        ),
        "maximum_root_angular_speed_rad_s": float(
            max(float(row["maximum_root_angular_speed_rad_s"]) for row in rows)
        ),
        "mean_command_sign_reversals": float(
            np.mean([int(row["command_sign_reversals"]) for row in rows])
        ),
        "mean_absolute_command_delta": float(
            np.mean([float(row["mean_absolute_command_delta"]) for row in rows])
        ),
    }


def _bilateral_endpoint_gap(rows: list[dict[str, Any]]) -> float:
    gaps: list[float] = []
    by_route = {(abs(float(row["target_lateral_m"])), int(row["seed"])): row for row in rows}
    for row in rows:
        if float(row["target_lateral_m"]) <= 0.0:
            continue
        partner = by_route[(abs(float(row["target_lateral_m"])), int(row["seed"]))]
        negative = next(
            item
            for item in rows
            if float(item["target_lateral_m"]) == -float(row["target_lateral_m"])
            and int(item["seed"]) == int(row["seed"])
        )
        gaps.append(abs(float(partner["endpoint_error_m"]) - float(negative["endpoint_error_m"])))
    return max(gaps, default=math.inf)


def _without_trajectories(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key != "trajectory"} for row in rows]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def run_lateral_athlete_cpu_exam(
    *,
    asset_root: Path,
    locomotion_policy_path: Path,
    expert_checkpoint_path: Path,
    training_report_path: Path,
    output_path: Path,
    trajectory_output_path: Path | None = None,
    config: LateralAthleteExamConfig | None = None,
) -> dict[str, Any]:
    """Run paired parent/candidate routes and write content-addressed evidence."""

    import torch

    active = config or LateralAthleteExamConfig()
    locomotion_path = locomotion_policy_path.expanduser().resolve()
    checkpoint_path = expert_checkpoint_path.expanduser().resolve()
    training = _load_bound_report(training_report_path)
    checkpoint_hash = hash_bytes(checkpoint_path.read_bytes())
    locomotion_hash = hash_bytes(locomotion_path.read_bytes())
    if (
        training.get("checkpoint_hash") != checkpoint_hash
        or training.get("locomotion_policy_hash") != locomotion_hash
        or not bool(training.get("fit_gate_passed", False))
    ):
        raise ValueError("lateral athlete training evidence binding is invalid")
    actor, checkpoint = load_lateral_athlete_expert(
        checkpoint_path=checkpoint_path,
        locomotion_policy_path=locomotion_path,
        device=torch.device("cpu"),
    )
    locomotion = torch.jit.load(str(locomotion_path), map_location="cpu")
    locomotion.eval()
    model = build_g1_stadium_model(asset_root.expanduser().resolve())
    cases = [
        (direction * distance, active.random_seed + seed_index)
        for distance in active.distances_m
        for seed_index in range(active.seeds_per_route)
        for direction in (-1.0, 1.0)
    ]
    parent_rows = [
        _rollout(
            model=model,
            locomotion=locomotion,
            actor=None,
            checkpoint=None,
            target_lateral_m=target,
            seed=seed,
            config=active,
        )
        for target, seed in cases
    ]
    candidate_rows = [
        _rollout(
            model=model,
            locomotion=locomotion,
            actor=actor,
            checkpoint=checkpoint,
            target_lateral_m=target,
            seed=seed,
            config=active,
        )
        for target, seed in cases
    ]
    parent_summary = _summary(parent_rows)
    candidate_summary = _summary(candidate_rows)
    bilateral_gap = _bilateral_endpoint_gap(candidate_rows)
    passed = bool(
        candidate_summary["pass_rate"] == 1.0
        and candidate_summary["safe_rate"] == 1.0
        and bilateral_gap <= active.maximum_bilateral_endpoint_gap_m
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.lateral_athlete_cpu_exam.v1",
        "physics_backend": "mujoco_cpu",
        "physics_scene_hash": g1_stadium_scene_hash(asset_root.expanduser().resolve()),
        "config": asdict(active),
        "config_hash": active.config_hash,
        "challenge": "PAIRED_0P5_TO_2P0_M_LATERAL_ACCELERATE_BRAKE_SUCCESSOR_STATE",
        "route_count": len(cases),
        "parent_controller": "S100_SIGN_ONLY_FULL_DRIVE",
        "candidate_controller": "BILATERAL_NEURAL_LATERAL_ATHLETE_EXPERT",
        "parent": {"summary": parent_summary, "routes": _without_trajectories(parent_rows)},
        "candidate": {
            "summary": candidate_summary,
            "maximum_bilateral_endpoint_gap_m": bilateral_gap,
            "routes": _without_trajectories(candidate_rows),
        },
        "training_report_hash": training["report_hash"],
        "expert_checkpoint_hash": checkpoint_hash,
        "locomotion_policy_hash": locomotion_hash,
        "fit_gate_passed": True,
        "passed": passed,
        "promotion_status": (
            "QUALIFIED_LATERAL_EXPERT_PENDING_DIVE_ROUTER"
            if passed
            else "REJECTED_BY_LATERAL_ATHLETE_CPU_EXAM"
        ),
        "video_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(output_path, report)
    if trajectory_output_path is not None:
        trajectory_payload = {
            "schema_version": "rosclaw_soccer.lateral_athlete_trajectories.v1",
            "exam_report_hash": report["report_hash"],
            "parent": parent_rows,
            "candidate": candidate_rows,
            "video_used_for_scoring": False,
            "activation_ceiling": "SIM_ONLY",
        }
        trajectory_payload["trajectory_hash"] = hash_json(trajectory_payload)
        _atomic_json(trajectory_output_path, trajectory_payload)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--locomotion-policy", type=Path, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path)
    args = parser.parse_args()
    report = run_lateral_athlete_cpu_exam(
        asset_root=args.asset_root,
        locomotion_policy_path=args.locomotion_policy,
        expert_checkpoint_path=args.expert_checkpoint,
        training_report_path=args.training_report,
        output_path=args.output,
        trajectory_output_path=args.trajectory_output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["LateralAthleteExamConfig", "run_lateral_athlete_cpu_exam"]
