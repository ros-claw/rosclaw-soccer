"""Paired CPU MuJoCo exam for far-corner dive muscle memory.

The exam isolates the first-shot problem requested by the academy: equally
many low, mid, and high far-corner shots, with no central/easy examples.  It
compares the same seeds against a ready-pose baseline and records every miss
for the next curriculum round.  Contact and dynamics come from MuJoCo; pixels
are never used for scoring.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_json
from rosclaw_soccer.training.goalkeeper_dive_memory import (
    decode_goalkeeper_dive_memory,
)
from rosclaw_soccer.training.goalkeeper_mjwarp import (
    _LOCO_DEFAULT,
    _LOCO_KD,
    _LOCO_KP,
    _LOCO_TO_MOTOR,
)
from rosclaw_soccer.world.field import build_g1_stadium_model, g1_stadium_scene_hash


@dataclass(frozen=True)
class GoalkeeperDiveMemoryExamConfig:
    shots_per_stratum: int = 20
    random_seed: int = 416_001
    control_dt_sec: float = 0.02
    physics_substeps: int = 10
    shot_release_sec: float = 0.70
    episode_duration_sec: float = 2.30
    far_lateral_range_m: tuple[float, float] = (0.74, 1.06)
    low_height_range_m: tuple[float, float] = (0.16, 0.58)
    mid_height_range_m: tuple[float, float] = (0.64, 1.06)
    high_height_range_m: tuple[float, float] = (1.14, 1.52)
    flight_time_range_sec: tuple[float, float] = (0.40, 0.54)
    ball_start_x_range_m: tuple[float, float] = (0.45, 1.45)
    ball_start_z_range_m: tuple[float, float] = (0.22, 0.78)
    minimum_target_save_rate: float = 0.80
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_dive_memory_exam_config.v1"

    def __post_init__(self) -> None:
        if not 8 <= self.shots_per_stratum <= 1_000:
            raise ValueError("goalkeeper dive exam stratum size is invalid")
        if not math.isclose(
            self.control_dt_sec / self.physics_substeps,
            0.002,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("goalkeeper dive exam physics step must be 0.002 s")
        if not 0.0 < self.shot_release_sec < self.episode_duration_sec:
            raise ValueError("goalkeeper dive exam timing is invalid")
        ranges = (
            self.far_lateral_range_m,
            self.low_height_range_m,
            self.mid_height_range_m,
            self.high_height_range_m,
            self.flight_time_range_sec,
            self.ball_start_x_range_m,
            self.ball_start_z_range_m,
        )
        if any(
            not all(math.isfinite(value) for value in limits) or limits[0] >= limits[1]
            for limits in ranges
        ):
            raise ValueError("goalkeeper dive exam range is invalid")
        if not 0.5 <= self.minimum_target_save_rate <= 1.0:
            raise ValueError("goalkeeper dive exam target rate is invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("goalkeeper dive memory exam is SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _contact_state(model: Any, data: Any) -> tuple[bool, bool]:
    ball_body = int(model.body("ball").id)
    hand_geoms = {
        int(model.geom("left_hand_collision").id),
        int(model.geom("right_hand_collision").id),
        int(model.geom("left_goalkeeper_glove").id),
        int(model.geom("right_goalkeeper_glove").id),
    }
    robot_contact = False
    hand_contact = False
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.dist > 0.002:
            continue
        geom_one = int(contact.geom1)
        geom_two = int(contact.geom2)
        body_one = int(model.geom_bodyid[geom_one])
        body_two = int(model.geom_bodyid[geom_two])
        ball_one = body_one == ball_body
        ball_two = body_two == ball_body
        robot_one = 1 <= body_one < ball_body
        robot_two = 1 <= body_two < ball_body
        if (ball_one and robot_two) or (ball_two and robot_one):
            robot_contact = True
            hand_contact |= bool(
                (ball_one and geom_two in hand_geoms) or (ball_two and geom_one in hand_geoms)
            )
    return robot_contact, hand_contact


def _ready_control() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ready = np.zeros(29, dtype=np.float64)
    order = np.asarray(_LOCO_TO_MOTOR, dtype=np.int64)
    ready[order] = np.asarray(_LOCO_DEFAULT, dtype=np.float64)
    kp = np.zeros(29, dtype=np.float64)
    kd = np.zeros(29, dtype=np.float64)
    kp[order] = np.asarray(_LOCO_KP, dtype=np.float64)
    kd[order] = np.asarray(_LOCO_KD, dtype=np.float64)
    kp[12:] = np.asarray((150.0, 150.0, 150.0) + ((150.0,) * 4 + (20.0,) * 3) * 2)
    kd[12:] = np.asarray((2.0, 2.0, 2.0) + ((2.0,) * 4 + (0.5,) * 3) * 2)
    return ready, kp, kd, np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)


def _sample_cases(config: GoalkeeperDiveMemoryExamConfig) -> list[dict[str, Any]]:
    rng = np.random.default_rng(config.random_seed)
    cases: list[dict[str, Any]] = []
    height_ranges = {
        "far_corner_low": config.low_height_range_m,
        "far_corner_mid": config.mid_height_range_m,
        "far_corner_high": config.high_height_range_m,
    }
    for stratum, height_range in height_ranges.items():
        for _ in range(config.shots_per_stratum):
            sign = float(rng.choice((-1.0, 1.0)))
            cases.append(
                {
                    "seed": config.random_seed + len(cases),
                    "stratum": stratum,
                    "target_y_m": sign * float(rng.uniform(*config.far_lateral_range_m)),
                    "target_z_m": float(rng.uniform(*height_range)),
                    "flight_sec": float(rng.uniform(*config.flight_time_range_sec)),
                    "start_x_m": float(rng.uniform(*config.ball_start_x_range_m)),
                    "start_z_m": float(rng.uniform(*config.ball_start_z_range_m)),
                }
            )
    rng.shuffle(cases)
    return cases


def _rollout_case(
    *,
    model: Any,
    case: dict[str, Any],
    trajectory: np.ndarray | None,
    config: GoalkeeperDiveMemoryExamConfig,
) -> dict[str, Any]:
    import mujoco

    data = mujoco.MjData(model)
    ready, kp, kd, limits = _ready_control()
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = (4.52, 0.0, 0.793, 0.0, 0.0, 0.0, 1.0)
    data.qpos[7:36] = ready
    data.qpos[36:43] = (-20.0, 0.0, 0.115, 1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    control_steps = int(round(config.episode_duration_sec / config.control_dt_sec))
    release_step = int(round(config.shot_release_sec / config.control_dt_sec))
    blend_steps = 25
    contact = False
    hand_contact = False
    contact_before_plane = False
    minimum_pelvis = math.inf
    maximum_root_speed = 0.0
    maximum_root_angular = 0.0
    maximum_torque_fraction = 0.0
    for control_step in range(control_steps):
        if trajectory is None:
            target = ready
        elif control_step < blend_steps:
            target = ready + (trajectory[0] - ready) * ((control_step + 1) / blend_steps)
        elif control_step - blend_steps < trajectory.shape[0]:
            target = trajectory[control_step - blend_steps]
        else:
            recovery_step = control_step - blend_steps - trajectory.shape[0]
            recovery_fraction = min(1.0, (recovery_step + 1) / blend_steps)
            target = trajectory[-1] + (ready - trajectory[-1]) * recovery_fraction
        if control_step == release_step:
            target_x = 4.44
            target_y = float(case["target_y_m"])
            target_z = float(case["target_z_m"])
            flight = float(case["flight_sec"])
            start_x = float(case["start_x_m"])
            start_z = float(case["start_z_m"])
            data.qpos[36:43] = (
                start_x,
                0.18 * target_y,
                start_z,
                1.0,
                0.0,
                0.0,
                0.0,
            )
            data.qvel[35:41] = 0.0
            data.qvel[35] = (target_x - start_x) / flight
            data.qvel[36] = (target_y - 0.18 * target_y) / flight
            data.qvel[37] = (target_z - start_z + 0.5 * 9.81 * flight * flight) / flight
            mujoco.mj_forward(model, data)
        for _ in range(config.physics_substeps):
            requested = kp * (target - data.qpos[7:36]) - kd * data.qvel[6:35]
            maximum_torque_fraction = max(
                maximum_torque_fraction, float(np.max(np.abs(requested) / limits))
            )
            data.ctrl[:] = np.clip(requested, -limits, limits)
            mujoco.mj_step(model, data)
            robot_now, hand_now = _contact_state(model, data)
            contact |= robot_now
            hand_contact |= hand_now
            contact_before_plane |= bool(robot_now and float(data.qpos[36]) <= 4.60)
            minimum_pelvis = min(minimum_pelvis, float(data.qpos[2]))
            maximum_root_speed = max(maximum_root_speed, float(np.linalg.norm(data.qvel[:3])))
            maximum_root_angular = max(maximum_root_angular, float(np.linalg.norm(data.qvel[3:6])))
    finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
    safe = bool(
        finite
        and minimum_pelvis >= 0.50
        and maximum_root_speed <= 2.0
        and maximum_root_angular <= 4.0
    )
    return {
        **case,
        "saved": bool(contact_before_plane and safe),
        "robot_contact": contact,
        "hand_contact": hand_contact,
        "safe": safe,
        "minimum_pelvis_height_m": minimum_pelvis,
        "maximum_root_speed_mps": maximum_root_speed,
        "maximum_root_angular_speed_rad_s": maximum_root_angular,
        "maximum_requested_torque_fraction": maximum_torque_fraction,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_stratum: dict[str, Any] = {}
    for stratum in ("far_corner_low", "far_corner_mid", "far_corner_high"):
        selected = [row for row in rows if row["stratum"] == stratum]
        by_stratum[stratum] = {
            "attempts": len(selected),
            "saves": sum(bool(row["saved"]) for row in selected),
            "save_rate": sum(bool(row["saved"]) for row in selected) / len(selected),
            "hand_save_rate": sum(bool(row["hand_contact"] and row["saved"]) for row in selected)
            / len(selected),
            "safe_rate": sum(bool(row["safe"]) for row in selected) / len(selected),
        }
    return {
        "attempts": len(rows),
        "save_rate": sum(bool(row["saved"]) for row in rows) / len(rows),
        "safe_rate": sum(bool(row["safe"]) for row in rows) / len(rows),
        "strata": by_stratum,
        "misses": [
            {
                "seed": row["seed"],
                "stratum": row["stratum"],
                "target_y_m": row["target_y_m"],
                "target_z_m": row["target_z_m"],
                "safe": row["safe"],
            }
            for row in rows
            if not row["saved"]
        ],
        "maximum_root_angular_speed_rad_s": max(
            float(row["maximum_root_angular_speed_rad_s"]) for row in rows
        ),
        "minimum_pelvis_height_m": min(float(row["minimum_pelvis_height_m"]) for row in rows),
    }


def run_goalkeeper_dive_memory_exam(
    *,
    asset_root: Path,
    checkpoint_path: Path,
    output_path: Path,
    config: GoalkeeperDiveMemoryExamConfig | None = None,
) -> dict[str, Any]:
    """Run paired ready-pose and learned-memory trials on identical hard shots."""

    active = config or GoalkeeperDiveMemoryExamConfig()
    phases = np.linspace(0.0, 1.0, 71, dtype=np.float64)
    decoded = {
        direction: decode_goalkeeper_dive_memory(
            checkpoint_path=checkpoint_path,
            direction=direction,
            phases=phases,
        )
        for direction in ("left", "right")
    }
    model = build_g1_stadium_model(asset_root.expanduser().resolve())
    cases = _sample_cases(active)
    baseline_rows = [
        _rollout_case(model=model, case=case, trajectory=None, config=active) for case in cases
    ]
    candidate_rows = [
        _rollout_case(
            model=model,
            case=case,
            trajectory=decoded["left" if float(case["target_y_m"]) < 0.0 else "right"],
            config=active,
        )
        for case in cases
    ]
    baseline = _summary(baseline_rows)
    candidate = _summary(candidate_rows)
    strata_pass = all(
        float(item["save_rate"]) >= active.minimum_target_save_rate
        for item in candidate["strata"].values()
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.goalkeeper_dive_memory_exam.v1",
        "physics_backend": "mujoco_cpu",
        "physics_scene_hash": g1_stadium_scene_hash(asset_root.expanduser().resolve()),
        "config": asdict(active),
        "config_hash": active.config_hash,
        "challenge": "FIRST_SHOT_FAR_CORNER_LOW_MID_HIGH_ONLY",
        "paired_seed_count": len(cases),
        "baseline": baseline,
        "candidate": candidate,
        "minimum_target_save_rate": active.minimum_target_save_rate,
        "passed_80_percent_each_stratum": strata_pass,
        "passed": bool(strata_pass and candidate["safe_rate"] == 1.0),
        "promotion_status": (
            "OPTION_PENDING_POLICY_INTEGRATION"
            if strata_pass and candidate["safe_rate"] == 1.0
            else "REJECTED_BY_HARD_STRATIFIED_EXAM"
        ),
        "video_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots-per-stratum", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=416_001)
    args = parser.parse_args()
    report = run_goalkeeper_dive_memory_exam(
        asset_root=args.asset_root,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        config=GoalkeeperDiveMemoryExamConfig(
            shots_per_stratum=args.shots_per_stratum,
            random_seed=args.random_seed,
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["GoalkeeperDiveMemoryExamConfig", "run_goalkeeper_dive_memory_exam"]
