"""Run physics-first football checks before any policy is trained or promoted."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.physics.standards import IFABRegulationSpec

_DT_SEC = 0.002


@dataclass(frozen=True)
class RealityPackCase:
    case_id: str
    passed: bool
    metrics: dict[str, float | bool]
    thresholds: dict[str, float | bool]


@dataclass(frozen=True)
class RealityPackReport:
    passed: bool
    cases: tuple[RealityPackCase, ...]
    output_path: str
    curves_path: str
    report_hash: str
    activation_ceiling: str = "SIM_ONLY"
    physics_authority: str = "CPU_MUJOCO"
    schema_version: str = "rosclaw_soccer.reality_pack.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "cases": [asdict(case) for case in self.cases],
        }


def run_reality_pack(
    *,
    output_dir: Path,
    source_checkout: Path,
    standard: IFABRegulationSpec | None = None,
) -> RealityPackReport:
    """Execute ball drop, roll, slide, frame, and compliant-net tests."""

    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("Reality Pack output must remain outside the source checkout")
    root.mkdir(parents=True, exist_ok=True)
    spec = standard or IFABRegulationSpec()
    cases: list[RealityPackCase] = []
    curves: dict[str, NDArray[np.float64]] = {}

    drop_case, drop_curves = _ball_drop(spec)
    cases.append(drop_case)
    curves.update(drop_curves)
    roll_case, roll_curves = _ball_roll(spec, matched_spin=True)
    cases.append(roll_case)
    curves.update(roll_curves)
    slide_case, slide_curves = _ball_roll(spec, matched_spin=False)
    cases.append(slide_case)
    curves.update(slide_curves)
    frame_case, frame_curves = _goal_frame_rebound(spec)
    cases.append(frame_case)
    curves.update(frame_curves)
    net_case, net_curves = _goal_net_capture(spec)
    cases.append(net_case)
    curves.update(net_curves)

    curves_path = root / "reality-pack-curves.npz"
    save_curves: Any = np.savez_compressed
    save_curves(curves_path, **curves)
    unsigned = {
        "schema_version": "rosclaw_soccer.reality_pack.v1",
        "activation_ceiling": "SIM_ONLY",
        "physics_authority": "CPU_MUJOCO",
        "standard": spec.to_dict(),
        "cases": [asdict(case) for case in cases],
        "curves_sha256": _hash_file(curves_path),
        "passed": all(case.passed for case in cases),
    }
    report_hash = _hash_json(unsigned)
    report_path = root / "reality-pack.json"
    report_path.write_text(
        json.dumps({**unsigned, "report_hash": report_hash}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return RealityPackReport(
        passed=bool(unsigned["passed"]),
        cases=tuple(cases),
        output_path=str(report_path),
        curves_path=str(curves_path),
        report_hash=report_hash,
    )


def _model(spec: IFABRegulationSpec) -> tuple[Any, Any, int, int]:
    import mujoco

    radius = spec.ball_radius_m
    inertia = spec.ball_solid_sphere_inertia_kg_m2
    xml = f"""
<mujoco model="rosclaw_soccer_reality_pack">
  <option timestep="{_DT_SEC}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <geom solref="0.001 1" solimp="0.95 0.995 0.001"/>
  </default>
  <worldbody>
    <geom name="pitch" type="plane" size="80 50 0.1" rgba="0.04 0.24 0.06 1"
          friction="0.9 0.005 0.0001"/>
    <body name="ball" pos="0 0 {radius}">
      <freejoint name="ball_free"/>
      <inertial pos="0 0 0" mass="{spec.ball_mass_kg}"
                diaginertia="{inertia} {inertia} {inertia}"/>
      <geom name="ball_collision" type="sphere" size="{radius}"
            rgba="0.96 0.96 0.94 1"
            friction="{spec.ball_sliding_friction} {spec.ball_torsional_friction}
                      {spec.ball_rolling_friction}"/>
    </body>
    {_goal_xml(spec)}
  </worldbody>
  <contact>
    <pair name="ball_pitch" geom1="ball_collision" geom2="pitch" condim="6"
          friction="{spec.ball_sliding_friction} {spec.ball_sliding_friction}
                    {spec.ball_torsional_friction} {spec.ball_rolling_friction}
                    {spec.ball_rolling_friction}"/>
  </contact>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    ball_body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball"))
    ball_joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"))
    qpos = int(model.jnt_qposadr[ball_joint])
    qvel = int(model.jnt_dofadr[ball_joint])
    model.dof_damping[qvel : qvel + 3] = spec.ball_linear_damping_n_s_m
    model.dof_damping[qvel + 3 : qvel + 6] = 0.00002
    return model, data, ball_body, qpos


def _goal_xml(spec: IFABRegulationSpec) -> str:
    x = 8.5
    half_width = spec.goal_inside_width_m / 2.0
    height = spec.goal_inside_height_m
    radius = spec.goal_frame_radius_m
    rear = x + spec.net_depth_m
    top_rear = x + 0.72 * spec.net_depth_m
    return f"""
    <geom name="left_post" type="capsule"
          fromto="{x} {-half_width} 0 {x} {-half_width} {height}"
          size="{radius}" rgba="1 1 1 1"/>
    <geom name="right_post" type="capsule"
          fromto="{x} {half_width} 0 {x} {half_width} {height}"
          size="{radius}" rgba="1 1 1 1"/>
    <geom name="crossbar" type="capsule"
          fromto="{x} {-half_width} {height} {x} {half_width} {height}"
          size="{radius}" rgba="1 1 1 1"/>
    <geom name="left_back_support" type="capsule"
          fromto="{rear} {-half_width} 0 {top_rear} {-half_width} {height}"
          size="{0.6 * radius}" rgba="0.9 0.9 0.9 1"/>
    <geom name="right_back_support" type="capsule"
          fromto="{rear} {half_width} 0 {top_rear} {half_width} {height}"
          size="{0.6 * radius}" rgba="0.9 0.9 0.9 1"/>
    """


def _reset_ball(
    model: Any,
    data: Any,
    qpos: int,
    position: tuple[float, float, float],
    linear_velocity: tuple[float, float, float],
    angular_velocity: tuple[float, float, float],
) -> int:
    import mujoco

    mujoco.mj_resetData(model, data)
    ball_joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"))
    qvel = int(model.jnt_dofadr[ball_joint])
    data.qpos[qpos : qpos + 3] = position
    data.qpos[qpos + 3 : qpos + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[qvel : qvel + 3] = linear_velocity
    data.qvel[qvel + 3 : qvel + 6] = angular_velocity
    mujoco.mj_forward(model, data)
    return qvel


def _trace(
    model: Any,
    data: Any,
    *,
    qpos: int,
    qvel: int,
    duration_sec: float,
    force: Any | None = None,
) -> dict[str, NDArray[np.float64]]:
    import mujoco

    rows = int(round(duration_sec / model.opt.timestep))
    time = np.empty(rows, dtype=np.float64)
    position = np.empty((rows, 3), dtype=np.float64)
    velocity = np.empty((rows, 3), dtype=np.float64)
    angular = np.empty((rows, 3), dtype=np.float64)
    for index in range(rows):
        if force is not None:
            force()
        mujoco.mj_step(model, data)
        time[index] = data.time
        position[index] = data.qpos[qpos : qpos + 3]
        velocity[index] = data.qvel[qvel : qvel + 3]
        angular[index] = data.qvel[qvel + 3 : qvel + 6]
    return {"time": time, "position": position, "velocity": velocity, "angular": angular}


def _ball_drop(
    spec: IFABRegulationSpec,
) -> tuple[RealityPackCase, dict[str, NDArray[np.float64]]]:
    model, data, _body, qpos = _model(spec)
    qvel = _reset_ball(model, data, qpos, (0.0, 0.0, 1.5), (0.0, 0.0, 0.0), (0, 0, 0))
    trace = _trace(model, data, qpos=qpos, qvel=qvel, duration_sec=2.5)
    z = trace["position"][:, 2]
    minimum_z = float(np.min(z))
    maximum_compression = max(0.0, spec.ball_radius_m - minimum_z)
    settled_height_error = abs(float(np.mean(z[-100:])) - spec.ball_radius_m)
    passed = maximum_compression <= 0.035 and settled_height_error <= 0.008
    case = RealityPackCase(
        case_id="ball_drop",
        passed=passed,
        metrics={
            "minimum_center_height_m": minimum_z,
            "maximum_contact_compression_m": maximum_compression,
            "settled_height_error_m": settled_height_error,
        },
        thresholds={
            "maximum_contact_compression_m": 0.035,
            "settled_height_error_m": 0.008,
        },
    )
    return case, {f"drop_{key}": value for key, value in trace.items()}


def _ball_roll(
    spec: IFABRegulationSpec,
    *,
    matched_spin: bool,
) -> tuple[RealityPackCase, dict[str, NDArray[np.float64]]]:
    model, data, _body, qpos = _model(spec)
    speed = 4.0
    spin = speed / spec.ball_radius_m if matched_spin else 0.0
    qvel = _reset_ball(
        model,
        data,
        qpos,
        (0.0, 0.0, spec.ball_radius_m),
        (speed, 0.0, 0.0),
        (0.0, spin, 0.0),
    )
    trace = _trace(model, data, qpos=qpos, qvel=qvel, duration_sec=3.0)
    rolling_speed = np.abs(trace["angular"][:, 1]) * spec.ball_radius_m
    slip = np.abs(trace["velocity"][:, 0] - rolling_speed)
    denominator = np.maximum(np.abs(trace["velocity"][:, 0]), 0.2)
    slip_ratio = slip / denominator
    stable = trace["time"] >= 0.6
    mean_stable_slip = float(np.mean(slip_ratio[stable]))
    distance = float(trace["position"][-1, 0] - trace["position"][0, 0])
    lateral_drift = float(np.max(np.abs(trace["position"][:, 1])))
    if matched_spin:
        passed = mean_stable_slip <= 0.12 and distance >= 2.5 and lateral_drift <= 0.01
        case_id = "ball_roll"
        slip_threshold = 0.12
    else:
        early_slip = float(np.mean(slip_ratio[:50]))
        passed = early_slip >= 0.70 and mean_stable_slip <= 0.18 and distance >= 2.0
        case_id = "ball_ground_slide"
        slip_threshold = 0.18
    case = RealityPackCase(
        case_id=case_id,
        passed=passed,
        metrics={
            "distance_m": distance,
            "lateral_drift_m": lateral_drift,
            "stable_slip_ratio": mean_stable_slip,
            "rotation_observed": bool(np.max(np.abs(trace["angular"][:, 1])) >= 5.0),
        },
        thresholds={
            "minimum_distance_m": 2.5 if matched_spin else 2.0,
            "maximum_stable_slip_ratio": slip_threshold,
            "maximum_lateral_drift_m": 0.01,
        },
    )
    prefix = "roll" if matched_spin else "slide"
    return case, {
        **{f"{prefix}_{key}": value for key, value in trace.items()},
        f"{prefix}_slip_ratio": slip_ratio,
    }


def _goal_frame_rebound(
    spec: IFABRegulationSpec,
) -> tuple[RealityPackCase, dict[str, NDArray[np.float64]]]:
    model, data, _body, qpos = _model(spec)
    y = spec.goal_inside_width_m / 2.0
    qvel = _reset_ball(
        model,
        data,
        qpos,
        (6.0, y, 0.8),
        (9.0, 0.0, 0.0),
        (0.0, 9.0 / spec.ball_radius_m, 0.0),
    )
    trace = _trace(model, data, qpos=qpos, qvel=qvel, duration_sec=1.2)
    minimum_post_distance = float(
        np.min(
            np.hypot(
                trace["position"][:, 0] - 8.5,
                trace["position"][:, 1] - y,
            )
        )
    )
    reverse_observed = bool(np.any(trace["velocity"][:, 0] < -0.2))
    passed = (
        minimum_post_distance <= spec.ball_radius_m + spec.goal_frame_radius_m + 0.01
        and reverse_observed
    )
    case = RealityPackCase(
        case_id="goal_frame_rebound",
        passed=passed,
        metrics={
            "minimum_post_center_distance_m": minimum_post_distance,
            "reverse_velocity_observed": reverse_observed,
        },
        thresholds={
            "maximum_contact_distance_m": spec.ball_radius_m + spec.goal_frame_radius_m + 0.01,
            "reverse_velocity_required": True,
        },
    )
    return case, {f"frame_{key}": value for key, value in trace.items()}


def _goal_net_capture(
    spec: IFABRegulationSpec,
) -> tuple[RealityPackCase, dict[str, NDArray[np.float64]]]:
    model, data, body, qpos = _model(spec)
    qvel = _reset_ball(
        model,
        data,
        qpos,
        (6.0, 2.8, 1.9),
        (12.0, 0.0, 1.0),
        (0.0, 12.0 / spec.ball_radius_m, 0.0),
    )

    def net_force() -> None:
        _apply_compliant_net(data, body, qpos, qvel, spec, goal_plane_x_m=8.5)

    trace = _trace(
        model,
        data,
        qpos=qpos,
        qvel=qvel,
        duration_sec=2.5,
        force=net_force,
    )
    x = trace["position"][:, 0]
    crossed = bool(np.any(x >= 8.5))
    back_limit = 8.5 + spec.net_depth_m + spec.ball_radius_m
    maximum_x = float(np.max(x))
    maximum_deflection = max(0.0, maximum_x - back_limit)
    final_speed = float(np.linalg.norm(trace["velocity"][-1]))
    retained = bool(
        8.5 - spec.ball_radius_m <= x[-1] <= back_limit
        and abs(trace["position"][-1, 1]) <= spec.goal_inside_width_m / 2.0
    )
    passed = crossed and maximum_deflection <= 0.12 and retained and final_speed <= 0.75
    case = RealityPackCase(
        case_id="goal_net",
        passed=passed,
        metrics={
            "goal_plane_crossed": crossed,
            "maximum_ball_x_m": maximum_x,
            "maximum_net_deflection_m": maximum_deflection,
            "retained_in_goal": retained,
            "final_speed_mps": final_speed,
        },
        thresholds={
            "maximum_net_deflection_m": 0.12,
            "retained_in_goal_required": True,
            "maximum_final_speed_mps": 0.75,
        },
    )
    return case, {f"net_{key}": value for key, value in trace.items()}


def _apply_compliant_net(
    data: Any,
    body_id: int,
    qpos: int,
    qvel: int,
    spec: IFABRegulationSpec,
    *,
    goal_plane_x_m: float,
) -> None:
    """Apply one-sided spring-damper forces on back, side, and roof netting."""

    data.xfrc_applied[body_id, :] = 0.0
    position = np.asarray(data.qpos[qpos : qpos + 3], dtype=np.float64)
    velocity = np.asarray(data.qvel[qvel : qvel + 3], dtype=np.float64)
    if position[0] < goal_plane_x_m - spec.ball_radius_m:
        return
    force = np.zeros(3, dtype=np.float64)
    back_x = goal_plane_x_m + spec.net_depth_m - spec.ball_radius_m
    if position[0] > back_x:
        force[0] -= spec.net_stiffness_n_m * (position[0] - back_x)
        force[0] -= spec.net_damping_n_s_m * max(0.0, velocity[0])
    side_limit = spec.goal_inside_width_m / 2.0 - spec.ball_radius_m
    side_penetration = abs(float(position[1])) - side_limit
    if side_penetration > 0.0:
        direction = math.copysign(1.0, float(position[1]))
        force[1] -= direction * spec.net_stiffness_n_m * side_penetration
        force[1] -= spec.net_damping_n_s_m * velocity[1]
    roof_z = spec.goal_inside_height_m - spec.ball_radius_m
    if position[2] > roof_z:
        force[2] -= spec.net_stiffness_n_m * (position[2] - roof_z)
        force[2] -= spec.net_damping_n_s_m * max(0.0, velocity[2])
    if position[0] > goal_plane_x_m and velocity[0] < 0.0:
        force[0] -= 0.35 * spec.net_damping_n_s_m * velocity[0]
    data.xfrc_applied[body_id, :3] = np.clip(force, -250.0, 250.0)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _hash_json(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


__all__ = ["RealityPackCase", "RealityPackReport", "run_reality_pack"]
