"""Research-only late-lunge high-save growth and strict replay evidence.

This stage composes three previously isolated capabilities in one continuous
CPU MuJoCo episode: the qualified lateral locomotion foundation, a bounded
lower-body imitation pulse from the pinned Humanoid-Goalkeeper atlas, and the
collision-faithful bimanual tip-over controller.  It is deliberately named a
``lunge`` rather than a flying dive: airborne motion is not inferred from an
upward velocity sample or from rendered pixels.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import (
    G1GoalkeeperConfig,
    G1SharedWorldResult,
    simulate_shared_world,
)
from rosclaw_soccer.training.three_role_aerial_save import (
    ThreeRoleAerialSaveConfig,
    evaluate_three_role_aerial_save,
)
from rosclaw_soccer.training.three_role_save_portfolio import (
    ThreeRoleSaveLane,
    ThreeRoleSavePortfolioConfig,
    three_role_save_lane_kwargs,
)

_DEFAULT_LANE = ThreeRoleSavePortfolioConfig().lanes[0]
_DEFAULT_AERIAL = replace(
    ThreeRoleAerialSaveConfig(),
    minimum_hand_height_m=1.10,
    minimum_goalkeeper_pelvis_height_m=0.65,
)


@dataclass(frozen=True)
class DynamicAerialLungeSaveConfig:
    """Failure-selected, bounded parameters for one research candidate."""

    lane: ThreeRoleSaveLane = _DEFAULT_LANE
    aerial_config: ThreeRoleAerialSaveConfig = _DEFAULT_AERIAL
    activation_lead_sec: float = 0.35
    initial_phase: float = 0.0
    arrival_phase: float = 0.40
    peak_phase: float = 0.40
    blend_in_sec: float = 0.10
    recovery_tail_sec: float = 0.80
    landing_capture_enabled: bool = False
    landing_capture_sec: float = 0.80
    landing_damping_scale: float = 1.50
    lower_body_scale: float = 0.60
    waist_scale: float = 0.20
    arm_scale: float = 0.0
    vertical_punch_force_scale: float = 0.40
    outward_punch_force_scale: float = 0.0
    joint_guard_impact_lead_sec: float = 0.0
    minimum_lunge_span_m: float = 0.14
    minimum_peak_lateral_speed_mps: float = 0.50
    minimum_controlled_pelvis_height_m: float = 0.65
    minimum_recovery_pelvis_height_m: float = 0.72
    minimum_recovery_upright_projection: float = 0.94
    maximum_recovery_linear_speed_mps: float = 0.45
    recovery_window_sec: float = 0.80
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    commercial_use_allowed: bool = False
    schema_version: str = "rosclaw_soccer.dynamic_aerial_lunge_save_config.v3"

    def __post_init__(self) -> None:
        values = (
            self.activation_lead_sec,
            self.initial_phase,
            self.arrival_phase,
            self.peak_phase,
            self.blend_in_sec,
            self.recovery_tail_sec,
            self.landing_capture_sec,
            self.landing_damping_scale,
            self.lower_body_scale,
            self.waist_scale,
            self.arm_scale,
            self.vertical_punch_force_scale,
            self.outward_punch_force_scale,
            self.joint_guard_impact_lead_sec,
            self.minimum_lunge_span_m,
            self.minimum_peak_lateral_speed_mps,
            self.minimum_controlled_pelvis_height_m,
            self.minimum_recovery_pelvis_height_m,
            self.minimum_recovery_upright_projection,
            self.maximum_recovery_linear_speed_mps,
            self.recovery_window_sec,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("dynamic aerial lunge settings must be finite")
        if not 0.10 <= self.activation_lead_sec <= 0.60:
            raise ValueError("dynamic aerial lunge activation lead is invalid")
        if not 0.0 <= self.initial_phase <= 0.30:
            raise ValueError("dynamic aerial lunge initial phase is invalid")
        if not 0.40 <= self.arrival_phase <= self.peak_phase <= 0.85:
            raise ValueError("dynamic aerial lunge phase envelope is invalid")
        if self.initial_phase >= self.arrival_phase:
            raise ValueError("dynamic aerial lunge initial phase must precede arrival")
        if not 0.10 <= self.blend_in_sec <= 0.30:
            raise ValueError("dynamic aerial lunge blend-in is invalid")
        if not 0.20 <= self.recovery_tail_sec <= 1.20:
            raise ValueError("dynamic aerial lunge recovery tail is invalid")
        if not isinstance(self.landing_capture_enabled, bool):
            raise ValueError("dynamic aerial lunge landing capture flag is invalid")
        if not 0.20 <= self.landing_capture_sec <= 1.50:
            raise ValueError("dynamic aerial lunge landing capture duration is invalid")
        if not 1.0 <= self.landing_damping_scale <= 3.0:
            raise ValueError("dynamic aerial lunge landing damping is invalid")
        if not 0.0 <= self.arm_scale <= 0.20:
            raise ValueError("dynamic aerial lunge cannot replace the qualified arm skill")
        if not 0.20 <= self.lower_body_scale <= 1.0 or not 0.0 <= self.waist_scale <= 0.50:
            raise ValueError("dynamic aerial lunge body authority is invalid")
        if not 0.0 <= self.vertical_punch_force_scale <= 1.0:
            raise ValueError("dynamic aerial lunge vertical punch is invalid")
        if not 0.0 <= self.outward_punch_force_scale <= 0.75:
            raise ValueError("dynamic aerial lunge outward punch is invalid")
        if not 0.0 <= self.joint_guard_impact_lead_sec <= 0.08:
            raise ValueError("dynamic aerial lunge impact guard lead is invalid")
        if not 0.10 <= self.minimum_lunge_span_m <= 0.80:
            raise ValueError("dynamic aerial lunge span gate is invalid")
        if not 0.20 <= self.minimum_peak_lateral_speed_mps <= 1.50:
            raise ValueError("dynamic aerial lunge speed gate is invalid")
        if not 0.55 <= self.minimum_controlled_pelvis_height_m <= 0.75:
            raise ValueError("dynamic aerial lunge pelvis envelope is invalid")
        if not self.minimum_controlled_pelvis_height_m <= self.minimum_recovery_pelvis_height_m:
            raise ValueError("dynamic aerial lunge recovery pelvis gate is invalid")
        if not 0.80 <= self.minimum_recovery_upright_projection <= 1.0:
            raise ValueError("dynamic aerial lunge upright gate is invalid")
        if not 0.10 <= self.maximum_recovery_linear_speed_mps <= 0.80:
            raise ValueError("dynamic aerial lunge recovery speed gate is invalid")
        if not 0.40 <= self.recovery_window_sec <= 1.50:
            raise ValueError("dynamic aerial lunge recovery window is invalid")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.commercial_use_allowed
        ):
            raise ValueError("dynamic aerial lunge must remain non-commercial SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def dynamic_aerial_lunge_kwargs(
    *,
    striker_actor_path: Path,
    goalkeeper_actor_path: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    dive_source_checkout: Path,
    config: DynamicAerialLungeSaveConfig | None = None,
) -> dict[str, Any]:
    """Compose the learned modules without granting the atlas arm authority."""

    active = config or DynamicAerialLungeSaveConfig()
    source = dive_source_checkout.expanduser().resolve()
    if not source.is_dir():
        raise ValueError("dynamic aerial lunge dive source is unavailable")
    kwargs: dict[str, Any] = three_role_save_lane_kwargs(
        lane=active.lane,
        striker_actor_path=striker_actor_path,
        goalkeeper_actor_path=goalkeeper_actor_path,
        gmt_model_path=gmt_model_path,
        gmt_skill_path=gmt_skill_path,
        aerial_config=active.aerial_config,
    )
    goalkeeper = kwargs.get("goalkeeper_config")
    if not isinstance(goalkeeper, G1GoalkeeperConfig):
        raise RuntimeError("dynamic aerial lunge goalkeeper parent is unavailable")
    kwargs["goalkeeper_config"] = replace(
        goalkeeper,
        actor_bimanual_punch_vertical_force_scale=active.vertical_punch_force_scale,
        actor_bimanual_punch_outward_force_scale=active.outward_punch_force_scale,
        joint_guard_impact_lead_sec=active.joint_guard_impact_lead_sec,
        balanced_dive_source_checkout=source,
        balanced_dive_blend=1.0,
        balanced_dive_minimum_lateral_error_m=0.20,
        balanced_dive_activation_lead_sec=active.activation_lead_sec,
        balanced_dive_initial_phase=active.initial_phase,
        balanced_dive_phase_at_arrival=active.arrival_phase,
        balanced_dive_peak_phase=active.peak_phase,
        balanced_dive_lower_body_scale=active.lower_body_scale,
        balanced_dive_waist_scale=active.waist_scale,
        balanced_dive_arm_scale=active.arm_scale,
        balanced_dive_blend_in_sec=active.blend_in_sec,
        balanced_dive_recovery_tail_sec=active.recovery_tail_sec,
        balanced_dive_landing_capture_enabled=active.landing_capture_enabled,
        balanced_dive_landing_capture_sec=active.landing_capture_sec,
        balanced_dive_landing_damping_scale=active.landing_damping_scale,
        post_contact_stabilization_enabled=True,
    )
    return kwargs


def evaluate_dynamic_aerial_lunge_save(
    *,
    result: G1SharedWorldResult,
    trajectory: dict[str, np.ndarray],
    config: DynamicAerialLungeSaveConfig,
) -> dict[str, Any]:
    """Require a true glove save, bounded lunge and measured stable tail."""

    base = evaluate_three_role_aerial_save(
        result=result,
        trajectory=trajectory,
        config=config.aerial_config,
    )
    if not isinstance(base.get("gates"), dict):
        return {"passed": False, "reason": base.get("reason", "base aerial-save gate failed")}
    time = np.asarray(trajectory["time"], dtype=np.float64)
    pelvis = np.asarray(trajectory["goalkeeper_pelvis_pose"], dtype=np.float64)
    dive_blend = np.asarray(trajectory["goalkeeper_balanced_dive_blend"], dtype=np.float64)
    if (
        time.ndim != 1
        or pelvis.shape != (time.size, 7)
        or dive_blend.shape != time.shape
        or time.size < 3
        or not all(np.all(np.isfinite(value)) for value in (time, pelvis, dive_blend))
    ):
        return {"passed": False, "reason": "dynamic aerial lunge trajectory is invalid"}
    root_velocity_value = trajectory.get("goalkeeper_root_velocity")
    root_velocity = (
        None
        if root_velocity_value is None
        else np.asarray(root_velocity_value, dtype=np.float64)
    )
    if root_velocity is not None and (
        root_velocity.shape != (time.size, 6) or not np.all(np.isfinite(root_velocity))
    ):
        return {"passed": False, "reason": "dynamic aerial lunge root velocity is invalid"}
    velocity = (
        np.gradient(pelvis[:, :3], time, axis=0)
        if root_velocity is None
        else root_velocity[:, :3]
    )
    active_indices = np.flatnonzero(dive_blend > 1.0e-6)
    if active_indices.size:
        start = max(0, int(active_indices[0]) - 1)
        stop = min(time.size, int(active_indices[-1]) + 2)
        lunge_span = float(np.ptp(pelvis[start:stop, 1]))
        lunge_peak_speed = float(np.max(np.abs(velocity[start:stop, 1])))
    else:
        lunge_span = 0.0
        lunge_peak_speed = 0.0
    quaternion = pelvis[:, 3:7]
    upright = 1.0 - 2.0 * (np.square(quaternion[:, 1]) + np.square(quaternion[:, 2]))
    tail_start = int(np.searchsorted(time, time[-1] - config.recovery_window_sec, side="left"))
    tail_speed = np.linalg.norm(velocity[tail_start:, :3], axis=1)
    metrics: dict[str, Any] = {
        "lunge_span_m": lunge_span,
        "lunge_peak_lateral_speed_mps": lunge_peak_speed,
        "minimum_pelvis_height_m": float(np.min(pelvis[:, 2])),
        "recovery_minimum_pelvis_height_m": float(np.min(pelvis[tail_start:, 2])),
        "recovery_minimum_upright_projection": float(np.min(upright[tail_start:])),
        "recovery_maximum_linear_speed_mps": float(np.max(tail_speed)),
        "final_pelvis_height_m": float(pelvis[-1, 2]),
        "final_upright_projection": float(upright[-1]),
        "velocity_authority": (
            "CONTROL_RATE_FINITE_DIFFERENCE" if root_velocity is None else "MUJOCO_QVEL"
        ),
        "airborne_claimed": False,
    }
    gates = dict(base["gates"])
    gates.update(
        {
            "data_driven_lower_body_lunge": bool(
                result.goalkeeper_balanced_dive_seed_hash is not None
                and result.goalkeeper_balanced_dive_peak_blend > 0.99
                and lunge_span >= config.minimum_lunge_span_m
                and lunge_peak_speed >= config.minimum_peak_lateral_speed_mps
            ),
            "controlled_lunge_envelope": bool(
                metrics["minimum_pelvis_height_m"]
                >= config.minimum_controlled_pelvis_height_m
            ),
            "post_save_recovered": bool(
                metrics["recovery_minimum_pelvis_height_m"]
                >= config.minimum_recovery_pelvis_height_m
                and metrics["recovery_minimum_upright_projection"]
                >= config.minimum_recovery_upright_projection
                and metrics["recovery_maximum_linear_speed_mps"]
                <= config.maximum_recovery_linear_speed_mps
            ),
        }
    )
    return {
        **base,
        "passed": bool(all(gates.values())),
        "gates": gates,
        "dynamic_metrics": metrics,
        "claim": "STABLE_LATE_LUNGE_TIP_OVER_NOT_AIRBORNE_DIVE",
    }


def run_dynamic_aerial_lunge_evidence(
    *,
    asset_root: Path,
    striker_actor_path: Path,
    goalkeeper_actor_path: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    dive_source_checkout: Path,
    output_dir: Path,
    source_checkout: Path,
    config: DynamicAerialLungeSaveConfig | None = None,
) -> dict[str, Any]:
    """Run two CPU replays and freeze only an identical all-gate result."""

    active = config or DynamicAerialLungeSaveConfig()
    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    dive_source = dive_source_checkout.expanduser().resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("dynamic aerial lunge evidence must be new and outside the checkout")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    kwargs = dynamic_aerial_lunge_kwargs(
        striker_actor_path=striker_actor_path,
        goalkeeper_actor_path=goalkeeper_actor_path,
        gmt_model_path=gmt_model_path,
        gmt_skill_path=gmt_skill_path,
        dive_source_checkout=dive_source,
        config=active,
    )
    artifacts = {
        "striker_actor_hash": hash_bytes(striker_actor_path.read_bytes()),
        "goalkeeper_actor_hash": hash_bytes(goalkeeper_actor_path.read_bytes()),
        "gmt_model_hash": hash_bytes(gmt_model_path.read_bytes()),
        "gmt_skill_hash": hash_bytes(gmt_skill_path.read_bytes()),
        "dive_source_commit": _git_head(dive_source),
        "dive_source_license_hash": hash_bytes((dive_source / "LICENSE").read_bytes()),
    }
    request = {
        "schema_version": "rosclaw_soccer.dynamic_aerial_lunge_request.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "goal_spec": asdict(kwargs["goal_spec"]),
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "artifacts": artifacts,
        "source_commit": _git_head(checkout),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
    }
    output.mkdir(parents=True)
    _write_json(output / "request.json", request)
    first_result, first_trajectory = simulate_shared_world(asset_root, **kwargs)
    replay_result, replay_trajectory = simulate_shared_world(asset_root, **kwargs)
    first = evaluate_dynamic_aerial_lunge_save(
        result=first_result,
        trajectory=first_trajectory,
        config=active,
    )
    replay = evaluate_dynamic_aerial_lunge_save(
        result=replay_result,
        trajectory=replay_trajectory,
        config=active,
    )
    strict_replay = bool(
        first_result.to_dict() == replay_result.to_dict()
        and trajectory_digest(first_trajectory) == trajectory_digest(replay_trajectory)
    )
    trajectory_path = output / "dynamic-lunge-trajectory.npz"
    np.savez_compressed(trajectory_path, **replay_trajectory)  # type: ignore[arg-type]
    passed = bool(first.get("passed") and replay.get("passed") and strict_replay)
    report = {
        "schema_version": "rosclaw_soccer.dynamic_aerial_lunge_evidence.v1",
        "passed": passed,
        "promotion_status": "FROZEN_RESEARCH_DEMO" if passed else "REJECTED_DEVELOPMENT",
        "strict_replay": strict_replay,
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "pixels_used_for_scoring": False,
        "single_shared_ball": True,
        "simultaneous_three_body_physics": True,
        "claim": "STABLE_LATE_LUNGE_TIP_OVER_NOT_AIRBORNE_DIVE",
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "trajectory_file": trajectory_path.name,
        "trajectory_hash": hash_bytes(trajectory_path.read_bytes()),
        "implementation_hash": _implementation_hash(),
        "artifacts": artifacts,
        "first": first,
        "replay": replay,
    }
    _write_json(output / "evidence.json", report)
    return report


def _implementation_hash() -> str:
    shared = Path(__file__).parents[1] / "skills" / "team" / "shared_world.py"
    return str(
        hash_json(
            {
                "evidence_loop": hash_bytes(Path(__file__).read_bytes()),
                "shared_world": hash_bytes(shared.read_bytes()),
            }
        )
    )


def _git_head(checkout: Path) -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "DynamicAerialLungeSaveConfig",
    "dynamic_aerial_lunge_kwargs",
    "evaluate_dynamic_aerial_lunge_save",
    "run_dynamic_aerial_lunge_evidence",
]
