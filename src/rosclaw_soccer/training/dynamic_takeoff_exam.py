"""Contact-grounded take-off and landing exam for integrated goalkeeper saves.

Pelvis velocity alone is not flight.  This exam requires a continuous interval
with neither foot touching the MuJoCo ground, a measured upward take-off, glove
contact during that interval, a subsequent foot landing, and stable recovery.
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
    G1SharedWorldResult,
    simulate_shared_world,
)
from rosclaw_soccer.training.dynamic_aerial_lunge_save import (
    DynamicAerialLungeSaveConfig,
    dynamic_aerial_lunge_kwargs,
    evaluate_dynamic_aerial_lunge_save,
)

_TAKEOFF_AERIAL = replace(
    DynamicAerialLungeSaveConfig().aerial_config,
    minimum_goalkeeper_pelvis_height_m=0.60,
    maximum_post_contact_speed_mps=15.0,
)
_TAKEOFF_LUNGE = replace(
    DynamicAerialLungeSaveConfig(),
    aerial_config=_TAKEOFF_AERIAL,
    activation_lead_sec=0.38,
    initial_phase=0.10,
    arrival_phase=0.85,
    peak_phase=0.85,
    recovery_tail_sec=0.30,
    lower_body_scale=0.67,
    waist_scale=0.20,
    vertical_punch_force_scale=0.80,
    minimum_lunge_span_m=0.10,
    minimum_controlled_pelvis_height_m=0.55,
    minimum_recovery_pelvis_height_m=0.70,
    maximum_recovery_linear_speed_mps=0.50,
)


@dataclass(frozen=True)
class DynamicTakeoffExamConfig:
    """Physics-only gates for a true, recoverable airborne save."""

    lunge_config: DynamicAerialLungeSaveConfig = _TAKEOFF_LUNGE
    minimum_airborne_duration_sec: float = 0.06
    minimum_takeoff_vertical_speed_mps: float = 0.12
    minimum_flight_pelvis_rise_m: float = 0.015
    maximum_landing_vertical_speed_mps: float = 1.20
    maximum_landing_angular_speed_rad_s: float = 3.50
    contact_time_tolerance_sec: float = 0.03
    search_lead_sec: float = 0.30
    search_tail_sec: float = 1.00
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    commercial_use_allowed: bool = False
    schema_version: str = "rosclaw_soccer.dynamic_takeoff_exam_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.minimum_airborne_duration_sec,
            self.minimum_takeoff_vertical_speed_mps,
            self.minimum_flight_pelvis_rise_m,
            self.maximum_landing_vertical_speed_mps,
            self.maximum_landing_angular_speed_rad_s,
            self.contact_time_tolerance_sec,
            self.search_lead_sec,
            self.search_tail_sec,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("dynamic takeoff settings must be finite and positive")
        if not 0.04 <= self.minimum_airborne_duration_sec <= 0.40:
            raise ValueError("dynamic takeoff airborne duration is invalid")
        if not 0.05 <= self.minimum_takeoff_vertical_speed_mps <= 1.50:
            raise ValueError("dynamic takeoff vertical speed is invalid")
        if not 0.005 <= self.minimum_flight_pelvis_rise_m <= 0.30:
            raise ValueError("dynamic takeoff pelvis rise is invalid")
        if not 0.30 <= self.maximum_landing_vertical_speed_mps <= 2.50:
            raise ValueError("dynamic takeoff landing speed is invalid")
        if not 1.0 <= self.maximum_landing_angular_speed_rad_s <= 6.0:
            raise ValueError("dynamic takeoff landing angular speed is invalid")
        if not 0.01 <= self.contact_time_tolerance_sec <= 0.06:
            raise ValueError("dynamic takeoff contact tolerance is invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("dynamic takeoff exam is SIM_ONLY")
        if self.commercial_use_allowed:
            raise ValueError("dynamic takeoff source is research-only")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def expanded_dynamic_takeoff_config() -> DynamicTakeoffExamConfig:
    """Return the failure-selected S91 longer-flight/landing-capture contract."""

    base = DynamicTakeoffExamConfig()
    return replace(
        base,
        lunge_config=replace(
            base.lunge_config,
            lower_body_scale=0.78,
            landing_capture_enabled=True,
            landing_capture_sec=0.20,
            landing_damping_scale=1.50,
        ),
        minimum_airborne_duration_sec=0.15,
        minimum_flight_pelvis_rise_m=0.035,
    )


def evaluate_dynamic_takeoff_save(
    *,
    result: G1SharedWorldResult,
    trajectory: dict[str, np.ndarray],
    config: DynamicTakeoffExamConfig | None = None,
) -> dict[str, Any]:
    """Require ground-contact-confirmed flight, landing and recovery."""

    active = config or DynamicTakeoffExamConfig()
    base = evaluate_dynamic_aerial_lunge_save(
        result=result,
        trajectory=trajectory,
        config=active.lunge_config,
    )
    required = {
        "time",
        "goalkeeper_pelvis_pose",
        "goalkeeper_root_velocity",
        "goalkeeper_foot_contact",
    }
    if not required <= set(trajectory):
        return {
            "passed": False,
            "reason": "goalkeeper takeoff telemetry is incomplete",
            "base": base,
        }
    time = np.asarray(trajectory["time"], dtype=np.float64)
    pelvis = np.asarray(trajectory["goalkeeper_pelvis_pose"], dtype=np.float64)
    root_velocity = np.asarray(trajectory["goalkeeper_root_velocity"], dtype=np.float64)
    foot_contact = np.asarray(trajectory["goalkeeper_foot_contact"], dtype=np.bool_)
    if (
        time.ndim != 1
        or pelvis.shape != (time.size, 7)
        or root_velocity.shape != (time.size, 6)
        or foot_contact.shape != (time.size, 2)
        or time.size < 4
        or not np.all(np.diff(time) > 0.0)
        or not np.all(np.isfinite(time))
        or not np.all(np.isfinite(pelvis))
        or not np.all(np.isfinite(root_velocity))
    ):
        return {
            "passed": False,
            "reason": "goalkeeper takeoff telemetry is invalid",
            "base": base,
        }
    shot_time = result.shot_contact_time_sec
    glove_time = result.goalkeeper_glove_contact_time_sec
    if shot_time is None or glove_time is None or glove_time <= shot_time:
        return {
            "passed": False,
            "reason": "goalkeeper takeoff contact order is invalid",
            "base": base,
        }
    window = (time >= shot_time - active.search_lead_sec) & (
        time <= glove_time + active.search_tail_sec
    )
    airborne = ~np.any(foot_contact, axis=1) & window
    flight_start, flight_stop = _longest_true_run(airborne)
    if flight_start is None or flight_stop is None:
        return {
            "passed": False,
            "reason": "no ground-contact-confirmed flight interval",
            "base": base,
        }
    dt = float(np.median(np.diff(time)))
    flight_duration = (flight_stop - flight_start + 1) * dt
    takeoff_start = max(0, flight_start - max(1, int(math.ceil(0.10 / dt))))
    takeoff_stop = min(time.size, flight_start + max(2, int(math.ceil(0.04 / dt))))
    takeoff_vertical_speed = float(np.max(root_velocity[takeoff_start:takeoff_stop, 2]))
    launch_height = float(pelvis[max(0, flight_start - 1), 2])
    flight_pelvis_rise = float(np.max(pelvis[flight_start : flight_stop + 1, 2]) - launch_height)
    landing_candidates = np.flatnonzero(
        np.any(foot_contact[flight_stop + 1 :], axis=1)
    )
    landing_index = (
        None
        if landing_candidates.size == 0
        else flight_stop + 1 + int(landing_candidates[0])
    )
    landing_vertical_speed = (
        math.inf if landing_index is None else abs(float(root_velocity[landing_index, 2]))
    )
    landing_angular_speed = (
        math.inf
        if landing_index is None
        else float(np.linalg.norm(root_velocity[landing_index, 3:6]))
    )
    glove_during_flight = bool(
        time[flight_start] - active.contact_time_tolerance_sec
        <= glove_time
        <= time[flight_stop] + active.contact_time_tolerance_sec
    )
    base_gates = base.get("gates")
    gates = {
        "base_save_and_recovery": bool(
            base.get("passed") is True
            and isinstance(base_gates, dict)
            and base_gates.get("post_save_recovered") is True
        ),
        "ground_contact_confirmed_airborne": bool(
            flight_duration + 1.0e-12 >= active.minimum_airborne_duration_sec
        ),
        "upward_takeoff": bool(
            takeoff_vertical_speed >= active.minimum_takeoff_vertical_speed_mps
            and flight_pelvis_rise >= active.minimum_flight_pelvis_rise_m
        ),
        "glove_contact_during_flight": glove_during_flight,
        "foot_landing_observed": landing_index is not None,
        "bounded_landing": bool(
            landing_vertical_speed <= active.maximum_landing_vertical_speed_mps
            and landing_angular_speed <= active.maximum_landing_angular_speed_rad_s
        ),
    }
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "metrics": {
            "airborne_duration_sec": flight_duration,
            "airborne_start_sec": float(time[flight_start]),
            "airborne_stop_sec": float(time[flight_stop]),
            "takeoff_peak_vertical_speed_mps": takeoff_vertical_speed,
            "flight_pelvis_rise_m": flight_pelvis_rise,
            "landing_time_sec": None if landing_index is None else float(time[landing_index]),
            "landing_vertical_speed_mps": landing_vertical_speed,
            "landing_angular_speed_rad_s": landing_angular_speed,
            "glove_contact_time_sec": glove_time,
        },
        "base": base,
        "claim": "TRUE_AIRBORNE_SAVE_ONLY_IF_ALL_CONTACT_GATES_PASS",
    }


def _longest_true_run(mask: np.ndarray) -> tuple[int | None, int | None]:
    """Return inclusive bounds of the longest true run."""

    values = np.asarray(mask, dtype=np.bool_)
    if values.ndim != 1:
        raise ValueError("airborne mask must be one-dimensional")
    padded = np.concatenate((np.asarray((False,)), values, np.asarray((False,))))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    if edges.size == 0:
        return None, None
    starts = edges[0::2]
    stops = edges[1::2] - 1
    if starts.size == 0:
        return None, None
    lengths = stops - starts + 1
    winner = int(np.argmax(lengths))
    return int(starts[winner]), int(stops[winner])


def run_dynamic_takeoff_evidence(
    *,
    asset_root: Path,
    striker_actor_path: Path,
    goalkeeper_actor_path: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    dive_source_checkout: Path,
    output_dir: Path,
    source_checkout: Path,
    config: DynamicTakeoffExamConfig | None = None,
) -> dict[str, Any]:
    """Freeze only two identical CPU-MuJoCo airborne-save replays."""

    active = config or DynamicTakeoffExamConfig()
    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    dive_source = dive_source_checkout.expanduser().resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("dynamic takeoff evidence must be new and outside the checkout")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    kwargs = dynamic_aerial_lunge_kwargs(
        striker_actor_path=striker_actor_path,
        goalkeeper_actor_path=goalkeeper_actor_path,
        gmt_model_path=gmt_model_path,
        gmt_skill_path=gmt_skill_path,
        dive_source_checkout=dive_source,
        config=active.lunge_config,
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
        "schema_version": "rosclaw_soccer.dynamic_takeoff_request.v1",
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
    first = evaluate_dynamic_takeoff_save(
        result=first_result,
        trajectory=first_trajectory,
        config=active,
    )
    replay = evaluate_dynamic_takeoff_save(
        result=replay_result,
        trajectory=replay_trajectory,
        config=active,
    )
    strict_replay = bool(
        first_result.to_dict() == replay_result.to_dict()
        and trajectory_digest(first_trajectory) == trajectory_digest(replay_trajectory)
    )
    trajectory_path = output / "dynamic-takeoff-trajectory.npz"
    np.savez_compressed(trajectory_path, **replay_trajectory)  # type: ignore[arg-type]
    passed = bool(first.get("passed") and replay.get("passed") and strict_replay)
    report = {
        "schema_version": "rosclaw_soccer.dynamic_takeoff_evidence.v1",
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
        "claim": "TRUE_AIRBORNE_SAVE_WITH_FOOT_CONTACT_GROUNDED_LANDING",
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
    source_root = Path(__file__).parents[1]
    return str(
        hash_json(
            {
                "takeoff_exam": hash_bytes(Path(__file__).read_bytes()),
                "dynamic_lunge": hash_bytes(
                    (source_root / "training" / "dynamic_aerial_lunge_save.py").read_bytes()
                ),
                "shared_world": hash_bytes(
                    (source_root / "skills" / "team" / "shared_world.py").read_bytes()
                ),
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
    "DynamicTakeoffExamConfig",
    "expanded_dynamic_takeoff_config",
    "evaluate_dynamic_takeoff_save",
    "run_dynamic_takeoff_evidence",
]
