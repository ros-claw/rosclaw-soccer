"""Failure-driven reachability audit for the current high-corner controller family.

This is a diagnosis, not a candidate generator.  It replays one frozen actor
and three bounded controller-family counterfactuals on the same high-corner
MJWarp seed panel.  If none can make a safe physical save, the next curriculum
must discover a new lateral-locomotion/dive expert instead of merely extending
PPO on the same residual topology.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.goalkeeper_mjwarp import goalkeeper_world_config
from rosclaw_soccer.training.goalkeeper_physics_ppo import (
    GoalkeeperPhysicsPPOConfig,
    _build_actor_critic,
    _build_environment,
    _load_actor_critic_state,
    _with_episode_duration_override,
    _with_first_shot_release_override,
    _with_recovery_progress_override,
    _with_root_angular_penalty_override,
    _with_save_event_bonus_override,
)

_ROUTES = (
    "bounded-parent",
    "learned-candidate",
    "full-drive-probe",
    "full-drive-lunge-probe",
)


@dataclass(frozen=True)
class HighCornerReachabilityAuditConfig:
    seeds: tuple[int, ...] = (91_031, 91_051)
    environment_count: int = 16
    flight_time_range_sec: tuple[float, float] = (0.48, 0.62)
    full_drive_scale: float = 1.0
    full_option_gate: float = 0.80
    lunge_blend: float = 0.80
    maximum_safe_failed_rate: float = 0.0
    maximum_safe_root_angular_speed_rad_s: float = 3.50
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.high_corner_reachability_audit_config.v1"

    def __post_init__(self) -> None:
        if (
            len(self.seeds) < 2
            or len(set(self.seeds)) != len(self.seeds)
            or any(isinstance(seed, bool) or not 0 <= seed < 2**31 for seed in self.seeds)
        ):
            raise ValueError("high-corner audit needs at least two distinct valid seeds")
        if not 8 <= self.environment_count <= 256:
            raise ValueError("high-corner audit environment count is invalid")
        low, high = self.flight_time_range_sec
        values = (
            low,
            high,
            self.full_drive_scale,
            self.full_option_gate,
            self.lunge_blend,
            self.maximum_safe_failed_rate,
            self.maximum_safe_root_angular_speed_rad_s,
        )
        if any(not math.isfinite(value) for value in values) or not 0.30 <= low < high <= 0.80:
            raise ValueError("high-corner audit settings must be finite and bounded")
        if not 0.80 <= self.full_drive_scale <= 1.0:
            raise ValueError("high-corner audit full-drive probe is invalid")
        if not 0.50 <= self.full_option_gate <= 0.80:
            raise ValueError("high-corner audit full option gate is invalid")
        if not 0.50 <= self.lunge_blend <= 1.0:
            raise ValueError("high-corner audit lunge probe is invalid")
        if not 0.0 <= self.maximum_safe_failed_rate <= 0.05:
            raise ValueError("high-corner audit failed-rate ceiling is invalid")
        if not 2.0 <= self.maximum_safe_root_angular_speed_rad_s <= 3.5:
            raise ValueError("high-corner audit angular-speed ceiling is invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("high-corner audit must remain SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def classify_high_corner_reachability(
    routes: dict[str, dict[str, Any]],
    *,
    maximum_safe_failed_rate: float = 0.0,
    maximum_safe_root_angular_speed_rad_s: float = 3.5,
) -> str:
    """Choose the next architecture from physical reach and safety outcomes."""

    if set(routes) != set(_ROUTES):
        raise ValueError("high-corner audit route set is incomplete")
    safe_save = False
    unsafe_save = False
    for route in routes.values():
        required = (
            "first_save_rate",
            "failed_rate",
            "maximum_root_angular_speed_rad_s",
            "finite_state",
            "paired_outcome_consistent",
        )
        if any(name not in route for name in required):
            raise ValueError("high-corner audit route metrics are incomplete")
        save = float(route["first_save_rate"])
        failed = float(route["failed_rate"])
        angular = float(route["maximum_root_angular_speed_rad_s"])
        if any(not math.isfinite(value) for value in (save, failed, angular)):
            raise ValueError("high-corner audit route metrics are non-finite")
        safe = bool(
            route["finite_state"]
            and route["paired_outcome_consistent"]
            and failed <= maximum_safe_failed_rate
            and angular <= maximum_safe_root_angular_speed_rad_s
        )
        safe_save |= save > 0.0 and safe
        unsafe_save |= save > 0.0 and not safe
    if safe_save:
        return "CURRENT_CONTROLLER_FAMILY_REACHABLE"
    if unsafe_save:
        return "UNSAFE_ACTION_BUDGET_ONLY"
    return "NEW_LATERAL_LOCOMOTION_DIVE_EXPERT_REQUIRED"


def run_high_corner_reachability_audit(
    *,
    asset_root: Path,
    locomotion_policy_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    source_checkout: Path,
    config: HighCornerReachabilityAuditConfig | None = None,
) -> dict[str, Any]:
    """Run paired reachability probes and write external diagnostic evidence."""

    import torch
    from torch import nn

    active = config or HighCornerReachabilityAuditConfig()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    checkpoint = checkpoint_path.expanduser().resolve()
    locomotion = locomotion_policy_path.expanduser().resolve()
    root = asset_root.expanduser().resolve()
    if (
        output.exists()
        or output.suffix.lower() != ".json"
        or output == checkout
        or checkout in output.parents
        or not checkout.is_dir()
    ):
        raise ValueError("high-corner audit output must be a new external JSON file")
    if not checkpoint.is_file() or not locomotion.is_file():
        raise ValueError("high-corner audit input policy is unavailable")
    qualification = qualify_g1_assets(root)
    qualification.require_eligible()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    training_payload = payload.get("training_config")
    if not isinstance(training_payload, dict):
        raise ValueError("high-corner audit checkpoint training config is invalid")
    parent = GoalkeeperPhysicsPPOConfig(**training_payload)
    model = _build_actor_critic(
        torch,
        nn,
        int(payload["observation_size"]),
        int(payload["action_size"]),
        int(payload["hidden_size"]),
    ).to(device)
    _load_actor_critic_state(model, payload["state_dict"])
    model.eval()
    route_configs = {
        "bounded-parent": parent,
        "learned-candidate": parent,
        "full-drive-probe": replace(
            parent,
            targeted_dive_lateral_drive_scale=active.full_drive_scale,
            targeted_dive_minimum_option_gate=active.full_option_gate,
        ),
        "full-drive-lunge-probe": replace(
            parent,
            targeted_dive_lateral_drive_scale=active.full_drive_scale,
            targeted_dive_minimum_option_gate=active.full_option_gate,
            targeted_dive_runtime_lateral_lunge_blend=active.lunge_blend,
        ),
    }
    routes: dict[str, dict[str, Any]] = {}
    for route_name in _ROUTES:
        route_config = route_configs[route_name]
        first = _run_route(
            torch=torch,
            model=model,
            learned=route_name == "learned-candidate",
            active=route_config,
            audit=active,
            asset_root=root,
            locomotion_policy_path=locomotion,
            device=device,
        )
        replay = _run_route(
            torch=torch,
            model=model,
            learned=route_name == "learned-candidate",
            active=route_config,
            audit=active,
            asset_root=root,
            locomotion_policy_path=locomotion,
            device=device,
        )
        replay_delta = _maximum_replay_metric_delta(first, replay)
        paired_outcome = _diagnostic_outcome(
            first,
            maximum_safe_failed_rate=active.maximum_safe_failed_rate,
            maximum_safe_root_angular_speed_rad_s=(active.maximum_safe_root_angular_speed_rad_s),
        ) == _diagnostic_outcome(
            replay,
            maximum_safe_failed_rate=active.maximum_safe_failed_rate,
            maximum_safe_root_angular_speed_rad_s=(active.maximum_safe_root_angular_speed_rad_s),
        )
        routes[route_name] = {
            **first,
            "strict_replay": first == replay,
            "maximum_paired_replay_metric_delta": replay_delta,
            "paired_outcome_consistent": paired_outcome,
            "replay_metrics": replay,
            "controller_config_hash": route_config.config_hash,
        }
    decision = classify_high_corner_reachability(
        routes,
        maximum_safe_failed_rate=active.maximum_safe_failed_rate,
        maximum_safe_root_angular_speed_rad_s=(active.maximum_safe_root_angular_speed_rad_s),
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.high_corner_reachability_audit.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "decision": decision,
        "routes": routes,
        "episodes_per_route": active.environment_count * len(active.seeds),
        "source_files": {
            str(checkpoint): hash_bytes(checkpoint.read_bytes()),
            str(locomotion): hash_bytes(locomotion.read_bytes()),
        },
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "implementation_hash": hash_bytes(Path(__file__).read_bytes()),
        "physics_backend": "mujoco_warp",
        "physics_authority": "DIAGNOSTIC_ONLY",
        "fresh_physics_performed": True,
        "source_actor_frozen": True,
        "candidate_generated": False,
        "promotion_eligible": False,
        "strict_replay": all(route["strict_replay"] for route in routes.values()),
        "paired_outcome_consistent": all(
            route["paired_outcome_consistent"] for route in routes.values()
        ),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "mujoco_warp_version": version("mujoco-warp"),
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "pixels_used_for_scoring": False,
        "commercial_use_allowed": False,
    }
    report["report_hash"] = hash_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    validate_high_corner_reachability_audit(output)
    return report


def validate_high_corner_reachability_audit(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("high-corner audit must be a JSON object")
    report_hash = payload.pop("report_hash", None)
    try:
        config = payload.get("config")
        routes = payload.get("routes")
        files = payload.get("source_files")
        if (
            not isinstance(config, dict)
            or not isinstance(routes, dict)
            or not isinstance(files, dict)
        ):
            raise ValueError("high-corner audit bindings are incomplete")
        for path_value, expected_hash in files.items():
            bound = Path(path_value).expanduser().resolve()
            if not bound.is_file() or hash_bytes(bound.read_bytes()) != expected_hash:
                raise ValueError("high-corner audit source binding changed")
        safe_failed_rate = float(config["maximum_safe_failed_rate"])
        safe_angular_speed = float(config["maximum_safe_root_angular_speed_rad_s"])
        if any(
            not isinstance(route.get("replay_metrics"), dict)
            or route.get("paired_outcome_consistent")
            is not (
                _diagnostic_outcome(
                    route,
                    maximum_safe_failed_rate=safe_failed_rate,
                    maximum_safe_root_angular_speed_rad_s=safe_angular_speed,
                )
                == _diagnostic_outcome(
                    route["replay_metrics"],
                    maximum_safe_failed_rate=safe_failed_rate,
                    maximum_safe_root_angular_speed_rad_s=safe_angular_speed,
                )
            )
            for route in routes.values()
        ):
            raise ValueError("high-corner audit paired outcome contract changed")
        decision = classify_high_corner_reachability(
            routes,
            maximum_safe_failed_rate=float(config["maximum_safe_failed_rate"]),
            maximum_safe_root_angular_speed_rad_s=float(
                config["maximum_safe_root_angular_speed_rad_s"]
            ),
        )
        if (
            payload.get("schema_version") != "rosclaw_soccer.high_corner_reachability_audit.v1"
            or payload.get("config_hash") != hash_json(config)
            or payload.get("implementation_hash") != hash_bytes(Path(__file__).read_bytes())
            or payload.get("decision") != decision
            or payload.get("physics_authority") != "DIAGNOSTIC_ONLY"
            or payload.get("fresh_physics_performed") is not True
            or payload.get("source_actor_frozen") is not True
            or payload.get("candidate_generated") is not False
            or payload.get("promotion_eligible") is not False
            or payload.get("paired_outcome_consistent") is not True
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("pixels_used_for_scoring") is not False
            or report_hash != hash_json(payload)
        ):
            raise ValueError("high-corner audit authority contract is invalid")
    finally:
        if report_hash is not None:
            payload["report_hash"] = report_hash
    return payload


def _maximum_replay_metric_delta(first: dict[str, Any], replay: dict[str, Any]) -> float:
    if set(first) != set(replay):
        raise ValueError("high-corner paired replay metrics are misaligned")
    deltas = [
        abs(float(first[name]) - float(replay[name])) for name in first if name != "finite_state"
    ]
    if any(not math.isfinite(value) for value in deltas):
        raise ValueError("high-corner paired replay delta is non-finite")
    return max(deltas, default=0.0)


def _diagnostic_outcome(
    metrics: dict[str, Any],
    *,
    maximum_safe_failed_rate: float,
    maximum_safe_root_angular_speed_rad_s: float,
) -> tuple[bool, bool, bool]:
    finite = bool(metrics.get("finite_state", False))
    save = float(metrics["first_save_rate"]) > 0.0
    safe = bool(
        finite
        and float(metrics["failed_rate"]) <= maximum_safe_failed_rate
        and float(metrics["maximum_root_angular_speed_rad_s"])
        <= maximum_safe_root_angular_speed_rad_s
    )
    return save, safe, finite


def _run_route(
    *,
    torch: Any,
    model: Any,
    learned: bool,
    active: GoalkeeperPhysicsPPOConfig,
    audit: HighCornerReachabilityAuditConfig,
    asset_root: Path,
    locomotion_policy_path: Path,
    device: Any,
) -> dict[str, Any]:
    world = goalkeeper_world_config(
        difficulty_profile=active.shot_difficulty_profile,  # type: ignore[arg-type]
        environment_count=audit.environment_count,
        second_shot_probability=0.0,
        shot_intent_cue_enabled=active.shot_intent_cue_enabled,
        hard_shot_fraction=1.0,
        hard_shot_height_mode="high",
        hard_shot_side_mode="balanced",
        hard_shot_flight_time_range_sec=audit.flight_time_range_sec,
        reach_reward_semantics=active.training_reach_reward_semantics,
        hard_height_reach_reward_scale=active.training_hard_height_reach_reward_scale,
        hard_height_reach_threshold_m=active.training_hard_height_reach_threshold_m,
        hard_height_reach_distance_decay=active.training_hard_height_reach_distance_decay,
        task_motion_reward_scale=active.training_task_motion_reward_scale,
        recovery_event_bonus=active.training_recovery_event_bonus,
        unsafe_penalty=active.training_unsafe_penalty,
        save_then_unsafe_penalty=active.training_save_then_unsafe_penalty,
    )
    world = _with_first_shot_release_override(world, active.training_first_shot_release_sec)
    world = _with_episode_duration_override(world, active.training_episode_duration_sec)
    world = _with_save_event_bonus_override(
        world, active.training_true_save_bonus, active.training_hand_save_bonus
    )
    world = _with_root_angular_penalty_override(
        world,
        active.training_root_angular_speed_penalty_scale,
        active.training_root_angular_speed_soft_limit_rad_s,
        active.training_root_angular_speed_excess_penalty_scale,
        active.training_flight_root_angular_penalty_scale,
    )
    world = _with_recovery_progress_override(
        world,
        active.training_recovery_progress_reward_scale,
        active.training_recovery_progress_linear_speed_decay,
        active.training_recovery_progress_angular_speed_decay,
    )
    environment = _build_environment(
        active=active,
        asset_root=asset_root,
        locomotion_policy_path=locomotion_policy_path,
        device=device,
        world_config=world,
    )
    summaries: list[dict[str, Any]] = []
    for seed in audit.seeds:
        observation = environment.reset(seed=seed)
        for _ in range(world.episode_steps):
            with torch.no_grad():
                if learned:
                    mean, _, _ = model(observation)
                    action = torch.tanh(mean)
                else:
                    action = torch.zeros(
                        (audit.environment_count, environment.action_size), device=device
                    )
            observation, _, _, _ = environment.step(action)
        if not environment.finite_state():
            raise FloatingPointError("high-corner reachability route became non-finite")
        summaries.append(environment.summary())
    mean_keys = (
        "first_save_rate",
        "first_hand_save_rate",
        "failed_rate",
        "mean_maximum_lateral_displacement_m",
        "mean_maximum_lateral_speed_mps",
        "mean_minimum_hand_target_distance_m",
        "mean_maximum_hand_displacement_m",
        "mean_maximum_hand_speed_mps",
    )
    result = {
        key: sum(float(item[key]) for item in summaries) / len(summaries) for key in mean_keys
    }
    result["maximum_root_angular_speed_rad_s"] = max(
        float(item["maximum_root_angular_speed_rad_s"]) for item in summaries
    )
    result["finite_state"] = True
    return result


__all__ = [
    "HighCornerReachabilityAuditConfig",
    "classify_high_corner_reachability",
    "run_high_corner_reachability_audit",
    "validate_high_corner_reachability_audit",
]
