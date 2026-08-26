"""Build a phase-attributed team Growth ledger from frozen physics evidence.

The ledger does not re-score rendered video and does not promote a policy.  It
turns a validated regulation-goal episode into role-local capability cells so
the next Growth wave can select one plastic player without forgetting the two
frozen teammates or the quality of their skill handoffs.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, cast

from rosclaw_soccer.growth.alternating_team_growth import (
    AlternatingTeamEpisode,
    CurriculumCell,
    GrowthPartition,
    PhaseScore,
    RoleGenerationBinding,
    TeamSkillPhase,
    prioritize_team_curriculum,
)
from rosclaw_soccer.growth.role_learning import SoccerRole
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.regulation_dead_corner_save import (
    validate_regulation_dead_corner_evidence,
)


def build_regulation_team_growth_ledger(
    *,
    evidence_path: Path,
    output_path: Path,
    source_checkout: Path,
) -> dict[str, Any]:
    """Freeze an S93-derived phase ledger and an honest next-learning queue."""

    source = evidence_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if not checkout.is_dir():
        raise ValueError("team Growth source checkout is unavailable")
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("team Growth ledger must be new and outside the checkout")
    evidence = validate_regulation_dead_corner_evidence(source)
    evidence_hash = hash_bytes(source.read_bytes())
    policies = _policy_bindings(evidence)
    episodes = tuple(
        _episode_from_lane(
            lane_id=lane_id,
            seed=index,
            case=_dict(case, f"case {lane_id}"),
            policies=policies,
            evidence_hash=evidence_hash,
            request_hash=_hash_value(evidence, "request_hash"),
        )
        for index, (lane_id, case) in enumerate(
            sorted(_dict(evidence.get("cases"), "evidence cases").items()), start=9300
        )
    )
    cells = _capability_cells(episodes=episodes, evidence_hash=evidence_hash)
    priorities = prioritize_team_curriculum(cells)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.regulation_team_growth_ledger.v1",
        "status": "READY_FOR_ALTERNATING_GROWTH",
        "source_claim": evidence.get("claim"),
        "source_evidence_path": str(source),
        "source_evidence_hash": evidence_hash,
        "source_request_hash": evidence.get("request_hash"),
        "episodes": [item.to_dict() for item in episodes],
        "capability_cells": [_curriculum_cell_dict(item) for item in cells],
        "curriculum_priorities": [item.to_dict() for item in priorities],
        "recommended_role_order": [
            {
                "order": 1,
                "plastic_role": SoccerRole.PASSER.value,
                "task": "dynamic_lead_pass",
                "frozen_roles": [SoccerRole.SHOOTER.value, SoccerRole.GOALKEEPER.value],
                "reason": "S93 proves a fixed receiver pass, not a moving lead pass",
            },
            {
                "order": 2,
                "plastic_role": SoccerRole.SHOOTER.value,
                "task": "upper_dead_corner_strike",
                "frozen_roles": [SoccerRole.PASSER.value, SoccerRole.GOALKEEPER.value],
                "reason": "the frozen crossing height is about 1.249 m, not a top corner",
            },
            {
                "order": 3,
                "plastic_role": SoccerRole.GOALKEEPER.value,
                "task": "center_origin_upper_corner_save",
                "frozen_roles": [SoccerRole.PASSER.value, SoccerRole.SHOOTER.value],
                "reason": (
                    "S93 uses causal angle-aligned positioning, not full-goal center coverage"
                ),
            },
            {
                "order": 4,
                "plastic_role": SoccerRole.GOALKEEPER.value,
                "task": "save_recover_second_threat",
                "frozen_roles": [SoccerRole.PASSER.value, SoccerRole.SHOOTER.value],
                "reason": "recovery-ready is measured once; a second physical save is untested",
            },
        ],
        "selection_semantics": {
            "plastic_roles_per_round": 1,
            "frozen_roles_per_round": 2,
            "phase_credit_not_match_outcome": True,
            "successor_state_is_promotion_objective": True,
            "discovery_and_holdout_must_be_disjoint": True,
            "video_pixels_used_for_scoring": False,
        },
        "evidence_ceiling": {
            "physics_authority": "CPU_MUJOCO",
            "activation_ceiling": "SIM_ONLY",
            "commercial_use_allowed": False,
            "statistical_success_rate_claimed": False,
            "fresh_training_performed": False,
            "statement": (
                "This ledger routes the next learning wave from validated frozen physics; "
                "it does not claim that the four untested frontier tasks already pass."
            ),
        },
        "hardware_command_sent": False,
        "source_checkout": str(checkout),
    }
    report["ledger_hash"] = hash_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, report)
    return report


def validate_regulation_team_growth_ledger(path: Path) -> dict[str, Any]:
    """Validate content binding and the one-plastic-player scheduling contract."""

    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("team Growth ledger must be a JSON object")
    ledger_hash = payload.pop("ledger_hash", None)
    try:
        if (
            payload.get("schema_version") != "rosclaw_soccer.regulation_team_growth_ledger.v1"
            or payload.get("status") != "READY_FOR_ALTERNATING_GROWTH"
            or payload.get("hardware_command_sent") is not False
            or payload.get("selection_semantics", {}).get("plastic_roles_per_round") != 1
            or payload.get("selection_semantics", {}).get("frozen_roles_per_round") != 2
            or payload.get("selection_semantics", {}).get("video_pixels_used_for_scoring")
            is not False
            or payload.get("evidence_ceiling", {}).get("physics_authority") != "CPU_MUJOCO"
            or payload.get("evidence_ceiling", {}).get("activation_ceiling") != "SIM_ONLY"
            or payload.get("evidence_ceiling", {}).get("fresh_training_performed") is not False
            or hash_json(payload) != ledger_hash
        ):
            raise ValueError("team Growth ledger authority contract is invalid")
        evidence_path = payload.get("source_evidence_path")
        evidence_hash = payload.get("source_evidence_hash")
        if not isinstance(evidence_path, str) or not isinstance(evidence_hash, str):
            raise ValueError("team Growth ledger source binding is invalid")
        evidence = Path(evidence_path)
        if not evidence.is_file() or hash_bytes(evidence.read_bytes()) != evidence_hash:
            raise ValueError("team Growth ledger source evidence changed")
        orders = payload.get("recommended_role_order")
        if not isinstance(orders, list) or not orders:
            raise ValueError("team Growth ledger role queue is empty")
        for item in orders:
            if not isinstance(item, dict) or len(item.get("frozen_roles", [])) != 2:
                raise ValueError("team Growth ledger role queue is not alternating")
            if item.get("plastic_role") in item.get("frozen_roles", []):
                raise ValueError("team Growth ledger freezes its plastic role")
    finally:
        if ledger_hash is not None:
            payload["ledger_hash"] = ledger_hash
    return cast(dict[str, Any], payload)


def _policy_bindings(evidence: dict[str, Any]) -> tuple[RoleGenerationBinding, ...]:
    artifacts = _dict(evidence.get("artifacts"), "evidence artifacts")
    implementation_hash = _hash_value(evidence, "implementation_hash")
    policy_hashes = {
        SoccerRole.PASSER: hash_json(
            {"role": "passer", "implementation_hash": implementation_hash, "stage": "s93"}
        ),
        SoccerRole.SHOOTER: _hash_value(artifacts, "striker_actor_hash"),
        SoccerRole.GOALKEEPER: hash_json(
            {
                "goalkeeper_actor_hash": _hash_value(artifacts, "goalkeeper_actor_hash"),
                "gmt_model_hash": _hash_value(artifacts, "gmt_model_hash"),
                "gmt_skill_hash": _hash_value(artifacts, "gmt_skill_hash"),
                "dive_source_commit": artifacts.get("dive_source_commit"),
            }
        ),
    }
    return tuple(
        RoleGenerationBinding(
            role=role,
            agent_id={
                SoccerRole.PASSER: "soccer.playmaker",
                SoccerRole.SHOOTER: "soccer.finisher",
                SoccerRole.GOALKEEPER: "soccer.goalkeeper",
            }[role],
            artifact_hash=policy_hashes[role],
            parent_artifact_hash=hash_json(
                {"role": role.value, "predecessor": "s92", "child": policy_hashes[role]}
            ),
            generation=93,
        )
        for role in SoccerRole
    )


def _episode_from_lane(
    *,
    lane_id: str,
    seed: int,
    case: dict[str, Any],
    policies: tuple[RoleGenerationBinding, ...],
    evidence_hash: str,
    request_hash: str,
) -> AlternatingTeamEpisode:
    baseline = _dict(case.get("baseline_replay"), f"{lane_id} baseline replay")
    save = _dict(case.get("save_replay"), f"{lane_id} save replay")
    baseline_result = _dict(baseline.get("result"), f"{lane_id} baseline result")
    save_result = _dict(save.get("result"), f"{lane_id} save result")
    takeoff = _dict(save.get("takeoff_exam"), f"{lane_id} takeoff exam")
    takeoff_metrics = _dict(takeoff.get("metrics"), f"{lane_id} takeoff metrics")
    takeoff_base = _dict(takeoff.get("base"), f"{lane_id} takeoff base")
    recovery = _dict(takeoff_base.get("dynamic_metrics"), f"{lane_id} recovery metrics")
    policy = {item.role: item.artifact_hash for item in policies}
    pass_time = _finite(baseline_result, "pass_contact_time_sec")
    shot_time = _finite(baseline_result, "shot_contact_time_sec")
    glove_time = _finite(save_result, "goalkeeper_glove_contact_time_sec")
    landing_time = _finite(takeoff_metrics, "landing_time_sec")
    pass_error = _finite(baseline_result, "pass_delivery_error_m")
    lateral_error = _finite(baseline_result, "pass_delivery_lateral_error_m")
    post_clearance = _finite(baseline, "post_surface_clearance_m")
    crossing_height = _finite(baseline, "goal_crossing_z_m")
    shot_speed = _finite(baseline_result, "shot_peak_ball_speed_mps")
    surface_distance = abs(_finite(save, "glove_surface_distance_m"))
    glove_position = save.get("glove_contact_position_m")
    if not isinstance(glove_position, list | tuple) or len(glove_position) != 3:
        raise ValueError(f"{lane_id} glove position is invalid")
    glove_height = float(glove_position[2])
    if not math.isfinite(glove_height):
        raise ValueError(f"{lane_id} glove height is invalid")
    safety_cost = float(
        not (
            baseline.get("passed") is True
            and save.get("passed") is True
            and takeoff.get("passed") is True
        )
    )
    phases = (
        PhaseScore(
            TeamSkillPhase.LEAD_PASS,
            SoccerRole.PASSER,
            policy[SoccerRole.PASSER],
            evidence_hash,
            bool(baseline.get("gates", {}).get("precise_pass")),
            _clamp01(1.0 - pass_error / 0.05),
            _clamp01(1.0 - lateral_error / 0.03),
            safety_cost,
            0.0,
            pass_time,
        ),
        PhaseScore(
            TeamSkillPhase.RUNNING_INTERCEPT,
            SoccerRole.SHOOTER,
            policy[SoccerRole.SHOOTER],
            evidence_hash,
            bool(baseline_result.get("shot_contact_observed") and shot_time > pass_time),
            _clamp01(1.0 - (shot_time - pass_time) / 4.0),
            _clamp01(shot_speed / 12.0),
            safety_cost,
            pass_time,
            shot_time,
        ),
        PhaseScore(
            TeamSkillPhase.STRIKE,
            SoccerRole.SHOOTER,
            policy[SoccerRole.SHOOTER],
            evidence_hash,
            bool(
                baseline.get("gates", {}).get("raised_shot")
                and baseline.get("gates", {}).get("regulation_post_dead_corner")
                and baseline.get("gates", {}).get("unopposed_goal")
            ),
            _clamp01(
                0.5 * (1.0 - post_clearance / 0.15)
                + 0.5 * (crossing_height / (2.44 - 0.06 - 0.115))
            ),
            _clamp01(_finite(baseline_result, "shooter_min_pelvis_height_m") / 0.75),
            safety_cost,
            shot_time,
            glove_time,
        ),
        PhaseScore(
            TeamSkillPhase.GLOVE_SAVE,
            SoccerRole.GOALKEEPER,
            policy[SoccerRole.GOALKEEPER],
            evidence_hash,
            bool(
                save.get("gates", {}).get("collision_faithful_glove")
                and save.get("gates", {}).get("physical_save")
            ),
            _clamp01(
                0.5 * (1.0 - surface_distance / 0.018)
                + 0.5 * (glove_height / (2.44 - 0.06 - 0.115))
            ),
            _clamp01(_finite(save_result, "goalkeeper_min_pelvis_height_m") / 0.75),
            safety_cost,
            glove_time,
            max(glove_time, _finite(takeoff_metrics, "airborne_stop_sec")),
        ),
        PhaseScore(
            TeamSkillPhase.CONTROLLED_LANDING,
            SoccerRole.GOALKEEPER,
            policy[SoccerRole.GOALKEEPER],
            evidence_hash,
            bool(takeoff.get("gates", {}).get("bounded_landing")),
            _clamp01(
                1.0
                - 0.5 * _finite(takeoff_metrics, "landing_vertical_speed_mps") / 1.2
                - 0.5 * _finite(takeoff_metrics, "landing_angular_speed_rad_s") / 3.5
            ),
            _clamp01(_finite(recovery, "recovery_minimum_pelvis_height_m") / 0.80),
            safety_cost,
            _finite(takeoff_metrics, "airborne_stop_sec"),
            landing_time,
        ),
        PhaseScore(
            TeamSkillPhase.SUCCESSOR_READY,
            SoccerRole.GOALKEEPER,
            policy[SoccerRole.GOALKEEPER],
            evidence_hash,
            bool(takeoff_base.get("gates", {}).get("post_save_recovered")),
            _clamp01(_finite(recovery, "recovery_minimum_upright_projection")),
            _clamp01(1.0 - _finite(recovery, "recovery_maximum_linear_speed_mps") / 0.50),
            safety_cost,
            landing_time,
            11.0,
        ),
    )
    return AlternatingTeamEpisode(
        episode_id=f"s93.{lane_id}",
        seed=seed,
        partition=GrowthPartition.DISCOVERY,
        scenario_hash=hash_json(
            {"request_hash": request_hash, "lane_id": lane_id, "source": "s93"}
        ),
        environment_hash=hash_json({"physics": "CPU_MUJOCO", "request_hash": request_hash}),
        trajectory_hash=hash_json(
            {
                "baseline": case.get("baseline_trajectory_hash"),
                "save": case.get("save_trajectory_hash"),
            }
        ),
        policies=policies,
        phases=phases,
        chain_success=bool(case.get("passed")),
    )


def _capability_cells(
    *, episodes: tuple[AlternatingTeamEpisode, ...], evidence_hash: str
) -> tuple[CurriculumCell, ...]:
    count = len(episodes)
    return (
        CurriculumCell(
            "fixed_precise_pass",
            SoccerRole.PASSER,
            0.35,
            count,
            sum(item.phase(TeamSkillPhase.LEAD_PASS).success for item in episodes),
            evidence_hash,
            "historical",
        ),
        CurriculumCell(
            "dynamic_lead_pass",
            SoccerRole.PASSER,
            0.65,
            0,
            0,
            None,
        ),
        CurriculumCell(
            "raised_lateral_dead_corner_strike",
            SoccerRole.SHOOTER,
            0.65,
            count,
            sum(item.phase(TeamSkillPhase.STRIKE).success for item in episodes),
            evidence_hash,
            "historical",
        ),
        CurriculumCell(
            "upper_dead_corner_strike",
            SoccerRole.SHOOTER,
            0.80,
            0,
            0,
            None,
        ),
        CurriculumCell(
            "angle_aligned_dead_corner_save",
            SoccerRole.GOALKEEPER,
            0.75,
            count,
            sum(item.phase(TeamSkillPhase.GLOVE_SAVE).success for item in episodes),
            evidence_hash,
            "historical",
        ),
        CurriculumCell(
            "center_origin_upper_corner_save",
            SoccerRole.GOALKEEPER,
            0.90,
            0,
            0,
            None,
        ),
        CurriculumCell(
            "post_save_successor_ready",
            SoccerRole.GOALKEEPER,
            0.75,
            count,
            sum(item.phase(TeamSkillPhase.SUCCESSOR_READY).success for item in episodes),
            evidence_hash,
            "historical",
        ),
        CurriculumCell(
            "save_recover_second_threat",
            SoccerRole.GOALKEEPER,
            1.0,
            0,
            0,
            None,
            "nightmare",
        ),
    )


def _curriculum_cell_dict(cell: CurriculumCell) -> dict[str, Any]:
    return {
        "schema_version": cell.schema_version,
        "cell_id": cell.cell_id,
        "role": cell.role.value,
        "difficulty": cell.difficulty,
        "attempts": cell.attempts,
        "successes": cell.successes,
        "success_rate": cell.success_rate,
        "evidence_hash": cell.evidence_hash,
        "source": cell.source,
        "route": cell.route,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _dict(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _hash_value(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.startswith("sha256:") or len(item) != 71:
        raise ValueError(f"{key} must be a sha256 content hash")
    return item


def _finite(value: dict[str, Any], key: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(item):
        raise ValueError(f"{key} must be finite")
    return float(item)


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("normalized team metric must be finite")
    return min(1.0, max(0.0, value))


__all__ = [
    "build_regulation_team_growth_ledger",
    "validate_regulation_team_growth_ledger",
]
