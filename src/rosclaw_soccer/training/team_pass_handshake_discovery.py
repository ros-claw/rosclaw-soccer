"""Discover a pass-arrival/receiver-readiness handshake from failed team play."""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.causal_strike_option import G1CausalStrikeOptionConfig
from rosclaw_soccer.growth.contact_handoff_actor import load_contact_handoff_actor
from rosclaw_soccer.growth.neural_contact_actor import load_g1_neural_contact_actor
from rosclaw_soccer.growth.target_contact_plan_actor import TargetContactPlanAction
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world
from rosclaw_soccer.training.causal_transition_growth import (
    CausalTransitionContext,
    CausalTransitionGrowthConfig,
    _context_kwargs,
    _load_lead_policy,
    _save_trajectory,
)
from rosclaw_soccer.training.contact_handoff_discovery import _context_from_dict
from rosclaw_soccer.training.intended_foot_alignment_discovery import (
    strict_intended_contact_quality,
)
from rosclaw_soccer.training.neural_contact_canary import _bound_json
from rosclaw_soccer.training.playmaker_pass_discovery import PlaymakerPassProbeAction
from rosclaw_soccer.training.playmaker_pass_discovery import (
    _implementation_hash as _discovery_implementation_hash,
)
from rosclaw_soccer.training.playmaker_pass_repair_discovery import (
    _implementation_hash as _repair_implementation_hash,
)


@dataclass(frozen=True)
class PassReceiveTimingProbe:
    """A team-level timing residual over frozen low-level role policies."""

    playmaker_action: PlaymakerPassProbeAction
    receiver_start_correction_sec: float
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.pass_receive_timing_probe.v1"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.receiver_start_correction_sec)
            or not -0.08 <= self.receiver_start_correction_sec <= 0.08
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("pass-receive timing probe exceeds its SIM-only envelope")

    @property
    def probe_hash(self) -> str:
        return str(hash_json(asdict(self)))


def receiver_timing_corrections_sec() -> tuple[float, ...]:
    return (-0.08, -0.04, 0.0, 0.04, 0.08)


def run_team_pass_handshake_discovery(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    source_playmaker_report_path: Path,
    finisher_actor_path: Path,
    teacher_discovery_report_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    """Search receiver readiness timing while all low-level actors are frozen."""

    if not 1 <= workers <= 4:
        raise ValueError("team handshake workers must be in [1, 4]")
    source_path = source_playmaker_report_path.expanduser().resolve()
    source_report = _bound_json(source_path)
    source_request = _read_object(source_path.parent / "request.json")
    finisher_source = finisher_actor_path.expanduser().resolve()
    teacher_path = teacher_discovery_report_path.expanduser().resolve()
    handoff_source = handoff_actor_path.expanduser().resolve()
    finisher = load_g1_neural_contact_actor(finisher_source)
    teacher = _bound_json(teacher_path)
    handoff = load_contact_handoff_actor(handoff_source)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if (
        not _supported_playmaker_source(source_report)
        or source_report.get("promotion_eligible") is not False
        or source_report.get("implementation_hash") != _source_implementation_hash(source_report)
        or hash_bytes((source_path.parent / "request.json").read_bytes())
        != source_report["request_hash"]
        or source_request.get("frozen_finisher_actor_hash") != finisher.actor_hash
        or finisher.body_hash != qualification.body_hash
        or handoff.body_hash != qualification.body_hash
        or teacher.get("status") != "REJECTED_ADAPTIVE_TARGET_TEACHER_DISCOVERY"
    ):
        raise ValueError("team handshake discovery lineage changed")
    for row in source_report["rows"]:
        artifact = row["trajectory"]
        if (
            hash_bytes((source_path.parent / artifact["file"]).read_bytes())
            != artifact["file_hash"]
        ):
            raise ValueError("team handshake source trajectory changed")
    source_rows = [row for row in teacher["rows"] if row["quality"]["chain_passed"]]
    if len(source_rows) != 1:
        raise ValueError("team handshake needs one frozen finisher action")
    payload = dict(source_rows[0]["action"])
    payload["target_foot_velocity_xyz_mps"] = tuple(payload["target_foot_velocity_xyz_mps"])
    finisher_action = TargetContactPlanAction(**payload)
    handoff_decision = handoff.decide(contact_policy_frame=finisher_action.contact_policy_frame)
    if not handoff_decision.accepted or handoff_decision.handoff_policy_frame is None:
        raise ValueError("team handshake finisher handoff is invalid")
    selected = cast(list[dict[str, Any]], source_report["selected"])
    contexts = tuple(_context_from_dict(row["context"]) for row in selected)
    playmaker_actions = tuple(PlaymakerPassProbeAction(**row["action"]) for row in selected)
    probes = tuple(
        (context, PassReceiveTimingProbe(action, correction))
        for context, action in zip(contexts, playmaker_actions, strict=True)
        for correction in receiver_timing_corrections_sec()
    )
    quality = quality_config or CausalTransitionGrowthConfig()
    output = _new_external_output(output_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.team_pass_handshake_request.v1",
        "partition": "CONSUMED_PLAYMAKER_DISCOVERY_FOR_TEAM_HANDSHAKE",
        "plastic_contract": "PASS_ARRIVAL_RECEIVER_READINESS_HANDSHAKE",
        "contexts": [asdict(context) for context in contexts],
        "context_hashes": [context.context_hash for context in contexts],
        "probes": [asdict(probe) for _, probe in probes],
        "probe_hashes": [probe.probe_hash for _, probe in probes],
        "source_playmaker_report_hash": source_report["report_hash"],
        "source_playmaker_status": source_report["status"],
        "frozen_playmaker_actions": [asdict(action) for action in playmaker_actions],
        "frozen_finisher_actor_hash": finisher.actor_hash,
        "frozen_finisher_actor_file_hash": hash_bytes(finisher_source.read_bytes()),
        "frozen_goalkeeper_policy_hash": source_request["frozen_goalkeeper_policy_hash"],
        "frozen_handoff_actor_hash": handoff.actor_hash,
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "body_hash": qualification.body_hash,
        "implementation_hash": _implementation_hash(),
        "physics_authority": "CPU_MUJOCO",
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = tuple(
        (
            asset_root.expanduser().resolve(),
            source_s95_dir.expanduser().resolve(),
            finisher_source,
            output,
            index,
            context,
            probe,
            finisher_action,
            handoff_decision.handoff_policy_frame,
            quality,
        )
        for index, (context, probe) in enumerate(probes)
    )
    if workers == 1:
        rows = [_run_timing_probe(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_timing_probe, jobs))
    chosen = [_select(rows, context.context_hash) for context in contexts]
    recovered = {row["context_hash"] for row in rows if row["strict_team_chain"]}
    gates = {
        "all_contexts_have_strict_handshake": recovered
        == {context.context_hash for context in contexts},
        "all_selected_safe": all(row["quality"]["safe"] for row in chosen),
        "all_selected_strict_team_chain": all(row["strict_team_chain"] for row in chosen),
        "all_selected_clear_outcome": all(row["quality"]["clear_outcome"] for row in chosen),
        "finisher_actor_executed_all": all(row["finisher_actor_active"] for row in rows),
        "no_teacher_or_scripted_contact": all(
            not row["teacher_active"] and not row["scripted_contact_active"] for row in rows
        ),
        "all_trajectories_bound": all(
            hash_bytes((output / row["trajectory"]["file"]).read_bytes())
            == row["trajectory"]["file_hash"]
            for row in rows
        ),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.team_pass_handshake_discovery.v1",
        "status": (
            "PASS_TEAM_PASS_HANDSHAKE_DISCOVERY"
            if all(gates.values())
            else "REJECTED_TEAM_PASS_HANDSHAKE_DISCOVERY"
        ),
        "promotion_eligible": False,
        "claim": "TEAM_TIMING_COORDINATION_OVER_FROZEN_LOW_LEVEL_ROLE_POLICIES",
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "metrics": {
            "context_count": len(contexts),
            "probe_count": len(rows),
            "safe_probe_count": sum(int(row["quality"]["safe"]) for row in rows),
            "strict_team_chain_count": sum(int(row["strict_team_chain"]) for row in rows),
            "recovered_context_count": len(recovered),
            "goal_count": sum(int(row["result"]["goal_crossed"]) for row in rows),
            "save_count": sum(int(row["result"]["goalkeeper_save_observed"]) for row in rows),
        },
        "selected": chosen,
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "physics_authority": "CPU_MUJOCO",
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "handshake-report.json", report)
    return report


def _run_timing_probe(
    job: tuple[
        Path,
        Path,
        Path,
        Path,
        int,
        CausalTransitionContext,
        PassReceiveTimingProbe,
        TargetContactPlanAction,
        int,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    (
        asset_root,
        source_dir,
        actor_path,
        output,
        index,
        context,
        probe,
        finisher_action,
        handoff_frame,
        quality,
    ) = job
    lead, _ = _load_lead_policy(source_dir)
    receiver_start = quality.parent_receiver_start_sec + probe.receiver_start_correction_sec
    kwargs = _context_kwargs(
        lead_policy=lead,
        config=quality,
        context=context,
        receiver_start_sec=receiver_start,
    )
    playmaker_action = probe.playmaker_action
    base = cast(dict[str, float], kwargs["passer_parameter_overrides"])
    kwargs["passer_parameter_overrides"] = {
        **base,
        "stance_offset_x": base["stance_offset_x"] + playmaker_action.stance_correction_x_m,
        "stance_offset_y": base["stance_offset_y"] + playmaker_action.stance_correction_y_m,
        "swing_speed_scale": playmaker_action.swing_speed_scale,
    }
    base_yaw = float(kwargs["passer_yaw_rad"])
    kwargs["passer_yaw_rad"] = math.atan2(
        math.sin(base_yaw + playmaker_action.body_yaw_correction_rad),
        math.cos(base_yaw + playmaker_action.body_yaw_correction_rad),
    )
    kwargs.update(
        shooter_parameter_overrides={
            "stance_offset_x": finisher_action.stance_offset_x_m,
            "stance_offset_y": finisher_action.stance_offset_y_m,
            "foot_yaw_offset": finisher_action.foot_yaw_offset_rad,
            "foot_pitch_offset": finisher_action.foot_pitch_offset_rad,
        },
        shooter_causal_strike_option_config=replace(
            G1CausalStrikeOptionConfig(),
            maximum_arrival_advance_frames=finisher_action.maximum_arrival_advance_frames,
        ),
        shooter_neural_contact_actor_path=actor_path,
        shooter_neural_contact_policy_frame=finisher_action.contact_policy_frame,
        shooter_neural_contact_target_velocity_xyz_mps=(
            finisher_action.target_foot_velocity_xyz_mps
        ),
        shooter_precontact_joint_guard_enabled=True,
        shooter_post_policy_frame=handoff_frame,
    )
    result, trajectory = simulate_shared_world(asset_root, **kwargs)
    artifact = _save_trajectory(output / f"probe-{index:03d}.npz", trajectory)
    quality_result = strict_intended_contact_quality(
        result=result,
        trajectory=trajectory,
        quality_config=quality,
        intended_contact_foot=1,
    )
    active = np.asarray(trajectory["shooter_neural_contact_actor_active"], dtype=np.bool_)
    teacher = np.asarray(trajectory["shooter_loft_teacher_active"], dtype=np.bool_)
    scripted = np.asarray(trajectory["shooter_ballistic_contact_torque_active"], dtype=np.bool_)
    pass_time = result.pass_contact_time_sec
    shot_time = result.shot_contact_time_sec
    return {
        "probe_index": index,
        "context": asdict(context),
        "context_hash": context.context_hash,
        "probe": asdict(probe),
        "probe_hash": probe.probe_hash,
        "receiver_start_sec": receiver_start,
        "observed_pass_to_receive_sec": (
            float(shot_time - pass_time)
            if pass_time is not None and shot_time is not None
            else None
        ),
        "result": result.to_dict(),
        "quality": quality_result,
        "strict_team_chain": bool(quality_result["strict_chain_passed"]),
        "finisher_actor_active": bool(np.any(active)),
        "teacher_active": bool(np.any(teacher)),
        "scripted_contact_active": bool(np.any(scripted)),
        "trajectory": artifact,
    }


def _select(rows: list[dict[str, Any]], context_hash: str) -> dict[str, Any]:
    candidates = [row for row in rows if row["context_hash"] == context_hash]
    return min(
        candidates,
        key=lambda row: (
            not row["strict_team_chain"],
            not row["quality"]["safe"],
            not row["quality"]["ordered_contacts"],
            _error(row),
            abs(float(row["probe"]["receiver_start_correction_sec"])),
        ),
    )


def _error(row: dict[str, Any]) -> float:
    value = row["result"].get("pass_delivery_error_m")
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 5.0


def _supported_playmaker_source(report: dict[str, Any]) -> bool:
    return bool(
        (
            report.get("schema_version") == "rosclaw_soccer.playmaker_pass_discovery.v1"
            and report.get("status") == "PASS_PLAYMAKER_PASS_DISCOVERY"
        )
        or (
            report.get("schema_version") == "rosclaw_soccer.playmaker_pass_repair_discovery.v1"
            and report.get("status") == "REJECTED_PLAYMAKER_PASS_REPAIR_DATA"
        )
    )


def _source_implementation_hash(report: dict[str, Any]) -> str:
    if report.get("schema_version") == "rosclaw_soccer.playmaker_pass_discovery.v1":
        return _discovery_implementation_hash()
    return _repair_implementation_hash()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "role_self_model.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
    ):
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("team handshake output must use a new external directory")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = [
    "PassReceiveTimingProbe",
    "receiver_timing_corrections_sec",
    "run_team_pass_handshake_discovery",
]
