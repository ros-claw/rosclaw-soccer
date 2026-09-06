"""Failure-driven, role-local playmaker pass adaptation in CPU MuJoCo.

Only the passer's bounded whole-body adapter is plastic.  The finisher neural
contact actor, its contact-to-recovery handoff, and the goalkeeper controller
remain content-hash frozen so team outcome credit cannot leak across roles.
"""

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
from rosclaw_soccer.training.neural_contact_holdout_exam import (
    validate_neural_contact_holdout_exam,
)


@dataclass(frozen=True)
class PlaymakerPassProbeAction:
    """A bounded residual around the retained S95 lead-pass policy."""

    body_yaw_correction_rad: float = 0.0
    stance_correction_x_m: float = 0.0
    stance_correction_y_m: float = 0.0
    swing_speed_scale: float = 0.80
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.playmaker_pass_probe_action.v1"

    def __post_init__(self) -> None:
        values = (
            self.body_yaw_correction_rad,
            self.stance_correction_x_m,
            self.stance_correction_y_m,
            self.swing_speed_scale,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or not -0.06 <= self.body_yaw_correction_rad <= 0.06
            or not -0.04 <= self.stance_correction_x_m <= 0.04
            or not -0.04 <= self.stance_correction_y_m <= 0.04
            or not 0.80 <= self.swing_speed_scale <= 0.95
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("playmaker pass action exceeds its SIM-only envelope")

    @property
    def action_hash(self) -> str:
        return str(hash_json(asdict(self)))


def default_playmaker_pass_actions() -> tuple[PlaymakerPassProbeAction, ...]:
    """Return a preregistered local search, ordered from least to most change."""

    raw = (
        PlaymakerPassProbeAction(),
        PlaymakerPassProbeAction(body_yaw_correction_rad=0.02),
        PlaymakerPassProbeAction(body_yaw_correction_rad=0.04),
        PlaymakerPassProbeAction(body_yaw_correction_rad=0.06),
        PlaymakerPassProbeAction(body_yaw_correction_rad=-0.02),
        PlaymakerPassProbeAction(stance_correction_x_m=-0.02),
        PlaymakerPassProbeAction(stance_correction_x_m=0.02),
        PlaymakerPassProbeAction(stance_correction_y_m=-0.02),
        PlaymakerPassProbeAction(stance_correction_y_m=0.02),
        PlaymakerPassProbeAction(body_yaw_correction_rad=0.04, stance_correction_x_m=-0.02),
        PlaymakerPassProbeAction(body_yaw_correction_rad=0.04, stance_correction_x_m=0.02),
        PlaymakerPassProbeAction(body_yaw_correction_rad=0.04, stance_correction_y_m=-0.02),
        PlaymakerPassProbeAction(body_yaw_correction_rad=0.04, stance_correction_y_m=0.02),
        PlaymakerPassProbeAction(body_yaw_correction_rad=0.04, swing_speed_scale=0.84),
        PlaymakerPassProbeAction(body_yaw_correction_rad=0.04, swing_speed_scale=0.88),
    )
    if len({action.action_hash for action in raw}) != len(raw):
        raise RuntimeError("playmaker pass search contains duplicate actions")
    return raw


def run_playmaker_pass_discovery(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    rejected_holdout_report_path: Path,
    teacher_discovery_report_path: Path,
    actor_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    quality_config: CausalTransitionGrowthConfig | None = None,
    maximum_delivery_error_m: float = 0.45,
    workers: int = 4,
) -> dict[str, Any]:
    """Search only the playmaker action on its two attributed failures."""

    if not 1 <= workers <= 4:
        raise ValueError("playmaker pass discovery workers must be in [1, 4]")
    if not 0.10 <= maximum_delivery_error_m <= 0.45:
        raise ValueError("playmaker delivery threshold must be in [0.10, 0.45] m")
    rejected_path = rejected_holdout_report_path.expanduser().resolve()
    rejected = validate_neural_contact_holdout_exam(rejected_path)
    teacher_path = teacher_discovery_report_path.expanduser().resolve()
    teacher = _bound_json(teacher_path)
    actor_source = actor_path.expanduser().resolve()
    handoff_source = handoff_actor_path.expanduser().resolve()
    actor = load_g1_neural_contact_actor(actor_source)
    handoff = load_contact_handoff_actor(handoff_source)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if (
        rejected.get("status") != "REJECTED_NEURAL_CONTACT_LOCAL_HOLDOUT"
        or rejected.get("actor_hash") != actor.actor_hash
        or teacher.get("status") != "REJECTED_ADAPTIVE_TARGET_TEACHER_DISCOVERY"
        or actor.body_hash != qualification.body_hash
        or handoff.body_hash != qualification.body_hash
    ):
        raise ValueError("playmaker pass discovery lineage changed")
    source_rows = [row for row in teacher["rows"] if row["quality"]["chain_passed"]]
    if len(source_rows) != 1:
        raise ValueError("playmaker pass discovery needs one frozen finisher action")
    payload = dict(source_rows[0]["action"])
    payload["target_foot_velocity_xyz_mps"] = tuple(payload["target_foot_velocity_xyz_mps"])
    finisher_action = TargetContactPlanAction(**payload)
    handoff_decision = handoff.decide(contact_policy_frame=finisher_action.contact_policy_frame)
    if not handoff_decision.accepted or handoff_decision.handoff_policy_frame is None:
        raise ValueError("playmaker pass discovery finisher handoff is invalid")

    rejected_request_path = rejected_path.parent / "request.json"
    if hash_bytes(rejected_request_path.read_bytes()) != rejected["request_hash"]:
        raise ValueError("playmaker pass rejected request changed")
    rejected_request = _read_object(rejected_request_path)
    contexts_by_hash = {
        str(hash_json(value)): _context_from_dict(value) for value in rejected_request["contexts"]
    }
    failed_rows = [
        row
        for row in rejected["rows"]
        if _delivery_error(row["primary"]["result"]) > maximum_delivery_error_m
    ]
    if len(failed_rows) != 2:
        raise ValueError("playmaker curriculum expects exactly two attributed pass failures")
    contexts = tuple(contexts_by_hash[row["context_hash"]] for row in failed_rows)
    actions = default_playmaker_pass_actions()
    quality = quality_config or CausalTransitionGrowthConfig()
    output = _new_external_output(output_dir)
    _, lead_source = _load_lead_policy(source_s95_dir)
    frozen_goalkeeper_policy_hash = hash_json(
        {
            "controller": "shared-world-goalkeeper-v23",
            "shared_world_implementation_hash": hash_bytes(
                (Path(__file__).parents[1] / "skills" / "team" / "shared_world.py").read_bytes()
            ),
        }
    )
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.playmaker_pass_discovery_request.v1",
        "partition": "CONSUMED_S131_PLAYMAKER_FAILURES",
        "plastic_agent_id": "red.playmaker",
        "plastic_skill": "lead_pass",
        "contexts": [asdict(context) for context in contexts],
        "context_hashes": [context.context_hash for context in contexts],
        "actions": [asdict(action) for action in actions],
        "action_hashes": [action.action_hash for action in actions],
        "maximum_delivery_error_m": maximum_delivery_error_m,
        "frozen_finisher_actor_hash": actor.actor_hash,
        "frozen_finisher_actor_file_hash": hash_bytes(actor_source.read_bytes()),
        "frozen_finisher_action_hash": finisher_action.action_hash,
        "frozen_handoff_actor_hash": handoff.actor_hash,
        "frozen_handoff_actor_file_hash": hash_bytes(handoff_source.read_bytes()),
        "frozen_goalkeeper_policy_hash": frozen_goalkeeper_policy_hash,
        "source_s95_evidence_hash": lead_source["evidence_hash"],
        "rejected_holdout_report_hash": rejected["report_hash"],
        "teacher_discovery_report_hash": teacher["report_hash"],
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
            actor_source,
            output,
            index,
            context,
            action,
            finisher_action,
            handoff_decision.handoff_policy_frame,
            quality,
            maximum_delivery_error_m,
        )
        for index, (context, action) in enumerate(
            (context, action) for context in contexts for action in actions
        )
    )
    if workers == 1:
        rows = [_run_probe(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_probe, jobs))
    selected = [_select_context(rows, context.context_hash) for context in contexts]
    recovered = {row["context_hash"] for row in rows if row["playmaker_pass_success"]}
    gates = {
        "all_failed_contexts_recovered": len(recovered) == len(contexts),
        "all_selected_safe": all(row["quality"]["safe"] for row in selected),
        "all_selected_ordered": all(row["quality"]["ordered_contacts"] for row in selected),
        "all_selected_within_delivery_threshold": all(
            row["result"]["pass_delivery_error_m"] <= maximum_delivery_error_m for row in selected
        ),
        "finisher_actor_executed_all": all(row["finisher_actor_active"] for row in rows),
        "finisher_actor_sole_contact_residual": all(
            not row["teacher_active"] and not row["scripted_contact_active"] for row in rows
        ),
        "all_trajectories_bound": all(
            hash_bytes((output / row["trajectory"]["file"]).read_bytes())
            == row["trajectory"]["file_hash"]
            for row in rows
        ),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.playmaker_pass_discovery.v1",
        "status": (
            "PASS_PLAYMAKER_PASS_DISCOVERY"
            if all(gates.values())
            else "REJECTED_PLAYMAKER_PASS_DISCOVERY"
        ),
        "promotion_eligible": False,
        "claim": "ROLE_LOCAL_PLAYMAKER_REPAIR_WITH_FINISHER_AND_GOALKEEPER_FROZEN",
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "plastic_agent_id": "red.playmaker",
        "frozen_finisher_actor_hash": actor.actor_hash,
        "frozen_goalkeeper_policy_hash": frozen_goalkeeper_policy_hash,
        "metrics": {
            "context_count": len(contexts),
            "probe_count": len(rows),
            "safe_probe_count": sum(int(row["quality"]["safe"]) for row in rows),
            "recovered_context_count": len(recovered),
            "baseline_mean_delivery_error_m": float(
                np.mean([_delivery_error(row["primary"]["result"]) for row in failed_rows])
            ),
            "selected_mean_delivery_error_m": float(
                np.mean([row["result"]["pass_delivery_error_m"] for row in selected])
            ),
        },
        "selected": selected,
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "physics_authority": "CPU_MUJOCO",
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "playmaker-pass-discovery.json", report)
    return report


def validate_playmaker_pass_discovery(path: Path) -> dict[str, Any]:
    """Validate report arithmetic, implementation, request, and raw trajectories."""

    source = path.expanduser().resolve()
    report = _bound_report(source)
    request = _read_object(source.parent / "request.json")
    rows = report.get("rows")
    selected = report.get("selected")
    if not isinstance(rows, list) or not isinstance(selected, list) or not rows or not selected:
        raise ValueError("playmaker pass evidence rows are invalid")
    threshold = request.get("maximum_delivery_error_m")
    if not isinstance(threshold, float) or not math.isfinite(threshold):
        raise ValueError("playmaker pass threshold is invalid")
    context_hashes = request.get("context_hashes")
    if not isinstance(context_hashes, list) or not all(
        isinstance(value, str) for value in context_hashes
    ):
        raise ValueError("playmaker pass context commitments are invalid")
    recovered = {
        row.get("context_hash")
        for row in rows
        if _row_passes(row, maximum_delivery_error_m=threshold)
    }
    derived_gates = {
        "all_failed_contexts_recovered": recovered == set(context_hashes),
        "all_selected_safe": all(row["quality"]["safe"] for row in selected),
        "all_selected_ordered": all(row["quality"]["ordered_contacts"] for row in selected),
        "all_selected_within_delivery_threshold": all(
            row["result"]["pass_delivery_error_m"] <= threshold for row in selected
        ),
        "finisher_actor_executed_all": all(row["finisher_actor_active"] for row in rows),
        "finisher_actor_sole_contact_residual": all(
            not row["teacher_active"] and not row["scripted_contact_active"] for row in rows
        ),
        "all_trajectories_bound": all(
            hash_bytes((source.parent / row["trajectory"]["file"]).read_bytes())
            == row["trajectory"]["file_hash"]
            for row in rows
        ),
    }
    expected_status = (
        "PASS_PLAYMAKER_PASS_DISCOVERY"
        if all(derived_gates.values())
        else "REJECTED_PLAYMAKER_PASS_DISCOVERY"
    )
    if (
        hash_bytes((source.parent / "request.json").read_bytes()) != report.get("request_hash")
        or report.get("implementation_hash") != _implementation_hash()
        or request.get("implementation_hash") != _implementation_hash()
        or report.get("gates") != derived_gates
        or report.get("status") != expected_status
        or report.get("promotion_eligible") is not False
        or report.get("plastic_agent_id") != "red.playmaker"
        or report.get("physics_authority") != "CPU_MUJOCO"
        or report.get("activation_ceiling") != "SIM_ONLY"
        or report.get("hardware_command_sent") is not False
        or report.get("pixels_used_for_scoring") is not False
    ):
        raise ValueError("playmaker pass evidence authority is invalid")
    return report


def _run_probe(
    job: tuple[
        Path,
        Path,
        Path,
        Path,
        int,
        CausalTransitionContext,
        PlaymakerPassProbeAction,
        TargetContactPlanAction,
        int,
        CausalTransitionGrowthConfig,
        float,
    ],
) -> dict[str, Any]:
    (
        asset_root,
        source_dir,
        actor_path,
        output,
        index,
        context,
        playmaker_action,
        finisher_action,
        handoff_frame,
        quality,
        maximum_delivery_error_m,
    ) = job
    lead, _ = _load_lead_policy(source_dir)
    kwargs = _context_kwargs(
        lead_policy=lead,
        config=quality,
        context=context,
        receiver_start_sec=quality.parent_receiver_start_sec,
    )
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
    row: dict[str, Any] = {
        "probe_index": index,
        "context": asdict(context),
        "context_hash": context.context_hash,
        "action": asdict(playmaker_action),
        "action_hash": playmaker_action.action_hash,
        "executed_passer_yaw_rad": kwargs["passer_yaw_rad"],
        "result": result.to_dict(),
        "quality": quality_result,
        "finisher_actor_active": bool(np.any(active)),
        "teacher_active": bool(np.any(teacher)),
        "scripted_contact_active": bool(np.any(scripted)),
        "trajectory": artifact,
    }
    row["playmaker_pass_success"] = _row_passes(
        row, maximum_delivery_error_m=maximum_delivery_error_m
    )
    return row


def _select_context(rows: list[dict[str, Any]], context_hash: str) -> dict[str, Any]:
    candidates = [row for row in rows if row["context_hash"] == context_hash]
    if not candidates:
        raise RuntimeError("playmaker search lost a curriculum context")
    return min(
        candidates,
        key=lambda row: (
            not row["quality"]["safe"],
            not row["quality"]["ordered_contacts"],
            _delivery_error(row["result"]),
            abs(float(row["result"].get("pass_delivery_lateral_error_m") or 1.0)),
            row["action_hash"],
        ),
    )


def _row_passes(row: dict[str, Any], *, maximum_delivery_error_m: float) -> bool:
    quality = row.get("quality")
    result = row.get("result")
    return bool(
        isinstance(quality, dict)
        and isinstance(result, dict)
        and quality.get("safe") is True
        and quality.get("ordered_contacts") is True
        and _delivery_error(result) <= maximum_delivery_error_m
    )


def _delivery_error(result: dict[str, Any]) -> float:
    value = result.get("pass_delivery_error_m")
    if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return math.inf


def _bound_report(path: Path) -> dict[str, Any]:
    value = _read_object(path)
    claimed = value.pop("report_hash", None)
    if claimed != hash_json(value):
        raise ValueError("playmaker pass report integrity changed")
    value["report_hash"] = claimed
    return value


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "neural_contact_actor.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
    ):
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("playmaker pass evidence must use a new external directory")
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
    "PlaymakerPassProbeAction",
    "default_playmaker_pass_actions",
    "run_playmaker_pass_discovery",
    "validate_playmaker_pass_discovery",
]
