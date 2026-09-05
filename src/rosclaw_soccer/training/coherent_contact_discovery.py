"""Failure-driven discovery of coherent pre-contact body modes.

This stage deliberately reuses a rejected sealed partition only after its
verdict is immutable.  It evaluates counterfactual contact modes with the
stance fixed from rollout start to contact: no late runtime decision may
rewrite whole-body geometry.  The output is development evidence and can
never authorize promotion on its own.
"""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.causal_strike_option import G1CausalStrikeOptionConfig
from rosclaw_soccer.growth.three_axis_contact_actor import (
    load_g1_three_axis_contact_actor,
)
from rosclaw_soccer.growth.upper_corner_strike import UpperCornerStrikePolicy
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world
from rosclaw_soccer.training.causal_transition_growth import (
    CausalTransitionContext,
    CausalTransitionGrowthConfig,
    _chain_quality,
    _context_kwargs,
    _load_lead_policy,
    _save_trajectory,
)


@dataclass(frozen=True)
class CoherentContactProbe:
    """One immutable, pre-contact whole-body/contact configuration."""

    context: CausalTransitionContext
    maximum_arrival_advance_frames: int
    stance_offset_x_m: float
    stance_offset_y_m: float
    contact_policy_frame: int = 248
    foot_yaw_offset_rad: float = 0.04
    foot_pitch_offset_rad: float = 0.01

    def __post_init__(self) -> None:
        values = (
            self.stance_offset_x_m,
            self.stance_offset_y_m,
            self.foot_yaw_offset_rad,
            self.foot_pitch_offset_rad,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or self.maximum_arrival_advance_frames not in {0, 6, 12, 18}
            or not -0.12 <= self.stance_offset_x_m <= 0.12
            or not -0.12 <= self.stance_offset_y_m <= 0.12
            or not 238 <= self.contact_policy_frame <= 258
            or not -0.12 <= self.foot_yaw_offset_rad <= 0.12
            or not -0.08 <= self.foot_pitch_offset_rad <= 0.08
        ):
            raise ValueError("coherent contact probe exceeds its SIM-only envelope")

    @property
    def probe_hash(self) -> str:
        return str(hash_json(asdict(self)))


def run_coherent_contact_discovery(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    rejected_exam_path: Path,
    contact_actor_path: Path,
    probes: tuple[CoherentContactProbe, ...],
    output_dir: Path,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 6,
) -> dict[str, Any]:
    """Evaluate teacher-free counterfactuals from a bound rejected exam."""

    if len(probes) < 6 or len({probe.probe_hash for probe in probes}) != len(probes):
        raise ValueError("coherent contact discovery needs six unique probes")
    if not 1 <= workers <= 8:
        raise ValueError("coherent contact discovery workers must be in [1, 8]")
    rejected_path = rejected_exam_path.expanduser().resolve()
    rejected = json.loads(rejected_path.read_text(encoding="utf-8"))
    claimed_rejected_hash = rejected.pop("report_hash", None)
    if (
        claimed_rejected_hash != hash_json(rejected)
        or rejected.get("status") != "REJECTED_DUAL_CLOCK_CONTACT_RETENTION"
        or rejected.get("promotion_eligible") is not False
        or rejected.get("sealed") is not True
    ):
        raise ValueError("coherent discovery requires an intact rejected sealed exam")
    rejected["report_hash"] = claimed_rejected_hash
    for index, row in enumerate(rejected["rows"]):
        case_dir = rejected_path.parent / f"case-{index:03d}"
        for key in ("candidate_artifact", "replay_artifact", "parent_artifact"):
            artifact = row[key]
            artifact_path = case_dir / str(artifact["file"])
            if (
                not artifact_path.is_file()
                or hash_bytes(artifact_path.read_bytes()) != artifact["file_hash"]
            ):
                raise ValueError("coherent discovery failure trajectory binding changed")
    rejected_context_hashes = {str(row["context_hash"]) for row in rejected["rows"]}
    probe_context_hashes = {probe.context.context_hash for probe in probes}
    if not probe_context_hashes <= rejected_context_hashes:
        raise ValueError("coherent probes must come from the consumed failure partition")

    quality = quality_config or CausalTransitionGrowthConfig()
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    actor = load_g1_three_axis_contact_actor(contact_actor_path)
    if actor.body_hash != qualification.body_hash:
        raise ValueError("coherent discovery contact actor Body identity changed")
    _, source = _load_lead_policy(source_s95_dir)
    output = _new_external_output(output_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw.growth.coherent_contact_discovery_request.v1",
        "partition": "CONSUMED_REJECTED_HOLDOUT_DEVELOPMENT",
        "probes": [asdict(probe) for probe in probes],
        "probe_hashes": [probe.probe_hash for probe in probes],
        "rejected_exam_report_hash": claimed_rejected_hash,
        "rejected_exam_file_hash": hash_bytes(rejected_path.read_bytes()),
        "contact_actor_hash": actor.actor_hash,
        "contact_actor_file_hash": hash_bytes(contact_actor_path.read_bytes()),
        "source_s95_evidence_hash": source["evidence_hash"],
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "implementation_hash": _implementation_hash(),
        "teacher_enabled": False,
        "late_stance_rewrite_allowed": False,
        "activation_ceiling": "SIM_ONLY",
        "physics_authority": "CPU_MUJOCO",
        "pixels_used_for_scoring": False,
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = tuple(
        (
            asset_root.expanduser().resolve(),
            source_s95_dir.expanduser().resolve(),
            contact_actor_path.expanduser().resolve(),
            output,
            index,
            probe,
            quality,
        )
        for index, probe in enumerate(probes)
    )
    if workers == 1:
        rows = [_run_probe(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_probe, jobs))
    success_count = sum(bool(row["quality"]["chain_passed"]) for row in rows)
    safe_count = sum(bool(row["quality"]["safe"]) for row in rows)
    active_count = sum(bool(row["contact_actor_active"]) for row in rows)
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.coherent_contact_discovery.v1",
        "status": (
            "PASS_COHERENT_CONTACT_DISCOVERY"
            if success_count >= 2 and safe_count == len(rows) and active_count >= 2
            else "REJECTED_COHERENT_CONTACT_DISCOVERY"
        ),
        "promotion_eligible": False,
        "partition": "CONSUMED_REJECTED_HOLDOUT_DEVELOPMENT",
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "rejected_exam_report_hash": claimed_rejected_hash,
        "contact_actor_hash": actor.actor_hash,
        "metrics": {
            "probe_count": len(rows),
            "chain_success_count": success_count,
            "safe_count": safe_count,
            "contact_actor_active_count": active_count,
        },
        "rows": rows,
        "teacher_enabled": False,
        "late_stance_rewrite_allowed": False,
        "implementation_hash": _implementation_hash(),
        "activation_ceiling": "SIM_ONLY",
        "physics_authority": "CPU_MUJOCO",
        "pixels_used_for_scoring": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "discovery-report.json", report)
    return report


def _run_probe(
    job: tuple[
        Path,
        Path,
        Path,
        Path,
        int,
        CoherentContactProbe,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    asset_root, source_s95_dir, actor_path, output, index, probe, quality = job
    lead_policy, _ = _load_lead_policy(source_s95_dir)
    kwargs = _context_kwargs(
        lead_policy=lead_policy,
        config=quality,
        context=probe.context,
        receiver_start_sec=quality.parent_receiver_start_sec,
    )
    kwargs.update(
        shooter_causal_strike_option_config=replace(
            G1CausalStrikeOptionConfig(),
            maximum_arrival_advance_frames=probe.maximum_arrival_advance_frames,
        ),
        shooter_ballistic_contact_torque_config=replace(
            UpperCornerStrikePolicy().torque_config(),
            contact_policy_frame=probe.contact_policy_frame,
        ),
        shooter_three_axis_contact_actor_path=actor_path,
        shooter_precontact_joint_guard_enabled=True,
        shooter_parameter_overrides={
            "stance_offset_x": probe.stance_offset_x_m,
            "stance_offset_y": probe.stance_offset_y_m,
            "foot_yaw_offset": probe.foot_yaw_offset_rad,
            "foot_pitch_offset": probe.foot_pitch_offset_rad,
        },
    )
    result, trajectory = simulate_shared_world(asset_root, **kwargs)
    artifact = _save_trajectory(output / f"probe-{index:03d}.npz", trajectory)
    active = np.asarray(trajectory["shooter_three_axis_contact_actor_active"], dtype=np.bool_)
    teacher = np.asarray(trajectory["shooter_loft_teacher_active"], dtype=np.bool_)
    return {
        "probe_index": index,
        "probe": asdict(probe),
        "probe_hash": probe.probe_hash,
        "result": result.to_dict(),
        "quality": _chain_quality(result, trajectory, quality),
        "contact_actor_active": bool(np.any(active)),
        "contact_actor_active_frame_count": int(np.count_nonzero(active)),
        "teacher_active": bool(np.any(teacher)),
        "trajectory": artifact,
    }


def _implementation_hash() -> str:
    paths = (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "three_axis_contact_actor.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
    )
    return str(hash_json({path.name: hash_bytes(path.read_bytes()) for path in paths}))


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("coherent contact discovery output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["CoherentContactProbe", "run_coherent_contact_discovery"]
