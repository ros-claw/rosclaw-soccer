"""Fresh-physics alternating Growth round for the playmaker.

The goalkeeper and finisher policies are held fixed while discovery rollouts
fit a small passer adapter.  Sealed holdout lanes then compare that adapter to
the frozen fixed-yaw parent in the same three-G1 MuJoCo scene.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.dynamic_lead_pass import (
    LeadPassCalibrationSample,
    fit_dynamic_lead_pass_policy,
)
from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.development_evidence import (
    three_role_development_kwargs,
)
from rosclaw_soccer.skills.team.shared_world import (
    G1SharedWorldResult,
    simulate_shared_world,
)


@dataclass(frozen=True)
class DynamicLeadPassHoldout:
    case_id: str
    receiver_phase_start_sec: float
    receiver_lateral_lane_m: float
    schema_version: str = "rosclaw_soccer.dynamic_lead_pass_holdout.v1"

    def __post_init__(self) -> None:
        if not self.case_id or not self.case_id.replace("-", "").isalnum():
            raise ValueError("dynamic lead-pass holdout id is invalid")
        if not 1.80 <= self.receiver_phase_start_sec <= 2.10:
            raise ValueError("dynamic lead-pass holdout phase is invalid")
        if not 0.06 <= abs(self.receiver_lateral_lane_m) <= 0.15:
            raise ValueError("dynamic lead-pass holdout must challenge a lateral lane")


@dataclass(frozen=True)
class DynamicLeadPassEvidenceConfig:
    discovery_receiver_phase_starts_sec: tuple[float, ...] = (1.80, 1.96, 2.04)
    discovery_passer_yaw_deltas_rad: tuple[float, ...] = (
        -0.06,
        -0.03,
        0.0,
        0.02,
        0.03,
        0.06,
    )
    holdouts: tuple[DynamicLeadPassHoldout, ...] = (
        # The -0.10 m discovery failure caused the frozen receiver to fall.
        # The next curriculum cell deliberately retreats only that side to
        # the nearest stable frontier instead of relaxing the stability gate.
        DynamicLeadPassHoldout("early-left-frontier", 1.92, -0.07),
        DynamicLeadPassHoldout("early-right-lane", 1.92, 0.10),
    )
    simulation_duration_sec: float = 9.0
    maximum_delivery_error_m: float = 0.05
    maximum_lateral_error_m: float = 0.02
    minimum_parent_error_reduction_m: float = 0.04
    minimum_receiver_precontact_speed_mps: float = 0.75
    minimum_longitudinal_fit_r2: float = 0.95
    minimum_lateral_fit_r2: float = 0.95
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.dynamic_lead_pass_evidence_config.v1"

    def __post_init__(self) -> None:
        values = (
            *self.discovery_receiver_phase_starts_sec,
            *self.discovery_passer_yaw_deltas_rad,
            self.simulation_duration_sec,
            self.maximum_delivery_error_m,
            self.maximum_lateral_error_m,
            self.minimum_parent_error_reduction_m,
            self.minimum_receiver_precontact_speed_mps,
            self.minimum_longitudinal_fit_r2,
            self.minimum_lateral_fit_r2,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("dynamic lead-pass evidence config must be finite")
        if len(set(self.discovery_receiver_phase_starts_sec)) < 3:
            raise ValueError("dynamic lead-pass discovery needs three receiver phases")
        if len(set(self.discovery_passer_yaw_deltas_rad)) < 5:
            raise ValueError("dynamic lead-pass discovery needs five passer yaw probes")
        if any(
            math.isclose(case.receiver_phase_start_sec, phase, abs_tol=1.0e-12)
            for case in self.holdouts
            for phase in self.discovery_receiver_phase_starts_sec
        ):
            raise ValueError("dynamic lead-pass holdout phases must be sealed from discovery")
        if len({case.case_id for case in self.holdouts}) != len(self.holdouts):
            raise ValueError("dynamic lead-pass holdout ids must be unique")
        if not 8.0 <= self.simulation_duration_sec <= 12.0:
            raise ValueError("dynamic lead-pass duration is invalid")
        if not 0.01 <= self.maximum_delivery_error_m <= 0.05:
            raise ValueError("dynamic lead-pass delivery gate is invalid")
        if not 0.005 <= self.maximum_lateral_error_m <= 0.03:
            raise ValueError("dynamic lead-pass lateral gate is invalid")
        if not 0.02 <= self.minimum_parent_error_reduction_m <= 0.15:
            raise ValueError("dynamic lead-pass improvement gate is invalid")
        if not 0.50 <= self.minimum_receiver_precontact_speed_mps <= 1.50:
            raise ValueError("dynamic lead-pass receiver speed gate is invalid")
        if not 0.80 <= min(self.minimum_longitudinal_fit_r2, self.minimum_lateral_fit_r2) <= 1.0:
            raise ValueError("dynamic lead-pass fit gates are invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("dynamic lead-pass evidence must remain SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def run_dynamic_lead_pass_evidence(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
    config: DynamicLeadPassEvidenceConfig | None = None,
) -> dict[str, Any]:
    """Fit in discovery and execute the learned passer on sealed holdouts."""

    active = config or DynamicLeadPassEvidenceConfig()
    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("dynamic lead-pass evidence must be new and outside the checkout")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if not checkout.is_dir():
        raise ValueError("dynamic lead-pass source checkout is unavailable")
    output.mkdir(parents=True)

    request = {
        "schema_version": "rosclaw_soccer.dynamic_lead_pass_request.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "source_commit": _git_head(checkout),
        "runtime": _runtime_manifest(),
        "growth_contract": {
            "plastic_role": "passer",
            "frozen_roles": ["shooter", "goalkeeper"],
            "discovery_and_holdout_disjoint": True,
            "action_conditioned_not_metric_only": True,
        },
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "pixels_used_for_scoring": False,
    }
    _write_json(output / "request.json", request)

    samples, discovery = _run_discovery(
        asset_root=asset_root,
        config=active,
    )
    policy = fit_dynamic_lead_pass_policy(samples)
    _write_json(output / "dynamic-lead-pass-policy.json", policy.to_dict())
    holdouts: dict[str, Any] = {}
    for case in active.holdouts:
        target = policy.reception_target(
            receiver_phase_start_sec=case.receiver_phase_start_sec,
            receiver_lateral_lane_m=case.receiver_lateral_lane_m,
        )
        candidate_kwargs = _simulation_kwargs(
            config=active,
            receiver_phase_start_sec=case.receiver_phase_start_sec,
            receiver_lateral_lane_m=case.receiver_lateral_lane_m,
            reception_target=target,
            passer_yaw_rad=policy.passer_world_yaw(target_lateral_m=case.receiver_lateral_lane_m),
        )
        parent_kwargs = dict(candidate_kwargs)
        parent_kwargs["passer_yaw_rad"] = math.pi
        parent_result, parent_trajectory = simulate_shared_world(asset_root, **parent_kwargs)
        candidate_result, candidate_trajectory = simulate_shared_world(
            asset_root, **candidate_kwargs
        )
        replay_result, replay_trajectory = simulate_shared_world(asset_root, **candidate_kwargs)
        strict_replay = bool(
            candidate_result.to_dict() == replay_result.to_dict()
            and trajectory_digest(candidate_trajectory) == trajectory_digest(replay_trajectory)
        )
        parent_metrics = _case_metrics(parent_result, parent_trajectory)
        candidate_metrics = _case_metrics(candidate_result, candidate_trajectory)
        parent_error = parent_result.pass_delivery_error_m
        candidate_error = candidate_result.pass_delivery_error_m
        receiver_speed = candidate_metrics["receiver_precontact_speed_mps"]
        gates = {
            "strict_replay": strict_replay,
            "ordered_physical_contacts": bool(
                candidate_result.pass_contact_time_sec is not None
                and candidate_result.shot_contact_time_sec is not None
                and candidate_result.pass_contact_time_sec < candidate_result.shot_contact_time_sec
            ),
            "conditioned_action_executed": not math.isclose(
                float(candidate_kwargs["passer_yaw_rad"]), math.pi, abs_tol=1.0e-6
            ),
            "delivery_precision": bool(
                candidate_error is not None
                and candidate_error <= active.maximum_delivery_error_m
                and candidate_result.pass_delivery_lateral_error_m is not None
                and candidate_result.pass_delivery_lateral_error_m <= active.maximum_lateral_error_m
            ),
            "beats_fixed_parent": bool(
                parent_error is not None
                and candidate_error is not None
                and parent_error - candidate_error >= active.minimum_parent_error_reduction_m
            ),
            "receiver_still_moving": bool(
                isinstance(receiver_speed, (float, int))
                and not isinstance(receiver_speed, bool)
                and math.isfinite(receiver_speed)
                and receiver_speed >= active.minimum_receiver_precontact_speed_mps
                and candidate_result.receiver_phase_hold_frames == 0
            ),
            "three_agents_present": candidate_result.goalkeeper_enabled,
            "frozen_role_stability": bool(
                candidate_result.shooter_min_pelvis_height_m >= 0.60
                and candidate_result.goalkeeper_min_pelvis_height_m is not None
                and candidate_result.goalkeeper_min_pelvis_height_m >= 0.70
            ),
            "joint_limits": not candidate_result.joint_limit_violation,
            "torque_limits": not candidate_result.torque_limit_violation,
            "zero_actuator_saturation": not candidate_result.actuator_saturation,
        }
        trajectory_path = output / f"{case.case_id}-candidate-trajectory.npz"
        np.savez_compressed(trajectory_path, **candidate_trajectory)  # type: ignore[arg-type]
        holdouts[case.case_id] = {
            "case": asdict(case),
            "passed": bool(all(gates.values())),
            "gates": gates,
            "target_m": target,
            "executed_passer_yaw_rad": candidate_kwargs["passer_yaw_rad"],
            "executed_passer_yaw_delta_rad": policy.passer_yaw_delta(
                target_lateral_m=case.receiver_lateral_lane_m
            ),
            "parent": {"result": parent_result.to_dict(), "metrics": parent_metrics},
            "candidate": {
                "result": candidate_result.to_dict(),
                "metrics": candidate_metrics,
            },
            "strict_replay": strict_replay,
            "parent_trajectory_digest": trajectory_digest(parent_trajectory),
            "candidate_trajectory_file": trajectory_path.name,
            "candidate_trajectory_hash": hash_bytes(trajectory_path.read_bytes()),
            "candidate_trajectory_digest": trajectory_digest(candidate_trajectory),
        }

    fit_gates = {
        "longitudinal_fit": policy.longitudinal_fit_r2 >= active.minimum_longitudinal_fit_r2,
        "lateral_fit": policy.lateral_fit_r2 >= active.minimum_lateral_fit_r2,
        "all_discovery_safe": all(item.safe for item in samples),
        "sealed_holdout": not {
            item.receiver_phase_start_sec for item in active.holdouts
        }.intersection(active.discovery_receiver_phase_starts_sec),
    }
    gates = {
        **fit_gates,
        "all_holdouts_passed": all(item["passed"] for item in holdouts.values()),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.dynamic_lead_pass_evidence.v1",
        "passed": bool(all(gates.values())),
        "promotion_status": (
            "FROZEN_RESEARCH_DEMO" if all(gates.values()) else "REJECTED_DEVELOPMENT"
        ),
        "claim": "CONDITIONED_MOVING_RECEIVER_LEAD_PASS_ONLY_IF_ALL_PHYSICS_GATES_PASS",
        "gates": gates,
        "policy": policy.to_dict(),
        "policy_hash": policy.artifact_hash,
        "discovery": discovery,
        "holdouts": holdouts,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "implementation_hash": _implementation_hash(),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "pixels_used_for_scoring": False,
        "fresh_training_performed": True,
    }
    report["evidence_hash"] = hash_json(report)
    _write_json(output / "evidence.json", report)
    return report


def validate_dynamic_lead_pass_evidence(path: Path) -> dict[str, Any]:
    """Recompute content bindings and fail closed on edited evidence."""

    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dynamic lead-pass evidence must be a JSON object")
    evidence_hash = payload.pop("evidence_hash", None)
    try:
        if (
            payload.get("schema_version") != "rosclaw_soccer.dynamic_lead_pass_evidence.v1"
            or payload.get("passed") is not True
            or payload.get("promotion_status") != "FROZEN_RESEARCH_DEMO"
            or payload.get("physics_authority") != "CPU_MUJOCO"
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("pixels_used_for_scoring") is not False
            or payload.get("fresh_training_performed") is not True
            or not all(payload.get("gates", {}).values())
            or hash_json(payload) != evidence_hash
        ):
            raise ValueError("dynamic lead-pass evidence authority contract is invalid")
        root = source.parent
        for case in payload.get("holdouts", {}).values():
            trajectory = root / case["candidate_trajectory_file"]
            if (
                not trajectory.is_file()
                or hash_bytes(trajectory.read_bytes()) != case["candidate_trajectory_hash"]
            ):
                raise ValueError("dynamic lead-pass trajectory binding changed")
    finally:
        if evidence_hash is not None:
            payload["evidence_hash"] = evidence_hash
    return payload


def _run_discovery(
    *, asset_root: Path, config: DynamicLeadPassEvidenceConfig
) -> tuple[tuple[LeadPassCalibrationSample, ...], dict[str, Any]]:
    samples: list[LeadPassCalibrationSample] = []
    records: dict[str, Any] = {}
    probes = [
        (f"phase-{phase:.2f}".replace(".", "p"), phase, 0.0)
        for phase in config.discovery_receiver_phase_starts_sec
    ] + [
        (f"yaw-{index}", 1.96, delta)
        for index, delta in enumerate(config.discovery_passer_yaw_deltas_rad)
    ]
    for sample_id, phase, yaw_delta in probes:
        target = (1.275, 0.0, 0.115)
        kwargs = _simulation_kwargs(
            config=config,
            receiver_phase_start_sec=phase,
            receiver_lateral_lane_m=0.0,
            reception_target=target,
            passer_yaw_rad=_world_yaw(yaw_delta),
        )
        result, trajectory = simulate_shared_world(asset_root, **kwargs)
        delivery = result.pass_delivery_position_m
        safe = bool(
            result.finite_state
            and result.pass_contact_observed
            and result.shot_contact_observed
            and delivery is not None
            and not result.joint_limit_violation
            and not result.torque_limit_violation
            and not result.actuator_saturation
        )
        if delivery is None:
            raise RuntimeError(f"lead-pass discovery {sample_id} did not reach the receiver")
        digest = trajectory_digest(trajectory)
        sample = LeadPassCalibrationSample(
            sample_id=sample_id,
            receiver_phase_start_sec=phase,
            passer_yaw_delta_rad=yaw_delta,
            delivery_position_m=delivery,
            trajectory_hash=digest,
            safe=safe,
        )
        samples.append(sample)
        records[sample_id] = {
            "sample": asdict(sample),
            "result": result.to_dict(),
            "trajectory_digest": digest,
        }
    return tuple(samples), records


def _simulation_kwargs(
    *,
    config: DynamicLeadPassEvidenceConfig,
    receiver_phase_start_sec: float,
    receiver_lateral_lane_m: float,
    reception_target: tuple[float, float, float],
    passer_yaw_rad: float,
) -> dict[str, Any]:
    kwargs = three_role_development_kwargs()
    kwargs.update(
        shooter_start_sec=receiver_phase_start_sec,
        shooter_origin=(0.0, receiver_lateral_lane_m, 0.0),
        pass_reception_target_m=reception_target,
        passer_yaw_rad=passer_yaw_rad,
        simulation_duration_sec=config.simulation_duration_sec,
    )
    return kwargs


def _case_metrics(
    result: G1SharedWorldResult, trajectory: dict[str, np.ndarray]
) -> dict[str, float | int | None]:
    contact = result.shot_contact_time_sec
    if contact is None:
        return {
            "receiver_precontact_speed_mps": 0.0,
            "receiver_precontact_displacement_m": 0.0,
            "receiver_phase_hold_frames": result.receiver_phase_hold_frames,
        }
    time = np.asarray(trajectory["time"], dtype=np.float64)
    pelvis = np.asarray(trajectory["shooter_pelvis_pose"], dtype=np.float64)[:, :2]
    velocity = np.linalg.norm(np.gradient(pelvis, time, axis=0), axis=1)
    window = (time >= contact - 0.12) & (time <= contact)
    points = pelvis[window]
    return {
        "receiver_precontact_speed_mps": float(np.mean(velocity[window])),
        "receiver_precontact_displacement_m": (
            0.0 if points.shape[0] < 2 else float(np.linalg.norm(points[-1] - points[0]))
        ),
        "receiver_phase_hold_frames": result.receiver_phase_hold_frames,
    }


def _world_yaw(delta: float) -> float:
    return math.pi + delta if delta <= 0.0 else -math.pi + delta


def _git_head(checkout: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _runtime_manifest() -> dict[str, str]:
    import mujoco
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
    }


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).parents[1] / "growth" / "dynamic_lead_pass.py"):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "DynamicLeadPassEvidenceConfig",
    "DynamicLeadPassHoldout",
    "run_dynamic_lead_pass_evidence",
    "validate_dynamic_lead_pass_evidence",
]
