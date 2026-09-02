"""Failure-driven Growth loop for a causal pass-to-shot skill hand-off.

Discovery varies the physical pass context and evaluates multiple frozen
receiver-entry timings in CPU MuJoCo.  The best safe timing from each context
becomes a small supervised sample.  GPU workers fit independent JSON-only
actors, while a fresh, sealed CPU suite remains the final authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.causal_skill_transition import (
    CausalTransitionSample,
    G1CausalSkillTransitionActor,
    fit_causal_skill_transition_actor,
    load_causal_skill_transition_actor,
    save_causal_skill_transition_actor,
)
from rosclaw_soccer.growth.dynamic_lead_pass import DynamicLeadPassPolicy
from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.development_evidence import three_role_development_kwargs
from rosclaw_soccer.skills.team.shared_world import G1SharedWorldResult, simulate_shared_world


@dataclass(frozen=True)
class CausalTransitionContext:
    case_id: str
    passer_origin_m: tuple[float, float, float]
    receiver_lane_m: float
    reception_target_x_m: float
    passer_ball_local_xy_m: tuple[float, float]
    predecessor_swing_speed_scale: float
    ball_ground_friction: float = 0.10
    schema_version: str = "rosclaw.growth.causal_transition_context.v1"

    def __post_init__(self) -> None:
        values = (
            *self.passer_origin_m,
            self.receiver_lane_m,
            self.reception_target_x_m,
            *self.passer_ball_local_xy_m,
            self.predecessor_swing_speed_scale,
            self.ball_ground_friction,
        )
        if (
            not self.case_id
            or not self.case_id.replace("-", "").replace(".", "").isalnum()
            or any(not math.isfinite(value) for value in values)
            or not 4.80 <= self.passer_origin_m[0] <= 5.40
            or abs(self.passer_origin_m[1]) > 0.40
            or abs(self.passer_origin_m[2]) > 1.0e-12
            or not -0.12 <= self.receiver_lane_m <= 0.12
            or not 1.15 <= self.reception_target_x_m <= 1.36
            or not 1.10 <= self.passer_ball_local_xy_m[0] <= 1.25
            or not -0.24 <= self.passer_ball_local_xy_m[1] <= -0.10
            or not 0.80 <= self.predecessor_swing_speed_scale <= 0.95
            or not 0.085 <= self.ball_ground_friction <= 0.115
        ):
            raise ValueError("causal transition context exceeds its SIM-only curriculum")

    @property
    def context_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class CausalTransitionGrowthConfig:
    receiver_start_candidates_sec: tuple[float, ...] = (1.88, 1.92, 1.96, 2.00, 2.04)
    parent_receiver_start_sec: float = 1.96
    simulation_duration_sec: float = 9.0
    minimum_shot_speed_mps: float = 6.0
    minimum_pelvis_height_m: float = 0.60
    minimum_success_gain_cases: int = 1
    minimum_actor_success_rate: float = 2.0 / 3.0
    minimum_qualified_development_contexts: int = 8
    minimum_incremental_qualified_contexts: int = 2
    maximum_root_step_m: float = 0.10
    maximum_ball_step_m: float = 0.35
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.growth.causal_transition_growth_config.v4"

    def __post_init__(self) -> None:
        values = (
            *self.receiver_start_candidates_sec,
            self.parent_receiver_start_sec,
            self.simulation_duration_sec,
            self.minimum_shot_speed_mps,
            self.minimum_pelvis_height_m,
            self.minimum_actor_success_rate,
            self.maximum_root_step_m,
            self.maximum_ball_step_m,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or len(self.receiver_start_candidates_sec) < 5
            or len(set(self.receiver_start_candidates_sec))
            != len(self.receiver_start_candidates_sec)
            or any(not 1.60 <= value <= 2.30 for value in self.receiver_start_candidates_sec)
            or not 1.80 <= self.parent_receiver_start_sec <= 2.10
            or not 8.0 <= self.simulation_duration_sec <= 12.0
            or not 5.0 <= self.minimum_shot_speed_mps <= 9.0
            or not 0.55 <= self.minimum_pelvis_height_m <= 0.70
            or isinstance(self.minimum_success_gain_cases, bool)
            or not 0 <= self.minimum_success_gain_cases <= 4
            or not 0.50 <= self.minimum_actor_success_rate <= 0.90
            or isinstance(self.minimum_qualified_development_contexts, bool)
            or not 8 <= self.minimum_qualified_development_contexts <= 32
            or isinstance(self.minimum_incremental_qualified_contexts, bool)
            or not 1 <= self.minimum_incremental_qualified_contexts <= 8
            or not 0.04 <= self.maximum_root_step_m <= 0.15
            or not 0.20 <= self.maximum_ball_step_m <= 0.40
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.schema_version != "rosclaw.growth.causal_transition_growth_config.v4"
        ):
            raise ValueError("causal transition Growth config violates its safety envelope")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def default_transition_development_contexts() -> tuple[CausalTransitionContext, ...]:
    return (
        CausalTransitionContext(
            "s124.dev.00", (5.10, -0.16406006503921598, 0.0), 0.00, 1.35, (1.205, -0.16), 0.80
        ),
        CausalTransitionContext(
            "s124.dev.01", (5.10, -0.16406006503921598, 0.0), -0.04, 1.33, (1.205, -0.16), 0.80
        ),
        CausalTransitionContext(
            "s124.dev.02", (5.10, -0.16406006503921598, 0.0), -0.07, 1.305, (1.205, -0.16), 0.80
        ),
        CausalTransitionContext(
            "s124.dev.03", (5.10, -0.16406006503921598, 0.0), 0.05, 1.282, (1.205, -0.16), 0.80
        ),
        CausalTransitionContext(
            "s124.dev.04", (5.10, -0.16406006503921598, 0.0), 0.00, 1.259, (1.205, -0.16), 0.80
        ),
        CausalTransitionContext(
            "s124.dev.05", (5.10, -0.16406006503921598, 0.0), 0.08, 1.236, (1.205, -0.16), 0.80
        ),
        CausalTransitionContext(
            "s124.dev.06", (5.10, -0.16406006503921598, 0.0), 0.10, 1.212, (1.205, -0.16), 0.80
        ),
        CausalTransitionContext(
            "s124.dev.07", (5.10, -0.16406006503921598, 0.0), 0.03, 1.189, (1.205, -0.16), 0.80
        ),
        CausalTransitionContext(
            "s124.dev.08", (5.10, -0.16406006503921598, 0.0), -0.02, 1.340, (1.205, -0.16), 0.80
        ),
        CausalTransitionContext(
            "s124.dev.09", (5.10, -0.16406006503921598, 0.0), -0.055, 1.318, (1.205, -0.16), 0.80
        ),
        CausalTransitionContext(
            "s124.dev.10", (5.10, -0.16406006503921598, 0.0), 0.02, 1.270, (1.205, -0.16), 0.80
        ),
        CausalTransitionContext(
            "s124.dev.11", (5.10, -0.16406006503921598, 0.0), 0.06, 1.247, (1.205, -0.16), 0.80
        ),
        # Failure-driven replay: these were sealed holdouts for generation v1.
        # That generation was rejected, so they may inform v2 development;
        # v2 receives a wholly new sealed suite below.
        CausalTransitionContext(
            "s124.dev.12",
            (5.10, -0.16406006503921598, 0.0),
            -0.02,
            1.340,
            (1.205, -0.16),
            0.80,
            0.095,
        ),
        CausalTransitionContext(
            "s124.dev.13",
            (5.10, -0.16406006503921598, 0.0),
            -0.06,
            1.294,
            (1.205, -0.16),
            0.80,
            0.105,
        ),
        CausalTransitionContext(
            "s124.dev.14",
            (5.10, -0.16406006503921598, 0.0),
            0.04,
            1.271,
            (1.205, -0.16),
            0.80,
            0.090,
        ),
        CausalTransitionContext(
            "s124.dev.15",
            (5.10, -0.16406006503921598, 0.0),
            0.07,
            1.247,
            (1.205, -0.16),
            0.80,
            0.110,
        ),
        CausalTransitionContext(
            "s124.dev.16",
            (5.10, -0.16406006503921598, 0.0),
            0.09,
            1.224,
            (1.205, -0.16),
            0.80,
            0.0975,
        ),
        CausalTransitionContext(
            "s124.dev.17",
            (5.10, -0.16406006503921598, 0.0),
            0.02,
            1.201,
            (1.205, -0.16),
            0.80,
            0.1025,
        ),
        # Failure-driven replay from the rejected v2 sealed exam.
        CausalTransitionContext(
            "s124.dev.18",
            (5.10, -0.16406006503921598, 0.0),
            -0.015,
            1.345,
            (1.205, -0.16),
            0.80,
            0.101,
        ),
        CausalTransitionContext(
            "s124.dev.19",
            (5.10, -0.16406006503921598, 0.0),
            -0.05,
            1.315,
            (1.205, -0.16),
            0.80,
            0.099,
        ),
        CausalTransitionContext(
            "s124.dev.20",
            (5.10, -0.16406006503921598, 0.0),
            -0.065,
            1.300,
            (1.205, -0.16),
            0.80,
            0.1075,
        ),
        CausalTransitionContext(
            "s124.dev.21",
            (5.10, -0.16406006503921598, 0.0),
            0.01,
            1.265,
            (1.205, -0.16),
            0.80,
            0.0925,
        ),
        CausalTransitionContext(
            "s124.dev.22",
            (5.10, -0.16406006503921598, 0.0),
            0.075,
            1.240,
            (1.205, -0.16),
            0.80,
            0.108,
        ),
        CausalTransitionContext(
            "s124.dev.23",
            (5.10, -0.16406006503921598, 0.0),
            0.095,
            1.218,
            (1.205, -0.16),
            0.80,
            0.096,
        ),
        # Failure-driven replay from the rejected v3 sealed exam.
        CausalTransitionContext(
            "s124.dev.24",
            (5.10, -0.16406006503921598, 0.0),
            -0.012,
            1.342,
            (1.205, -0.16),
            0.80,
            0.103,
        ),
        CausalTransitionContext(
            "s124.dev.25",
            (5.10, -0.16406006503921598, 0.0),
            -0.047,
            1.322,
            (1.205, -0.16),
            0.80,
            0.097,
        ),
        CausalTransitionContext(
            "s124.dev.26",
            (5.10, -0.16406006503921598, 0.0),
            -0.062,
            1.306,
            (1.205, -0.16),
            0.80,
            0.109,
        ),
        CausalTransitionContext(
            "s124.dev.27",
            (5.10, -0.16406006503921598, 0.0),
            0.015,
            1.260,
            (1.205, -0.16),
            0.80,
            0.094,
        ),
        CausalTransitionContext(
            "s124.dev.28",
            (5.10, -0.16406006503921598, 0.0),
            0.072,
            1.243,
            (1.205, -0.16),
            0.80,
            0.106,
        ),
        CausalTransitionContext(
            "s124.dev.29",
            (5.10, -0.16406006503921598, 0.0),
            0.092,
            1.221,
            (1.205, -0.16),
            0.80,
            0.098,
        ),
        # Failure-driven replay from the rejected v4 risk-sensitive exam.
        CausalTransitionContext(
            "s124.dev.30",
            (5.10, -0.16406006503921598, 0.0),
            -0.01,
            1.338,
            (1.205, -0.16),
            0.80,
            0.104,
        ),
        CausalTransitionContext(
            "s124.dev.31",
            (5.10, -0.16406006503921598, 0.0),
            -0.043,
            1.326,
            (1.205, -0.16),
            0.80,
            0.096,
        ),
        CausalTransitionContext(
            "s124.dev.32",
            (5.10, -0.16406006503921598, 0.0),
            -0.058,
            1.310,
            (1.205, -0.16),
            0.80,
            0.111,
        ),
        CausalTransitionContext(
            "s124.dev.33",
            (5.10, -0.16406006503921598, 0.0),
            0.02,
            1.257,
            (1.205, -0.16),
            0.80,
            0.093,
        ),
        CausalTransitionContext(
            "s124.dev.34",
            (5.10, -0.16406006503921598, 0.0),
            0.068,
            1.246,
            (1.205, -0.16),
            0.80,
            0.1045,
        ),
        CausalTransitionContext(
            "s124.dev.35",
            (5.10, -0.16406006503921598, 0.0),
            0.088,
            1.225,
            (1.205, -0.16),
            0.80,
            0.099,
        ),
        # Failure-driven replay from the rejected v5 memory exam.
        CausalTransitionContext(
            "s124.dev.36",
            (5.10, -0.16406006503921598, 0.0),
            -0.018,
            1.343,
            (1.205, -0.16),
            0.80,
            0.102,
        ),
        CausalTransitionContext(
            "s124.dev.37",
            (5.10, -0.16406006503921598, 0.0),
            -0.052,
            1.319,
            (1.205, -0.16),
            0.80,
            0.098,
        ),
        CausalTransitionContext(
            "s124.dev.38",
            (5.10, -0.16406006503921598, 0.0),
            -0.055,
            1.313,
            (1.205, -0.16),
            0.80,
            0.112,
        ),
        CausalTransitionContext(
            "s124.dev.39",
            (5.10, -0.16406006503921598, 0.0),
            0.025,
            1.254,
            (1.205, -0.16),
            0.80,
            0.0915,
        ),
        CausalTransitionContext(
            "s124.dev.40",
            (5.10, -0.16406006503921598, 0.0),
            0.065,
            1.249,
            (1.205, -0.16),
            0.80,
            0.103,
        ),
        CausalTransitionContext(
            "s124.dev.41",
            (5.10, -0.16406006503921598, 0.0),
            0.085,
            1.228,
            (1.205, -0.16),
            0.80,
            0.1005,
        ),
    )


def default_transition_holdouts() -> tuple[CausalTransitionContext, ...]:
    return (
        CausalTransitionContext(
            "s124.holdout.v6.00",
            (5.10, -0.16406006503921598, 0.0),
            -0.021,
            1.341,
            (1.205, -0.16),
            0.80,
            0.1008,
        ),
        CausalTransitionContext(
            "s124.holdout.v6.01",
            (5.10, -0.16406006503921598, 0.0),
            -0.054,
            1.317,
            (1.205, -0.16),
            0.80,
            0.1002,
        ),
        CausalTransitionContext(
            "s124.holdout.v6.02",
            (5.10, -0.16406006503921598, 0.0),
            -0.053,
            1.316,
            (1.205, -0.16),
            0.80,
            0.113,
        ),
        CausalTransitionContext(
            "s124.holdout.v6.03",
            (5.10, -0.16406006503921598, 0.0),
            0.03,
            1.251,
            (1.205, -0.16),
            0.80,
            0.0905,
        ),
        CausalTransitionContext(
            "s124.holdout.v6.04",
            (5.10, -0.16406006503921598, 0.0),
            0.062,
            1.251,
            (1.205, -0.16),
            0.80,
            0.1015,
        ),
        CausalTransitionContext(
            "s124.holdout.v6.05",
            (5.10, -0.16406006503921598, 0.0),
            0.082,
            1.230,
            (1.205, -0.16),
            0.80,
            0.0985,
        ),
    )


def run_causal_transition_discovery(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    source_s123_dir: Path,
    output_dir: Path,
    config: CausalTransitionGrowthConfig | None = None,
    contexts: tuple[CausalTransitionContext, ...] | None = None,
    source_failure_memory_path: Path | None = None,
    source_discovery_report_path: Path | None = None,
    workers: int = 4,
) -> tuple[tuple[CausalTransitionSample, ...], dict[str, Any]]:
    """Select one best safe trigger per context from fresh CPU physics."""

    active = config or CausalTransitionGrowthConfig()
    selected_contexts = contexts or default_transition_development_contexts()
    if not selected_contexts or len({case.context_hash for case in selected_contexts}) != len(
        selected_contexts
    ):
        raise ValueError("causal transition discovery needs unique new contexts")
    output = _new_external_output(output_dir)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    source_stage = _load_bound_json(source_s123_dir / "stage-summary.json", "stage_hash")
    if source_stage.get("status") != "PASS_TEMPORAL_MULTI_AGENT_GROWTH_STAGE":
        raise ValueError("causal transition source S123 stage is not passing")
    lead_policy, source_lead = _load_lead_policy(source_s95_dir)
    reused_samples: tuple[CausalTransitionSample, ...] = ()
    source_discovery: dict[str, Any] | None = None
    if source_discovery_report_path is not None:
        reused_samples, source_discovery = _load_discovery_samples(source_discovery_report_path)
        if (
            source_discovery.get("source_s123_stage_hash") != source_stage["stage_hash"]
            or source_discovery.get("source_s95_policy_hash") != lead_policy.artifact_hash
        ):
            raise ValueError("causal transition source discovery lineage changed")
        reused_ids = {sample.sample_id for sample in reused_samples}
        if any(case.case_id in reused_ids for case in selected_contexts):
            raise ValueError("causal transition incremental context id was already learned")
    source_failure_memory: dict[str, Any] | None = None
    if source_failure_memory_path is not None:
        source_failure_memory = _load_bound_json(source_failure_memory_path, "report_hash")
        if source_failure_memory.get("status") != "REJECTED_CAUSAL_CHAIN":
            raise ValueError("causal transition failure memory must be a rejected sealed exam")
        replay_signatures = {
            _context_physics_signature(row["context"])
            for row in source_failure_memory.get("rows", [])
        }
        selected_signatures = {
            _context_physics_signature(asdict(case)) for case in selected_contexts
        }
        if not replay_signatures or not replay_signatures <= selected_signatures:
            raise ValueError("causal transition failure contexts were not replayed in development")
    request = {
        "schema_version": "rosclaw.growth.causal_transition_discovery_request.v3",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "contexts": [asdict(case) for case in selected_contexts],
        "context_hashes": [case.context_hash for case in selected_contexts],
        "source_s95_evidence_hash": source_lead["evidence_hash"],
        "source_s95_policy_hash": lead_policy.artifact_hash,
        "source_s123_stage_hash": source_stage["stage_hash"],
        "source_failure_memory_report_hash": (
            None if source_failure_memory is None else source_failure_memory["report_hash"]
        ),
        "source_failure_memory_file_hash": (
            None
            if source_failure_memory_path is None
            else hash_bytes(source_failure_memory_path.expanduser().resolve().read_bytes())
        ),
        "source_discovery_report_hash": (
            None if source_discovery is None else source_discovery["report_hash"]
        ),
        "source_discovery_file_hash": (
            None
            if source_discovery_report_path is None
            else hash_bytes(source_discovery_report_path.expanduser().resolve().read_bytes())
        ),
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "runtime": _runtime_manifest(),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "pixels_used_for_scoring": False,
    }
    _write_json(output / "request.json", request)
    jobs = [
        (asset_root, lead_policy, active, case, start)
        for case in selected_contexts
        for start in active.receiver_start_candidates_sec
    ]
    if not 1 <= workers <= 8:
        raise ValueError("causal transition discovery workers must be in [1, 8]")
    if workers == 1:
        records = [_run_timing_probe(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_run_timing_probe, jobs))
    by_case: dict[str, list[dict[str, Any]]] = {case.case_id: [] for case in selected_contexts}
    for record in records:
        by_case[str(record["case_id"])].append(record)
    _write_json(
        output / "timing-probes.json",
        {
            "schema_version": "rosclaw.growth.causal_transition_timing_probes.v1",
            "records": records,
            "records_hash": hash_json(records),
            "physics_authority": "CPU_MUJOCO",
        },
    )

    samples: list[CausalTransitionSample] = list(reused_samples)
    new_samples: list[CausalTransitionSample] = []
    case_reports: dict[str, Any] = {}
    rejected_cases: dict[str, Any] = {}
    for case in selected_contexts:
        probes = sorted(
            by_case[case.case_id],
            key=lambda row: (-float(row["selection_score"]), float(row["receiver_start_sec"])),
        )
        eligible = [row for row in probes if row["chain_passed"]]
        if not eligible:
            rejected_cases[case.case_id] = {
                "context": asdict(case),
                "context_hash": case.context_hash,
                "probes": probes,
            }
            continue
        best = eligible[0]
        best_result, best_trajectory = _simulate_context(
            asset_root=asset_root,
            lead_policy=lead_policy,
            config=active,
            context=case,
            receiver_start_sec=float(best["receiver_start_sec"]),
        )
        artifact_path = output / f"{case.case_id}-best-trajectory.npz"
        np.savez_compressed(artifact_path, **best_trajectory)  # type: ignore[arg-type]
        entry_index = int(
            np.searchsorted(
                np.asarray(best_trajectory["time"], dtype=np.float64),
                float(best["receiver_start_sec"]),
                side="left",
            )
        )
        trigger_frame = int(
            np.asarray(best_trajectory["passer_policy_frame"], dtype=np.int64)[entry_index]
        )
        features = tuple(
            float(value)
            for value in np.asarray(
                best_trajectory["shooter_transition_features"], dtype=np.float64
            )[0]
        )
        sample = CausalTransitionSample(
            sample_id=case.case_id,
            features=features,
            optimal_trigger_policy_frame=trigger_frame,
            source_trajectory_hash=hash_bytes(artifact_path.read_bytes()),
            safe=True,
        )
        samples.append(sample)
        new_samples.append(sample)
        case_reports[case.case_id] = {
            "context": asdict(case),
            "context_hash": case.context_hash,
            "selected": best,
            "selected_result": best_result.to_dict(),
            "selected_trigger_policy_frame": trigger_frame,
            "selected_features": list(features),
            "trajectory_file": artifact_path.name,
            "trajectory_file_hash": sample.source_trajectory_hash,
            "trajectory_digest": trajectory_digest(best_trajectory),
            "probes": probes,
        }
    insufficient_total = len(samples) < active.minimum_qualified_development_contexts
    insufficient_increment = bool(
        reused_samples and len(new_samples) < active.minimum_incremental_qualified_contexts
    )
    if insufficient_total or insufficient_increment:
        rejection = {
            "schema_version": "rosclaw.growth.causal_transition_discovery_rejection.v2",
            "status": "REJECTED_INSUFFICIENT_QUALIFIED_HANDOFFS",
            "context_count": len(selected_contexts),
            "qualified_context_count": len(samples),
            "minimum_qualified_context_count": (active.minimum_qualified_development_contexts),
            "new_qualified_context_count": len(new_samples),
            "minimum_incremental_qualified_context_count": (
                active.minimum_incremental_qualified_contexts
            ),
            "rejected_context_count": len(rejected_cases),
            "rejected_cases": rejected_cases,
            "failure_memory_preserved": True,
        }
        rejection["report_hash"] = hash_json(rejection)
        _write_json(output / "rejected-discovery.json", rejection)
        raise RuntimeError(
            "causal transition discovery has only "
            f"{len(samples)} total/{len(new_samples)} new qualified contexts"
        )
    payload: dict[str, Any] = {
        "schema_version": "rosclaw.growth.causal_transition_discovery.v3",
        "status": "PASS_CAUSAL_TRANSITION_DISCOVERY",
        "context_count": len(selected_contexts),
        "sample_count": len(samples),
        "new_sample_count": len(new_samples),
        "reused_sample_count": len(reused_samples),
        "minimum_qualified_context_count": active.minimum_qualified_development_contexts,
        "minimum_incremental_qualified_context_count": (
            active.minimum_incremental_qualified_contexts
        ),
        "rejected_context_count": len(rejected_cases),
        "rejected_cases": rejected_cases,
        "samples": [asdict(sample) for sample in samples],
        "cases": case_reports,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "source_s123_stage_hash": source_stage["stage_hash"],
        "source_s95_policy_hash": lead_policy.artifact_hash,
        "source_failure_memory_report_hash": (
            None if source_failure_memory is None else source_failure_memory["report_hash"]
        ),
        "source_discovery_report_hash": (
            None if source_discovery is None else source_discovery["report_hash"]
        ),
        "failure_memory_consumed": source_failure_memory is not None,
        "implementation_hash": _implementation_hash(),
        "failure_memory_preserved": True,
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "pixels_used_for_scoring": False,
    }
    payload["report_hash"] = hash_json(payload)
    _write_json(output / "discovery-report.json", payload)
    return tuple(samples), payload


def train_causal_transition_population(
    *,
    samples: tuple[CausalTransitionSample, ...],
    source_stage_hash: str,
    output_dir: Path,
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
    seeds: tuple[int, ...] = (1240, 1241, 1242, 1243),
    epochs: int = 1500,
) -> tuple[G1CausalSkillTransitionActor, dict[str, Any]]:
    """Fit four candidates, select without the final sealed CPU holdout."""

    if len(samples) < 8 or len(devices) != len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("causal transition population mapping is invalid")
    output = _new_external_output(output_dir)
    validation = samples[1::4]
    training = tuple(sample for sample in samples if sample not in validation)
    if len(training) < 6 or len(validation) < 2:
        raise ValueError("causal transition population split is too small")
    candidates: list[tuple[G1CausalSkillTransitionActor, dict[str, Any]]] = []
    for index, (device, seed) in enumerate(zip(devices, seeds, strict=True)):
        actor = fit_causal_skill_transition_actor(
            training,
            source_stage_hash=source_stage_hash,
            seed=seed,
            hidden_size=8,
            epochs=epochs,
            device=device,
        )
        metrics = _actor_metrics(actor, validation)
        path = output / f"candidate-{index:02d}.json"
        save_causal_skill_transition_actor(actor, path)
        candidates.append(
            (
                actor,
                {
                    "candidate_index": index,
                    "device": device,
                    "seed": seed,
                    "actor_hash": actor.actor_hash,
                    "actor_file": path.name,
                    "actor_file_hash": hash_bytes(path.read_bytes()),
                    "training_rmse_frames": actor.training_rmse_frames,
                    "validation": metrics,
                },
            )
        )
    candidates.sort(
        key=lambda item: (
            float(item[1]["validation"]["rmse_frames"]),
            int(item[1]["seed"]),
        )
    )
    champion, selected = candidates[0]
    seed = int(selected["seed"])
    champion_path = output / "causal-transition-champion.json"
    save_causal_skill_transition_actor(champion, champion_path)
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.causal_transition_population.v3",
        "status": "PASS_CAUSAL_TRANSITION_POPULATION",
        "source_stage_hash": source_stage_hash,
        "candidate_count": len(candidates),
        "candidates": [item[1] for item in candidates],
        "selected_candidate_index": selected["candidate_index"],
        "selected_seed": seed,
        "champion_actor_hash": champion.actor_hash,
        "champion_file": champion_path.name,
        "champion_file_hash": hash_bytes(champion_path.read_bytes()),
        "champion_all_development": _actor_metrics(champion, samples),
        "champion_refit_on_validation": False,
        "four_gpu_training_only": all(device.startswith("cuda:") for device in devices),
        "final_runtime": "NUMPY_JSON_ONLY",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "pixels_used_for_scoring": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "population-report.json", report)
    return champion, report


def run_causal_transition_retention_exam(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    actor_path: Path,
    output_dir: Path,
    config: CausalTransitionGrowthConfig | None = None,
    holdouts: tuple[CausalTransitionContext, ...] | None = None,
) -> dict[str, Any]:
    """Matched actor/parent/replay exam in fresh CPU MuJoCo contexts."""

    active = config or CausalTransitionGrowthConfig()
    cases = holdouts or default_transition_holdouts()
    if len(cases) < 6 or len({case.context_hash for case in cases}) != len(cases):
        raise ValueError("causal transition holdout must contain six unique contexts")
    output = _new_external_output(output_dir)
    actor = load_causal_skill_transition_actor(actor_path)
    lead_policy, source_lead = _load_lead_policy(source_s95_dir)
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        actor_kwargs = _context_kwargs(
            lead_policy=lead_policy,
            config=active,
            context=case,
            receiver_start_sec=active.parent_receiver_start_sec,
        )
        actor_kwargs.update(
            receiver_phase_sync_enabled=True,
            shooter_transition_actor_path=actor_path,
        )
        parent_kwargs = _context_kwargs(
            lead_policy=lead_policy,
            config=active,
            context=case,
            receiver_start_sec=active.parent_receiver_start_sec,
        )
        parent_kwargs["receiver_phase_sync_enabled"] = False
        actor_result, actor_trajectory = simulate_shared_world(asset_root, **actor_kwargs)
        replay_result, replay_trajectory = simulate_shared_world(asset_root, **actor_kwargs)
        parent_result, parent_trajectory = simulate_shared_world(asset_root, **parent_kwargs)
        case_dir = output / f"case-{index:03d}"
        case_dir.mkdir(parents=True)
        actor_artifact = _save_trajectory(case_dir / "actor-primary.npz", actor_trajectory)
        replay_artifact = _save_trajectory(case_dir / "actor-replay.npz", replay_trajectory)
        parent_artifact = _save_trajectory(case_dir / "parent.npz", parent_trajectory)
        actor_quality = _chain_quality(actor_result, actor_trajectory, active)
        parent_quality = _chain_quality(parent_result, parent_trajectory, active)
        rows.append(
            {
                "case_id": case.case_id,
                "context": asdict(case),
                "context_hash": case.context_hash,
                "actor": {"result": actor_result.to_dict(), "quality": actor_quality},
                "parent": {"result": parent_result.to_dict(), "quality": parent_quality},
                "actor_chain_passed": actor_quality["chain_passed"],
                "parent_chain_passed": parent_quality["chain_passed"],
                "actor_safe": actor_quality["safe"],
                "parent_safe": parent_quality["safe"],
                "exact_replay": bool(
                    actor_result.to_dict() == replay_result.to_dict()
                    and actor_artifact["trajectory_digest"] == replay_artifact["trajectory_digest"]
                ),
                "actor_artifact": actor_artifact,
                "replay_artifact": replay_artifact,
                "parent_artifact": parent_artifact,
            }
        )
    count = len(rows)
    actor_success = sum(row["actor_chain_passed"] for row in rows)
    parent_success = sum(row["parent_chain_passed"] for row in rows)
    metrics = {
        "case_count": count,
        "actor_chain_success_count": actor_success,
        "parent_chain_success_count": parent_success,
        "actor_chain_success_rate": actor_success / count,
        "parent_chain_success_rate": parent_success / count,
        "actor_safe_rate": sum(row["actor_safe"] for row in rows) / count,
        "parent_safe_rate": sum(row["parent_safe"] for row in rows) / count,
        "exact_replay_rate": sum(row["exact_replay"] for row in rows) / count,
        "actor_goal_count": sum(row["actor"]["result"]["goal_crossed"] for row in rows),
        "actor_save_count": sum(row["actor"]["result"]["goalkeeper_save_observed"] for row in rows),
        "mean_actor_shot_speed_mps": float(
            np.mean([row["actor"]["result"]["shot_peak_ball_speed_mps"] for row in rows])
        ),
        "mean_parent_shot_speed_mps": float(
            np.mean([row["parent"]["result"]["shot_peak_ball_speed_mps"] for row in rows])
        ),
        "nonzero_transition_residual_case_count": sum(
            row["actor"]["result"]["shooter_transition_residual_frames"] != 0 for row in rows
        ),
        "maximum_actor_root_step_m": max(
            row["actor"]["quality"]["maximum_root_step_m"] for row in rows
        ),
        "maximum_actor_ball_step_m": max(
            row["actor"]["quality"]["maximum_ball_step_m"] for row in rows
        ),
    }
    gates = {
        "all_actor_transitions_accepted": all(
            row["actor"]["result"]["shooter_transition_actor_accepted"] for row in rows
        ),
        "all_actor_transitions_triggered": all(
            row["actor"]["result"]["shooter_transition_triggered"] for row in rows
        ),
        "actor_safe_rate": metrics["actor_safe_rate"] == 1.0,
        "parent_safe_rate": metrics["parent_safe_rate"] == 1.0,
        "exact_replay_rate": metrics["exact_replay_rate"] == 1.0,
        "actor_not_worse_than_parent": actor_success >= parent_success,
        "actor_has_measured_gain": actor_success
        >= parent_success + active.minimum_success_gain_cases,
        "actor_success_rate": (
            actor_success / count + 1.0e-12 >= active.minimum_actor_success_rate
        ),
        "both_goal_and_save_outcomes": metrics["actor_goal_count"] >= 1
        and metrics["actor_save_count"] >= 1,
        "learned_residual_exercised": metrics["nonzero_transition_residual_case_count"] >= 2,
        "continuous_root_state": metrics["maximum_actor_root_step_m"] <= active.maximum_root_step_m,
        "continuous_ball_state": metrics["maximum_actor_ball_step_m"] <= active.maximum_ball_step_m,
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.causal_transition_retention_exam.v4",
        "status": (
            "PASS_CAUSAL_CONTINUOUS_CHAIN" if all(gates.values()) else "REJECTED_CAUSAL_CHAIN"
        ),
        "actor_hash": actor.actor_hash,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "actor_implementation_hash": actor.implementation_hash,
        "minimum_actor_success_rate": active.minimum_actor_success_rate,
        "source_s95_policy_hash": lead_policy.artifact_hash,
        "source_s95_evidence_hash": source_lead["evidence_hash"],
        "holdout_context_hashes": [case.context_hash for case in cases],
        "metrics": metrics,
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "evidence_boundary": {
            "activation_ceiling": "SIM_ONLY",
            "physics_authority": "CPU_MUJOCO",
            "whole_body_g1_count": 3,
            "one_shared_solver_and_ball": True,
            "world_reset_after_pass_or_shot": False,
            "pose_or_ball_teleport_after_start": False,
            "pass_and_shot_from_measured_foot_contacts": True,
            "transition_actor_pose_joint_torque_or_ball_authority": False,
            "out_of_support_route": "FROZEN_PARENT_TRIGGER",
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "retention-exam.json", report)
    return report


def validate_causal_transition_retention(path: Path) -> dict[str, Any]:
    """Fail closed when a passing sealed report or artifact is edited."""

    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("causal transition retention report must be an object")
    report_hash = payload.pop("report_hash", None)
    try:
        if (
            payload.get("schema_version") != "rosclaw.growth.causal_transition_retention_exam.v4"
            or payload.get("status") != "PASS_CAUSAL_CONTINUOUS_CHAIN"
            or hash_json(payload) != report_hash
            or not all(payload.get("gates", {}).values())
        ):
            raise ValueError("causal transition retention authority contract is invalid")
        boundary = payload.get("evidence_boundary", {})
        if (
            boundary.get("activation_ceiling") != "SIM_ONLY"
            or boundary.get("physics_authority") != "CPU_MUJOCO"
            or boundary.get("hardware_command_sent") is not False
            or boundary.get("pixels_used_for_scoring") is not False
            or boundary.get("transition_actor_pose_joint_torque_or_ball_authority") is not False
        ):
            raise ValueError("causal transition evidence boundary is invalid")
        root = source.parent
        for row in payload.get("rows", []):
            case_dir = root / f"case-{payload['rows'].index(row):03d}"
            for key in ("actor_artifact", "replay_artifact", "parent_artifact"):
                artifact = row[key]
                file_path = case_dir / artifact["file"]
                if (
                    not file_path.is_file()
                    or hash_bytes(file_path.read_bytes()) != artifact["file_hash"]
                ):
                    raise ValueError("causal transition trajectory binding changed")
    finally:
        if report_hash is not None:
            payload["report_hash"] = report_hash
    return payload


def _run_timing_probe(
    job: tuple[
        Path,
        DynamicLeadPassPolicy,
        CausalTransitionGrowthConfig,
        CausalTransitionContext,
        float,
    ],
) -> dict[str, Any]:
    asset_root, lead_policy, config, context, receiver_start_sec = job
    result, trajectory = _simulate_context(
        asset_root=asset_root,
        lead_policy=lead_policy,
        config=config,
        context=context,
        receiver_start_sec=receiver_start_sec,
    )
    quality = _chain_quality(result, trajectory, config)
    return {
        "case_id": context.case_id,
        "receiver_start_sec": receiver_start_sec,
        "safe": quality["safe"],
        "chain_passed": quality["chain_passed"],
        "clear_outcome": quality["clear_outcome"],
        "selection_score": _selection_score(result, quality),
        "pass_contact_time_sec": result.pass_contact_time_sec,
        "shot_contact_time_sec": result.shot_contact_time_sec,
        "shot_peak_ball_speed_mps": result.shot_peak_ball_speed_mps,
        "pass_delivery_error_m": result.pass_delivery_error_m,
        "goal_crossed": result.goal_crossed,
        "goalkeeper_save_observed": result.goalkeeper_save_observed,
        "shooter_min_pelvis_height_m": result.shooter_min_pelvis_height_m,
        "shooter_post_contact_support_foot_slip_m": (
            result.shooter_post_contact_support_foot_slip_m
        ),
        "trajectory_digest": trajectory_digest(trajectory),
    }


def _simulate_context(
    *,
    asset_root: Path,
    lead_policy: DynamicLeadPassPolicy,
    config: CausalTransitionGrowthConfig,
    context: CausalTransitionContext,
    receiver_start_sec: float,
) -> tuple[G1SharedWorldResult, dict[str, NDArray[Any]]]:
    return simulate_shared_world(
        asset_root,
        **_context_kwargs(
            lead_policy=lead_policy,
            config=config,
            context=context,
            receiver_start_sec=receiver_start_sec,
        ),
    )


def _context_kwargs(
    *,
    lead_policy: DynamicLeadPassPolicy,
    config: CausalTransitionGrowthConfig,
    context: CausalTransitionContext,
    receiver_start_sec: float,
) -> dict[str, Any]:
    kwargs = three_role_development_kwargs()
    nominal_passer_ball_local_xy = (1.205, -0.160)
    stance_alignment = (
        context.passer_ball_local_xy_m[0] - nominal_passer_ball_local_xy[0],
        context.passer_ball_local_xy_m[1] - nominal_passer_ball_local_xy[1],
    )
    reception_target = (
        context.reception_target_x_m,
        context.receiver_lane_m,
        0.115,
    )
    kwargs.update(
        shooter_start_sec=receiver_start_sec,
        shooter_origin=(0.0, context.receiver_lane_m, 0.0),
        passer_origin=context.passer_origin_m,
        passer_ball_local_xy=context.passer_ball_local_xy_m,
        passer_parameter_overrides={
            "swing_speed_scale": context.predecessor_swing_speed_scale,
            # Keep the immutable football where the context placed it while
            # expressing the predecessor's stance in the qualified contact
            # frame.  No qpos or ball state is changed after simulation start.
            "stance_offset_x": stance_alignment[0],
            "stance_offset_y": stance_alignment[1],
        },
        pass_reception_target_m=reception_target,
        passer_yaw_rad=lead_policy.passer_world_yaw(target_lateral_m=context.receiver_lane_m),
        ball_ground_friction=context.ball_ground_friction,
        receiver_phase_sync_enabled=False,
        # Small ball-placement changes shift the pass impact by millimetres.
        # Guarding only after contact allowed the predecessor waist pitch to
        # cross its mechanical range during the final pre-impact control tick.
        passer_precontact_joint_guard_enabled=True,
        simulation_duration_sec=config.simulation_duration_sec,
    )
    return kwargs


def _chain_quality(
    result: G1SharedWorldResult,
    trajectory: dict[str, NDArray[Any]],
    config: CausalTransitionGrowthConfig,
) -> dict[str, Any]:
    ordered = bool(
        result.pass_contact_time_sec is not None
        and result.shot_contact_time_sec is not None
        and result.pass_contact_time_sec < result.shot_contact_time_sec
    )
    safe = bool(
        result.finite_state
        and result.passer_min_pelvis_height_m >= config.minimum_pelvis_height_m
        and result.shooter_min_pelvis_height_m >= config.minimum_pelvis_height_m
        and result.goalkeeper_min_pelvis_height_m is not None
        and result.goalkeeper_min_pelvis_height_m >= config.minimum_pelvis_height_m
        and not result.joint_limit_violation
        and not result.torque_limit_violation
        and not result.actuator_saturation
    )
    roles = np.asarray(trajectory["ball_contact_role"], dtype=np.int64)
    role_order = tuple(int(value) for value in roles[roles != 0])
    first_passer = next((index for index, value in enumerate(role_order) if value == 1), None)
    first_shooter = next((index for index, value in enumerate(role_order) if value == 2), None)
    physical_order = bool(
        first_passer is not None and first_shooter is not None and first_passer < first_shooter
    )
    clear_outcome = bool(result.goal_crossed or result.goalkeeper_save_observed)
    maximum_root_step = max(
        _maximum_step(np.asarray(trajectory[f"{role}_pelvis_pose"], dtype=np.float64)[:, :3])
        for role in ("passer", "shooter", "goalkeeper")
    )
    maximum_ball_step = _maximum_step(np.asarray(trajectory["ball_pose"], dtype=np.float64)[:, :3])
    chain_passed = bool(
        safe
        and ordered
        and physical_order
        and result.shot_peak_ball_speed_mps >= config.minimum_shot_speed_mps
        and clear_outcome
        and maximum_root_step <= config.maximum_root_step_m
        and maximum_ball_step <= config.maximum_ball_step_m
    )
    return {
        "chain_passed": chain_passed,
        "safe": safe,
        "ordered_contacts": ordered,
        "physical_contact_role_order": physical_order,
        "clear_outcome": clear_outcome,
        "maximum_root_step_m": maximum_root_step,
        "maximum_ball_step_m": maximum_ball_step,
        "final_shooter_pelvis_height_m": float(
            np.asarray(trajectory["shooter_pelvis_pose"], dtype=np.float64)[-1, 2]
        ),
        "final_goalkeeper_pelvis_height_m": float(
            np.asarray(trajectory["goalkeeper_pelvis_pose"], dtype=np.float64)[-1, 2]
        ),
    }


def _selection_score(result: G1SharedWorldResult, quality: dict[str, Any]) -> float:
    if not quality["safe"]:
        return -1000.0
    outcome = 1.0 if quality["clear_outcome"] else 0.0
    ordered = 1.0 if quality["ordered_contacts"] else 0.0
    pass_error = 1.0 if result.pass_delivery_error_m is None else result.pass_delivery_error_m
    return float(
        6.0 * outcome
        + 2.0 * ordered
        + min(result.shot_peak_ball_speed_mps, 12.0) / 4.0
        - 2.0 * min(pass_error, 1.0)
        - result.shooter_post_contact_support_foot_slip_m
        - 0.25 * result.shooter_tail_wobble_index
    )


def _actor_metrics(
    actor: G1CausalSkillTransitionActor,
    samples: tuple[CausalTransitionSample, ...],
) -> dict[str, Any]:
    predicted = [
        actor.decide(np.asarray(sample.features, dtype=np.float64)).trigger_policy_frame
        for sample in samples
    ]
    labels = [sample.optimal_trigger_policy_frame for sample in samples]
    errors = np.asarray(predicted, dtype=np.float64) - np.asarray(labels, dtype=np.float64)
    return {
        "sample_count": len(samples),
        "rmse_frames": float(np.sqrt(np.mean(np.square(errors)))),
        "maximum_abs_error_frames": float(np.max(np.abs(errors))),
        "accepted_rate": float(
            np.mean(
                [
                    actor.decide(np.asarray(sample.features, dtype=np.float64)).accepted
                    for sample in samples
                ]
            )
        ),
    }


def _load_discovery_samples(
    path: Path,
) -> tuple[tuple[CausalTransitionSample, ...], dict[str, Any]]:
    source = path.expanduser().resolve()
    payload = _load_bound_json(source, "report_hash")
    if (
        payload.get("schema_version")
        not in {
            "rosclaw.growth.causal_transition_discovery.v2",
            "rosclaw.growth.causal_transition_discovery.v3",
        }
        or payload.get("status") != "PASS_CAUSAL_TRANSITION_DISCOVERY"
    ):
        raise ValueError("causal transition source discovery is not passing")
    raw_samples = payload.get("samples")
    cases = payload.get("cases")
    if not isinstance(raw_samples, list) or not isinstance(cases, dict):
        raise ValueError("causal transition source discovery samples are malformed")
    samples: list[CausalTransitionSample] = []
    for raw in raw_samples:
        if not isinstance(raw, dict) or not isinstance(raw.get("features"), list):
            raise ValueError("causal transition source sample is malformed")
        sample = CausalTransitionSample(**{**raw, "features": tuple(raw["features"])})
        case = cases.get(sample.sample_id)
        if not isinstance(case, dict):
            raise ValueError("causal transition source sample case binding is missing")
        trajectory_path = source.parent / str(case.get("trajectory_file", ""))
        if (
            not trajectory_path.is_file()
            or hash_bytes(trajectory_path.read_bytes()) != sample.source_trajectory_hash
            or case.get("trajectory_file_hash") != sample.source_trajectory_hash
        ):
            raise ValueError("causal transition source trajectory binding changed")
        samples.append(sample)
    if len(samples) != payload.get("sample_count") or len(
        {sample.sample_id for sample in samples}
    ) != len(samples):
        raise ValueError("causal transition source discovery count changed")
    return tuple(samples), payload


def _load_lead_policy(source_dir: Path) -> tuple[DynamicLeadPassPolicy, dict[str, Any]]:
    root = source_dir.expanduser().resolve()
    evidence = _load_bound_json(root / "evidence.json", "evidence_hash")
    if evidence.get("promotion_status") != "FROZEN_RESEARCH_DEMO":
        raise ValueError("source lead-pass evidence is not frozen passing evidence")
    payload = json.loads((root / "dynamic-lead-pass-policy.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source lead-pass policy must be an object")
    payload["discovery_sample_hashes"] = tuple(payload["discovery_sample_hashes"])
    policy = DynamicLeadPassPolicy(**payload)
    if policy.artifact_hash != evidence.get("policy_hash"):
        raise ValueError("source lead-pass policy binding changed")
    return policy, evidence


def _load_bound_json(path: Path, hash_key: str) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("bound JSON artifact must be an object")
    claimed = payload.pop(hash_key, None)
    try:
        if claimed != hash_json(payload):
            raise ValueError("bound JSON artifact integrity changed")
    finally:
        if claimed is not None:
            payload[hash_key] = claimed
    return payload


def _maximum_step(position: NDArray[np.float64]) -> float:
    if position.ndim != 2 or position.shape[1] != 3 or position.shape[0] < 2:
        return math.inf
    return float(np.max(np.linalg.norm(np.diff(position, axis=0), axis=1)))


def _context_physics_signature(context: dict[str, Any]) -> str:
    required = {
        "passer_origin_m",
        "receiver_lane_m",
        "reception_target_x_m",
        "passer_ball_local_xy_m",
        "predecessor_swing_speed_scale",
        "ball_ground_friction",
    }
    if not isinstance(context, dict) or not required <= context.keys():
        raise ValueError("causal transition failure context is malformed")
    return str(hash_json({key: context[key] for key in sorted(required)}))


def _save_trajectory(path: Path, trajectory: dict[str, NDArray[Any]]) -> dict[str, str]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **trajectory)  # type: ignore[arg-type]
    os.replace(temporary, path)
    return {
        "file": path.name,
        "file_hash": hash_bytes(path.read_bytes()),
        "trajectory_digest": trajectory_digest(trajectory),
    }


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("causal transition evidence output must be new and external")
    output.mkdir(parents=True)
    return output


def _runtime_manifest() -> dict[str, Any]:
    import mujoco
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "causal_skill_transition.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _git_head(checkout: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = [
    "CausalTransitionContext",
    "CausalTransitionGrowthConfig",
    "default_transition_development_contexts",
    "default_transition_holdouts",
    "run_causal_transition_discovery",
    "run_causal_transition_retention_exam",
    "train_causal_transition_population",
    "validate_causal_transition_retention",
]
