"""SIM_ONLY bounded foot-phase feedback on consumed router failures."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_measured_skill_router_v207 import FRESH_COURSES, _score
from rsi_r1_middle_basis_cem_v187 import PHYSICAL_KEYS

from rosclaw_soccer.rsi.receiving_foot_phase_router import (
    ReceivingFootPhaseReference,
    ReceivingFootPhaseRouter,
)
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.rsi.team_receive_side_navigation import TeamReceiveSideNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window

SCHEMA = "rosclaw_soccer.rsi.r1_foot_phase_development_v209.result.v1"
GAINS = (0.0, 0.3, 0.6, 1.0)


def _run_task(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        asset_root,
        policy_path,
        course,
        coordination,
        left,
        right,
        slope,
        knots,
        references,
        gain,
        *extra,
    ) = task
    post_multiplier = float(extra[0]) if extra else 1.0
    shin_guard_m = float(extra[1]) if len(extra) > 1 else 0.0
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingFootPhaseRouter(
        course.agent_id,
        schedule.contract_hash,
        mailbox,
        coordination,
        left,
        right,
        left_post_gain=0.0,
        right_post_gain=0.4,
        old_lateral_slope=slope,
        far_lateral_slope=(0.0,) * 12,
        skill_knots=knots,
        references=references,
        foot_gain=gain,
        post_multiplier=post_multiplier,
        shin_guard_m=shin_guard_m,
    )
    navigation = TeamReceiveSideNavigation(
        course.agent_id,
        hash_bytes((asset_root / "policy/loco_mode/model/policy_29dof.pt").read_bytes()),
        hash_bytes((asset_root / "policy/loco_mode/config/LocoMode.yaml").read_bytes()),
        mailbox,
        (0.0, 0.0, 0.5, 0.0, 0.0),
        (0.0,) * 5,
    )
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy_path,
        course=course,
        scenario_id=f"s199.rsi.r1.coordinated-receiving.{course.seed}",
        configuration_profile="R1_CONTACT_TAP",
        oracle=schedule,
        feedback_provider=feedback,
        research_navigation_policy=navigation,
        research_contact_leg_stiffness_scale=0.4,
        research_contact_distance_threshold_m=0.24,
        research_contact_max_active_substeps=32,
        physics_evidence_consumers={course.agent_id: ReceiveContactEvidence(mailbox)},
    )
    info = result.to_dict()
    ids = tuple(sorted(row["agent_id"] for row in info["qualities"]))
    n = len(trace["time"])
    if n >= 120:
        _, outcome = receiving_window(
            trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
        )
        detail = explain_receiving_window(
            trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
        )
        distance = detail["tail_maximum_foot_distance_m"]
        speed = detail["tail_maximum_ball_speed_mps"]
    else:
        outcome = {"controlled_reception": False}
        distance = 2.0
        speed = 3.0
    code = ids.index(course.agent_id) + 1
    foot = np.asarray(trace["ball_contact_agent_code"])
    foot_force = np.asarray(trace["ball_contact_force_n"])
    nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
    nonfoot_force = np.asarray(trace["ball_nonfoot_contact_force_n"])
    active = np.asarray(trace["receiving_impedance_active_substeps"], dtype=np.int64)
    return {
        "course": vars(course),
        "gain": gain,
        "post_multiplier": post_multiplier,
        "shin_guard_m": shin_guard_m,
        "selected_expert": feedback.selected_expert,
        "selected_reference": feedback.selected_reference,
        "nonzero_frames": feedback.nonzero_frames,
        "peak_correction_rad": feedback.peak_correction_rad,
        "frames_recorded": n,
        "safe": info["safe"] and n >= 120,
        "fault_agents": info["physics_evidence_fault_agents"],
        "first_foot_frame": next(
            (
                frame
                for frame in range(20, min(n, 120))
                if foot[frame] == code and foot_force[frame] > 0
            ),
            None,
        ),
        "own_nonfoot_frames": [
            frame
            for frame in range(20, min(n, 120))
            if nonfoot[frame] == code and nonfoot_force[frame] > 0
        ],
        "active_substeps": int(active.sum()),
        "controlled_reception": outcome["controlled_reception"],
        "tail_maximum_foot_distance_m": distance,
        "tail_maximum_ball_speed_mps": speed,
        "physical_trace_hash": hash_json(
            {key: np.asarray(trace[key]).tolist() for key in PHYSICAL_KEYS}
        ),
    }


def develop(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    router_report_path: Path,
    teacher_dir: Path,
    map_report_path: Path,
    foot_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY development evidence required")
    parent, right_parent, refine, lateral, router, teachers, mapping, foot_trace = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            router_report_path,
            teacher_dir / "report.json",
            map_report_path,
            foot_report_path,
        )
    )
    if (
        router["status"] != "REJECTED_MEASURED_SKILL_ROUTER_GATE"
        or foot_trace["physical_equal_count"] != 8
        or foot_trace["router_report_hash"] != router["report_hash"]
        or teachers["physical_equal_count"] != 7
        or teachers["map_report_hash"] != mapping["report_hash"]
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed failed fresh and physical teacher lineage required")
    knots = tuple(
        ReceivingSkillKnot(
            float(row["lateral_m"]),
            float(row["closing_speed_mps"]),
            row["expert"],
            tuple(float(value) for value in row["weights"]),
        )
        for row in router["skill_knots"]
    )
    references = []
    for index, row in enumerate(teachers["rows"]):
        teacher_path = teacher_dir / f"teacher-{index}.npz"
        if hash_bytes(teacher_path.read_bytes()) != row["feature_hash"]:
            raise ValueError("sealed teacher feature hash required")
        with np.load(teacher_path, allow_pickle=False) as arrays:
            features = np.asarray(arrays["features"], dtype=np.float64)
        historical = next(
            item["summary"]
            for item in mapping["rows"]
            if item["expert"] == row["expert"]
            and item["summary"]["course"]["seed"] == row["course"]["seed"]
        )
        references.append(
            ReceivingFootPhaseReference(
                row["expert"],
                float(features[0, 1] * 0.2),
                float(features[0, 3] * 2.0),
                int(historical["first_foot_frame"]),
                tuple(tuple(float(value) for value in frame) for frame in features),
            )
        )
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7)
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_foot_phase_development_v209.py",
            "src/rosclaw_soccer/rsi/receiving_foot_phase_router.py",
            "src/rosclaw_soccer/rsi/receiving_measured_skill_router.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    tasks = [
        (
            asset_root,
            policy_path,
            course,
            coordination,
            left,
            right,
            slope,
            knots,
            tuple(references),
            gain,
        )
        for gain in GAINS
        for course in FRESH_COURSES
    ]
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        rows = list(pool.map(_run_task, tasks))
    blocks = [rows[i : i + 8] for i in range(0, len(rows), 8)]
    baseline_equal = all(
        row["physical_trace_hash"] == old["physical_trace_hash"]
        and row["selected_expert"] == old["selected_expert"]
        and row["nonzero_frames"] == 0
        for row, old in zip(blocks[0], router["fresh_candidate"], strict=True)
    )
    if not baseline_equal:
        raise ValueError("zero-gain foot feedback must preserve every measured router trajectory")
    scores = [_score(block) for block in blocks]
    for gain, score in zip(GAINS, scores, strict=True):
        print(json.dumps({"gain": gain, "strict": score[0], "clean": score[1]}), flush=True)
    qualified = [
        (gain, block, score)
        for gain, block, score in zip(GAINS[1:], blocks[1:], scores[1:], strict=True)
        if score[0] >= 3
        and score[1] >= 7
        and all(row["safe"] and row["peak_correction_rad"] <= 0.0400001 for row in block)
    ]
    report = {
        "schema": SCHEMA,
        "router_report_hash": router["report_hash"],
        "teacher_report_hash": teachers["report_hash"],
        "foot_trace_report_hash": foot_trace["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_V207_FRESH_NOW_DEVELOPMENT_NOT_NEW_EXAM",
        "gains": GAINS,
        "rows": blocks,
        "scores": scores,
        "zero_gain_physical_equal": baseline_equal,
        "qualified_gains": [item[0] for item in qualified],
        "status": "DEVELOPMENT_FOOT_PHASE_CANDIDATE_ONLY"
        if qualified
        else "REJECTED_FOOT_PHASE_DEVELOPMENT_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during foot phase development")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "asset-root",
        "policy",
        "parent-report",
        "right-report",
        "refine-report",
        "lateral-report",
        "router-report",
        "teacher-dir",
        "map-report",
        "foot-report",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = develop(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.router_report,
        args.teacher_dir,
        args.map_report,
        args.foot_report,
        args.output,
    )
    print(json.dumps({key: report[key] for key in ("status", "scores", "report_hash")}))


if __name__ == "__main__":
    main()
