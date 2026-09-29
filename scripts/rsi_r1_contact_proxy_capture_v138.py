"""Capture paired, SIM_ONLY R1 contact windows for a fidelity-gated fast proxy."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.rsi.receiving_coordinated_feedback import ReceivingCoordinatedFeedback
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule

SCHEMA = "rosclaw_soccer.rsi.r1_contact_proxy_capture_v138.result.v1"
SNAPSHOT_FRAME = 15
LAST_FRAME = 120


@dataclass
class RecordingCoordinatedFeedback(ReceivingCoordinatedFeedback):
    """Read-only trace of the exact observation passed to the same actor."""

    observations: list[ReceivingFeedbackObservation] = field(init=False, default_factory=list)

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        action = super().propose(observation)
        self.observations.append(observation)
        return action


def capture_one(
    asset_root: Path,
    policy: Path,
    course: Any,
    weights: tuple[float, ...],
    expected_hash: str,
    output: Path,
) -> dict[str, Any]:
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A1_body29", SNAPSHOT_FRAME, 10, ((0.0,) * 29,)
    )

    def run(tapped: bool) -> tuple[Any, dict[str, Any], RecordingCoordinatedFeedback]:
        mailbox = ReceiveContactMailbox(course.agent_id)
        actor = RecordingCoordinatedFeedback(
            course.agent_id,
            schedule.contract_hash,
            mailbox,
            0.35,
            0.0,
            0.0,
            target_depth_m=0.25,
            target_lateral_m=0.12,
            coordination=weights,
        )
        result, trace = simulate_r0_receiving_course(
            asset_root=asset_root,
            reference_policy_path=policy,
            course=course,
            scenario_id=f"s199.rsi.r1.coordinated-receiving.{course.seed}",
            configuration_profile="R1_CONTACT_TAP",
            oracle=schedule,
            feedback_provider=actor,
            physics_evidence_consumers={course.agent_id: ReceiveContactEvidence(mailbox)},
            capture_oracle_authority=tapped,
            checkpoint_frame=SNAPSHOT_FRAME if tapped else 0,
        )
        return result, trace, actor

    parent_result, parent_trace, parent_actor = run(False)
    parent_hash = hash_json(parent_result.to_dict())
    if parent_hash != expected_hash:
        raise ValueError("recording wrapper failed to reproduce the sealed eight-player parent")
    result, trace, actor = run(True)
    physics_keys = (
        "ball_pose",
        "ball_velocity",
        "ball_contact_agent_code",
        "ball_contact_effector_code",
        "ball_contact_force_n",
        "ball_nonfoot_contact_agent_code",
        "ball_nonfoot_contact_force_n",
    )
    physical_equal = all(
        np.array_equal(np.asarray(parent_trace[key]), np.asarray(trace[key]))
        for key in physics_keys
    ) and all(
        np.array_equal(
            np.asarray([getattr(o, name) for o in parent_actor.observations]),
            np.asarray([getattr(o, name) for o in actor.observations]),
        )
        for name in ("qpos", "qvel")
    )
    if not physical_equal:
        raise ValueError("read-only authority capture changed ball, contact, or body physics")
    observations = actor.observations
    frames = np.asarray([o.frame for o in observations], dtype=np.int64)
    if len(frames) < LAST_FRAME - SNAPSHOT_FRAME or not np.array_equal(
        frames[: LAST_FRAME - SNAPSHOT_FRAME], np.arange(SNAPSHOT_FRAME, LAST_FRAME)
    ):
        raise ValueError("complete consecutive pre/post-contact receiving observations required")
    prefix = "receiving_authority_"
    tape_names = (
        "control_frame",
        "time_sec",
        "pd_target_rad",
        "kp",
        "kd",
        "joint_position_rad",
        "joint_velocity_radps",
        "raw_torque_nm",
        "executed_torque_nm",
        "teacher_inputs",
    )
    tape = {name: np.asarray(trace[prefix + name]) for name in tape_names}
    if (
        any(len(array) != len(tape["control_frame"]) for array in tape.values())
        or len(tape["control_frame"]) < LAST_FRAME * 10
        or not np.array_equal(
            tape["control_frame"][: LAST_FRAME * 10], np.repeat(np.arange(LAST_FRAME), 10)
        )
        or any(not np.isfinite(array).all() for array in tape.values())
    ):
        raise ValueError("complete finite 500 Hz executed motor tape required")
    arrays = {
        "initial_local_qpos": np.asarray(
            trace["receiving_authority_initial_local_qpos"][0], dtype=np.float64
        ),
        "initial_local_qvel": np.asarray(
            trace["receiving_authority_initial_local_qvel"][0], dtype=np.float64
        ),
        "observation_frame": frames,
        "observation_qpos": np.asarray([o.qpos for o in observations], dtype=np.float64),
        "observation_qvel": np.asarray([o.qvel for o in observations], dtype=np.float64),
        "ball_contact_agent_code": np.asarray(trace["ball_contact_agent_code"], dtype=np.int64),
        "ball_contact_effector_code": np.asarray(
            trace["ball_contact_effector_code"], dtype=np.int64
        ),
        "ball_contact_force_n": np.asarray(trace["ball_contact_force_n"], dtype=np.float64),
        "ball_nonfoot_contact_agent_code": np.asarray(
            trace["ball_nonfoot_contact_agent_code"], dtype=np.int64
        ),
        "ball_nonfoot_contact_force_n": np.asarray(
            trace["ball_nonfoot_contact_force_n"], dtype=np.float64
        ),
        **{"motor_" + name: value for name, value in tape.items()},
    }
    if (
        arrays["initial_local_qpos"].shape != (43,)
        or arrays["initial_local_qvel"].shape != (41,)
        or not np.isfinite(arrays["initial_local_qpos"]).all()
        or not np.isfinite(arrays["initial_local_qvel"]).all()
    ):
        raise ValueError("finite exact frame-zero receiving state required")
    np.savez_compressed(output, **arrays)  # type: ignore[arg-type]
    return {
        "course": vars(course),
        "parent_result_hash": parent_hash,
        "capture_result_hash": hash_json(result.to_dict()),
        "trace_hash": hash_bytes(output.read_bytes()),
        "observation_frames": len(frames),
        "motor_substeps": len(tape["control_frame"]),
        "parent_physics_equal": physical_equal,
    }


def capture(asset_root: Path, policy: Path, parent_report: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY proxy capture required")
    parent = json.loads(parent_report.read_text())
    if (
        parent["schema"] != "rosclaw_soccer.rsi.r1_coordinated_receiving_v136.result.v1"
        or parent["report_hash"]
        != hash_json({k: v for k, v in parent.items() if k != "report_hash"})
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
        or parent["selected"] is None
    ):
        raise ValueError("integrity-checked eight-player physical parent required")
    selected = parent["selected"]
    weights = tuple(selected["weights"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_contact_proxy_capture_v138.py",
            "src/rosclaw_soccer/rsi/receiving_coordinated_feedback.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    rows = []
    for course, trial in zip(COURSES, selected["trials"], strict=True):
        if vars(course) != trial["course"]:
            raise ValueError("parent course mismatch")
        rows.append(
            capture_one(
                asset_root,
                policy,
                course,
                weights,
                trial["result_hash"],
                output / f"course-{course.seed}.npz",
            )
        )
        print(json.dumps(rows[-1]), flush=True)
    report = {
        "schema": SCHEMA,
        "partition": "CONSUMED_BILATERAL_EIGHT_G1_DEVELOPMENT",
        "source_hashes": sources,
        "parent_report_hash": parent["report_hash"],
        "policy_hash": parent["policy_hash"],
        "snapshot_frame": SNAPSHOT_FRAME,
        "last_frame_exclusive": LAST_FRAME,
        "rows": rows,
        "status": "CAPTURE_QUALIFIED"
        if all(row["parent_physics_equal"] for row in rows)
        else "REJECTED",
        "promotion_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during proxy capture")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = capture(args.asset_root, args.policy, args.parent_report, args.output)
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
