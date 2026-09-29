"""Prove a continuous read-only receiver proprioceptive stream is noninterfering."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_receiver_bridge_v71 import run

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _observe(asset_root: Path, output: Path, scene: dict[str, Any]) -> dict[str, Any]:
    run(
        asset_root,
        output,
        enabled=True,
        preview_pass=True,
        motor_agent_id="red.playmaker",
        motor_entry_frame=0,
        duration_sec=10.0,
        directed_pass_speed_mps=1.0,
        precontact_pass_standoff_m=0.35,
        handoff_profile="tracking",
        stance_lateral_m=-0.19,
        ball_x_m=scene["ball_x_m"],
        ball_y_m=scene["ball_y_m"],
        seed=scene["seed"],
        receive_teacher_tuning=(0.0, 0.18),
        navigation_profile="receive_tap",
    )
    protocol_path, report_path, trace_path = (
        output / "protocol.json",
        output / "report.json",
        output / "trace.npz",
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tap = report["receive_body_context"]
    context_path = output / "receive-body-context.npz"
    if (
        report["report_hash"] != hash_json({k: v for k, v in report.items() if k != "report_hash"})
        or report["protocol_hash"] != hash_bytes(protocol_path.read_bytes())
        or report["trace_hash"] != hash_bytes(trace_path.read_bytes())
        or protocol["navigation_profile"] != "receive_tap"
        or protocol["receive_teacher_tuning"] != [0.0, 0.18]
        or tap["archive_hash"] != hash_bytes(context_path.read_bytes())
        or tap["sample_count"] != 500
        or tap["contract_hash"] != protocol["navigation_contract_hash"]
        or protocol["activation_ceiling"] != "SIM_ONLY"
        or protocol["promotion_authorized"] is not False
    ):
        raise ValueError(f"unbound continuous body context: {output}")
    with np.load(context_path, allow_pickle=False) as context:
        if (
            context["frame"].tolist() != list(range(500))
            or context["foot_position"].shape != (500, 2, 3)
            or context["foot_velocity"].shape != (500, 2, 3)
            or context["joint_position"].shape != (500, 29)
            or context["joint_velocity"].shape != (500, 29)
            or not all(np.isfinite(context[key]).all() for key in context.files)
        ):
            raise ValueError("incomplete measured continuous receiver body context")
    return {
        "scene": scene["id"],
        "report_hash": report["report_hash"],
        "protocol_hash": report["protocol_hash"],
        "trace_hash": report["trace_hash"],
        "context_hash": tap["archive_hash"],
        "context_sample_count": tap["sample_count"],
        "safe": bool(report["result"]["safe"]),
    }


def audit(asset_root: Path, protocol_path: Path, output: Path, workers: int = 2) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    scenes = [
        {"id": "s02", "ball_x_m": 2.24, "ball_y_m": -0.76, "seed": 512003},
        {"id": "s04", "ball_x_m": 2.32, "ball_y_m": -0.76, "seed": 512005},
    ]
    if (
        protocol["schema"] != "rosclaw_soccer.rsi.r1_continuous_body_context_v110.protocol.v1"
        or protocol["partition"] != "CONSUMED_READ_ONLY_OBSERVATION"
        or protocol["scenes"] != scenes
        or protocol["rollout_count"] != 2
        or protocol["required_context_samples_per_episode"] != 500
        or protocol["parent_evidence_root"]
        != "/data/rosclaw_overflow/rsi-r1-local-b6-teacher-grid-v103"
        or protocol["parent_candidate"] != "candidate-04"
        or not 1 <= workers <= 2
        or output.exists()
    ):
        raise ValueError("frozen bounded continuous context course required")
    root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(root):
        raise ValueError("external immutable continuous context evidence required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_continuous_body_context_v110.py",
            "scripts/rsi_r1_receiver_bridge_v71.py",
            "src/rosclaw_soccer/rsi/team_receive_body_tap.py",
            "src/rosclaw_soccer/skills/team/navigation_option.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_observe, asset_root, output / str(scene["id"]), scene) for scene in scenes
        ]
        rows = [future.result() for future in futures]
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("continuous context source changed during physical rollouts")
    compared = []
    required = (
        "ball_pose",
        "ball_velocity",
        "ball_contact_agent_code",
        "ball_contact_foot_code",
        "ball_nonfoot_contact_agent_code",
        "red_finisher_joint_position",
        "red_finisher_joint_velocity",
    )
    for row in rows:
        old = Path(str(protocol["parent_evidence_root"])) / (
            str(protocol["parent_candidate"]) + "-" + row["scene"]
        )
        old_report = json.loads((old / "report.json").read_text(encoding="utf-8"))
        if old_report["report_hash"] != hash_json(
            {k: v for k, v in old_report.items() if k != "report_hash"}
        ) or old_report["trace_hash"] != hash_bytes((old / "trace.npz").read_bytes()):
            raise ValueError("parent physical trace commitment changed")
        with (
            np.load(old / "trace.npz", allow_pickle=False) as parent,
            np.load(output / row["scene"] / "trace.npz", allow_pickle=False) as observed,
        ):
            matches = {key: bool(np.array_equal(parent[key], observed[key])) for key in required}
        compared.append(
            {
                "scene": row["scene"],
                "parent_report_hash": old_report["report_hash"],
                "observed_report_hash": row["report_hash"],
                "array_matches": matches,
                "six_body_safety_matches": row["safe"] == old_report["result"]["safe"],
                "noninterfering": all(matches.values())
                and row["safe"] == old_report["result"]["safe"],
            }
        )
    passed = all(row["noninterfering"] for row in compared)
    result = {
        "schema": "rosclaw_soccer.rsi.r1_continuous_body_context_v110.result.v1",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "source_hashes": sources,
        "actual_rollouts": len(rows),
        "observations": rows,
        "comparisons": compared,
        "status": "READ_ONLY_CONTEXT_QUALIFIED" if passed else "READ_ONLY_CONTEXT_INTERFERED",
        "continuous_context_qualified": passed,
        "sequence_actor_training_authorized": False,
        "fresh_evaluation_run": False,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    result["report_hash"] = hash_json(result)
    (output / "audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    result = audit(args.asset_root, args.protocol, args.output, args.workers)
    print(json.dumps({"status": result["status"], "report_hash": result["report_hash"]}))


if __name__ == "__main__":
    main()
