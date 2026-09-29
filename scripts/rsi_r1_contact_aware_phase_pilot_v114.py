"""Prove causal 500Hz foot-contact feedback is noninvasive before post-touch learning."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_receiver_bridge_v71 import run

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _episode(
    asset_root: Path,
    output: Path,
    scene: dict[str, Any],
    weights: tuple[float, float, float, float, float],
) -> dict[str, Any]:
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
        navigation_profile="receive_contact",
        navigation_phase_weights=weights,
        navigation_post_weights=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    report = json.loads((output / "report.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    contact = report["receive_contact_actor"]
    context = report["receive_body_context"]
    if (
        report["report_hash"] != hash_json({k: v for k, v in report.items() if k != "report_hash"})
        or report["protocol_hash"] != hash_bytes((output / "protocol.json").read_bytes())
        or report["trace_hash"] != hash_bytes((output / "trace.npz").read_bytes())
        or protocol["navigation_profile"] != "receive_contact"
        or protocol["navigation_phase_weights"] != list(weights)
        or protocol["navigation_post_weights"] != [0.0] * 5
        or contact["post_active_frames"] != 0
        or contact["first_own_foot_time_sec"] is None
        or context["sample_count"] != 500
        or context["contract_hash"] != contact["contract_hash"]
        or context["archive_hash"] != hash_bytes((output / "receive-body-context.npz").read_bytes())
    ):
        raise ValueError("unbound contact-aware zero-action observation")
    return {
        "scene": scene["id"],
        "report_hash": report["report_hash"],
        "trace_hash": report["trace_hash"],
        "context_hash": context["archive_hash"],
        "first_own_foot_time_sec": contact["first_own_foot_time_sec"],
        "first_own_foot": contact["first_own_foot"],
        "post_active_frames": contact["post_active_frames"],
        "safe": bool(report["result"]["safe"]),
    }


def audit(asset_root: Path, protocol_path: Path, output: Path, workers: int = 2) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    scenes = [
        {"id": "s02", "ball_x_m": 2.24, "ball_y_m": -0.76, "seed": 512003},
        {"id": "s04", "ball_x_m": 2.32, "ball_y_m": -0.76, "seed": 512005},
    ]
    weights = (
        -0.1853410496026612,
        -0.2120161425477506,
        -0.10038265224958914,
        0.1304744032562346,
        0.16730409297832222,
    )
    if (
        protocol["schema"] != "rosclaw_soccer.rsi.r1_contact_aware_phase_pilot_v114.protocol.v1"
        or protocol["partition"] != "CONSUMED_CONTACT_OBSERVATION"
        or protocol["parent_selection"]
        != "/data/rosclaw_overflow/rsi-r1-retention-constrained-phase-v112/selection.json"
        or protocol["parent_selection_hash"]
        != "sha256:ee4321a90b8d02c356b17ae1cafd3053e5bdc0fd4e077ff143b9552e9a85eb73"
        or protocol["precontact_weights"] != list(weights)
        or protocol["postcontact_weights"] != [0.0] * 5
        or protocol["scenes"] != scenes
        or protocol["rollout_count"] != 2
        or not 1 <= workers <= 2
        or output.exists()
    ):
        raise ValueError("frozen contact-aware pilot required")
    selection = json.loads(Path(str(protocol["parent_selection"])).read_text())
    if (
        selection["report_hash"] != protocol["parent_selection_hash"]
        or selection["report_hash"]
        != hash_json({k: v for k, v in selection.items() if k != "report_hash"})
        or selection["selected_weights"] != list(weights)
    ):
        raise ValueError("parent physical actor selection changed")
    root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(root):
        raise ValueError("external immutable contact-aware evidence required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_contact_aware_phase_pilot_v114.py",
            "scripts/rsi_r1_receiver_bridge_v71.py",
            "src/rosclaw_soccer/rsi/team_receive_contact_phase_actor.py",
            "src/rosclaw_soccer/rsi/team_receive_contact_evidence.py",
            "src/rosclaw_soccer/skills/team/physics_evidence.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_episode, asset_root, output / str(scene["id"]), scene, weights)
            for scene in scenes
        ]
        rows = [future.result() for future in futures]
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("contact-aware source changed during physics")
    old = Path(str(protocol["parent_selection"])).parent
    selected = next(
        row
        for row in selection["generation_1"]
        if row["weights"] == list(weights) and row["two_scene_b6_passed"]
    )
    selected_index = selected["index"]
    comparisons = []
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
        parent_dir = old / f"generation-1-candidate-{selected_index:02d}" / row["scene"]
        parent_report = json.loads((parent_dir / "report.json").read_text())
        if parent_report["trace_hash"] != hash_bytes((parent_dir / "trace.npz").read_bytes()):
            raise ValueError("parent selected trace changed")
        with (
            np.load(parent_dir / "trace.npz", allow_pickle=False) as parent,
            np.load(output / row["scene"] / "trace.npz", allow_pickle=False) as observed,
        ):
            matches = {key: bool(np.array_equal(parent[key], observed[key])) for key in required}
        comparisons.append(
            {
                "scene": row["scene"],
                "array_matches": matches,
                "safety_matches": row["safe"] == parent_report["result"]["safe"],
                "noninterfering": all(matches.values())
                and row["safe"] == parent_report["result"]["safe"],
            }
        )
    result = {
        "schema": "rosclaw_soccer.rsi.r1_contact_aware_phase_pilot_v114.result.v1",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "source_hashes": sources,
        "actual_rollouts": len(rows),
        "observations": rows,
        "comparisons": comparisons,
        "status": (
            "CAUSAL_CONTACT_OBSERVATION_QUALIFIED"
            if all(row["noninterfering"] for row in comparisons)
            else "CAUSAL_CONTACT_OBSERVATION_INTERFERED"
        ),
        "post_actor_training_authorized": False,
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
