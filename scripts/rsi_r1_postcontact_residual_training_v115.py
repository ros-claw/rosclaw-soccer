"""Bounded physical postcontact learning with hard old-skill retention."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, cast

import numpy as np
from rsi_r1_receiver_bridge_v71 import run

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _episode(
    asset_root: Path,
    output: Path,
    scene: dict[str, Any],
    pre: tuple[float, ...],
    post: tuple[float, ...],
) -> dict[str, Any]:
    if (output / "report.json").exists():
        report = json.loads((output / "report.json").read_text())
        episode_protocol = json.loads((output / "protocol.json").read_text())
        root = Path(__file__).resolve().parents[1]
        if any(
            hash_bytes((root / name).read_bytes()) != digest
            for name, digest in episode_protocol["source_hashes"].items()
        ):
            raise ValueError("stored physical rollout source changed")
    else:
        report = run(
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
            navigation_phase_weights=cast(tuple[float, float, float, float, float], pre),
            navigation_post_weights=cast(tuple[float, float, float, float, float], post),
        )
        episode_protocol = json.loads((output / "protocol.json").read_text())
    if (
        report["report_hash"]
        != hash_json({key: value for key, value in report.items() if key != "report_hash"})
        or report["trace_hash"] != hash_bytes((output / "trace.npz").read_bytes())
        or report["protocol_hash"] != hash_bytes((output / "protocol.json").read_bytes())
        or episode_protocol["navigation_phase_weights"] != list(pre)
        or episode_protocol["navigation_post_weights"] != list(post)
        or episode_protocol["scenario"]["ball_initial_position_m"][:2]
        != [scene["ball_x_m"], scene["ball_y_m"]]
        or episode_protocol["scenario"]["seed"] != scene["seed"]
        or report["result"]["navigation_fault_agents"]
        or report["result"]["physics_evidence_fault_agents"]
    ):
        raise ValueError("contact policy rollout has invalid integrity or lost evidence")
    with np.load(output / "trace.npz", allow_pickle=False) as trace:
        own = np.flatnonzero(
            (trace["ball_contact_agent_code"] == 4)
            & np.isin(trace["ball_contact_foot_code"], (1, 2))
        )
        first = int(own[0]) if len(own) else None
        incoming = (
            None
            if first is None or first == 0
            else float(np.linalg.norm(trace["ball_velocity"][first - 1, :2]))
        )
        second = []
        if first is not None:
            for frame in own:
                frame = int(frame)
                if float(trace["time"][frame]) - float(trace["time"][first]) < 0.20:
                    continue
                if np.any(trace["ball_nonfoot_contact_agent_code"][first : frame + 1] != 0):
                    continue
                speed = float(np.linalg.norm(trace["ball_velocity"][frame, :3]))
                if speed >= 2.0:
                    second.append({"frame": frame, "speed_mps": speed})
        shin = np.flatnonzero(
            (trace["ball_nonfoot_contact_agent_code"] == 4)
            & (trace["ball_nonfoot_contact_geom_id"] == 240)
        )
        shin_before_foot = bool(len(shin) and (first is None or int(shin[0]) < first))
    safe = bool(report["result"]["safe"])
    collision_free = int(report["result"]["robot_robot_contact_count"]) == 0
    clean = bool((report["chain"] or {}).get("clean_transfer_observed"))
    dynamic = bool(incoming is not None and incoming >= 0.30)
    qualified = bool(safe and collision_free and clean and dynamic and second)
    return {
        "scene": scene["id"],
        "report_hash": report["report_hash"],
        "trace_hash": report["trace_hash"],
        "safe": safe,
        "collision_free": collision_free,
        "clean_dynamic": clean and dynamic,
        "first_foot_frame": first,
        "second_foot": second[:3],
        "shin_before_foot": shin_before_foot,
        "post_active_frames": report["receive_contact_actor"]["post_active_frames"],
        "qualified": qualified,
    }


def _candidate(
    asset_root: Path,
    output: Path,
    scenes: list[dict[str, Any]],
    pre: tuple[float, ...],
    post: tuple[float, ...],
    index: int,
) -> dict[str, Any]:
    rows = [_episode(asset_root, output / scene["id"], scene, pre, post) for scene in scenes]
    return {
        "index": index,
        "post_weights": list(post),
        "scenes": rows,
        "anchor_retained": all(row["qualified"] for row in rows[:2]),
        "development_qualified_count": sum(row["qualified"] for row in rows[2:]),
        "all_safe": all(row["safe"] and row["collision_free"] for row in rows),
    }


def train(asset_root: Path, protocol_path: Path, output: Path, workers: int) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    parent_path = Path(str(protocol["parent_observation_audit"]))
    parent = json.loads(parent_path.read_text())
    if (
        protocol["schema"] != "rosclaw_soccer.rsi.r1_postcontact_residual_training_v115.protocol.v1"
        or protocol["partition"] != "CONSUMED_DEV_POSTCONTACT_SEARCH"
        or parent["report_hash"] != protocol["parent_observation_hash"]
        or parent["report_hash"]
        != hash_json({key: value for key, value in parent.items() if key != "report_hash"})
        or parent["status"] != "CAUSAL_CONTACT_OBSERVATION_QUALIFIED"
        or protocol["candidate_count"] != 9
        or protocol["rollout_count"] != 36
        or protocol["seed"] != 115
        or len(protocol["training_scenes"]) != 4
        or not 1 <= workers <= 4
        or (output / "selection-r2.json").exists()
    ):
        raise ValueError("frozen contact-aware development training required")
    root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(root):
        raise ValueError("external evidence output required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_postcontact_residual_training_v115.py",
            "scripts/rsi_r1_receiver_bridge_v71.py",
            "src/rosclaw_soccer/rsi/team_receive_contact_phase_actor.py",
            "src/rosclaw_soccer/rsi/team_receive_contact_evidence.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    pre = tuple(float(value) for value in protocol["precontact_weights"])
    rng = np.random.default_rng(115)
    candidates: list[tuple[float, ...]] = [
        (0.0,) * 5,
        (0.0, 0.0, 0.0, 0.7, 0.0),
        (0.0, 0.0, 0.0, -0.7, 0.0),
        (0.0, 0.5, 0.0, 0.0, 0.0),
        (0.0, -0.5, 0.0, 0.0, 0.0),
    ]
    candidates.extend(tuple(float(v) for v in rng.normal(0.0, 0.35, 5)) for _ in range(4))
    if output.exists() and len(list(output.glob("candidate-*/**/report.json"))) != 36:
        raise ValueError("partial physical course cannot be silently resumed")
    output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _candidate,
                asset_root,
                output / f"candidate-{index:02d}",
                protocol["training_scenes"],
                pre,
                post,
                index,
            )
            for index, post in enumerate(candidates)
        ]
        rows = [future.result() for future in futures]
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("postcontact learning source drift")
    zero_parent = rows[0]
    eligible = [row for row in rows[1:] if row["anchor_retained"] and row["all_safe"]]
    eligible.sort(key=lambda row: (-row["development_qualified_count"], row["index"]))
    winner = (
        eligible[0]
        if zero_parent["anchor_retained"]
        and eligible
        and eligible[0]["development_qualified_count"] > zero_parent["development_qualified_count"]
        else None
    )
    prior = output / "selection.json"
    prior_hash = hash_bytes(prior.read_bytes()) if prior.exists() else None
    result = {
        "schema": "rosclaw_soccer.rsi.r1_postcontact_residual_training_v115.result.v1",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "parent_observation_hash": parent["report_hash"],
        "source_hashes": sources,
        "actual_rollouts": len(rows) * len(protocol["training_scenes"]),
        "candidates": rows,
        "zero_parent_retained": zero_parent["anchor_retained"],
        "zero_parent_development_qualified_count": zero_parent["development_qualified_count"],
        "superseded_selection_file_hash": prior_hash,
        "selected_post_weights": None if winner is None else winner["post_weights"],
        "status": "CONSUMED_DEV_IMPROVEMENT" if winner else "REJECTED_NO_DEV_GAIN",
        "fresh_evaluation_run": False,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    result["report_hash"] = hash_json(result)
    (output / "selection-r2.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    result = train(args.asset_root, args.protocol, args.output, args.workers)
    print(
        json.dumps(
            {
                "status": result["status"],
                "zero_parent_retained": result["zero_parent_retained"],
                "selected_post_weights": result["selected_post_weights"],
                "report_hash": result["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
