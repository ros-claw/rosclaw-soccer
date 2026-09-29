"""Physical constrained mixture search over measured-state receive gate."""

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
    retained: tuple[float, ...],
    alternative: tuple[float, ...],
    gate: tuple[float, ...],
) -> dict[str, Any]:
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
        navigation_profile="receive_mixture",
        navigation_phase_weights=cast(tuple[float, float, float, float, float], retained),
        navigation_alternative_weights=cast(tuple[float, float, float, float, float], alternative),
        navigation_gate_weights=cast(tuple[float, float, float, float], gate),
    )
    protocol = json.loads((output / "protocol.json").read_text())
    actor = report["receive_mixture_actor"]
    if (
        report["report_hash"]
        != hash_json({key: value for key, value in report.items() if key != "report_hash"})
        or report["trace_hash"] != hash_bytes((output / "trace.npz").read_bytes())
        or report["protocol_hash"] != hash_bytes((output / "protocol.json").read_bytes())
        or protocol["navigation_phase_weights"] != list(retained)
        or protocol["navigation_alternative_weights"] != list(alternative)
        or protocol["navigation_gate_weights"] != list(gate)
        or actor["contract_hash"] != protocol["navigation_contract_hash"]
        or report["result"]["navigation_fault_agents"]
        or report["result"]["physics_evidence_fault_agents"]
    ):
        raise ValueError("invalid measured-state mixture physical episode")
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
        shin_first = bool(len(shin) and (first is None or int(shin[0]) <= first))
    safe = bool(report["result"]["safe"])
    collision_free = int(report["result"]["robot_robot_contact_count"]) == 0
    clean = bool((report["chain"] or {}).get("clean_transfer_observed"))
    dynamic = bool(incoming is not None and incoming >= 0.30)
    clean_first = bool(
        safe and collision_free and clean and dynamic and first is not None and not shin_first
    )
    return {
        "scene": scene["id"],
        "report_hash": report["report_hash"],
        "trace_hash": report["trace_hash"],
        "safe": safe,
        "collision_free": collision_free,
        "clean_dynamic_first": clean_first,
        "qualified_second": bool(clean_first and second),
        "first_foot_frame": first,
        "second_foot": second[:3],
        "shin_first_or_simultaneous": shin_first,
        "alternative_active_frames": actor["alternative_active_frames"],
        "maximum_gate": actor["maximum_gate"],
    }


def _candidate(
    asset_root: Path,
    output: Path,
    scenes: list[dict[str, Any]],
    retained: tuple[float, ...],
    alternative: tuple[float, ...],
    gate: tuple[float, ...],
    index: int,
) -> dict[str, Any]:
    rows = [
        _episode(asset_root, output / scene["id"], scene, retained, alternative, gate)
        for scene in scenes
    ]
    return {
        "index": index,
        "alternative_weights": list(alternative),
        "gate_weights": list(gate),
        "scenes": rows,
        "retained": all(row["qualified_second"] for row in rows[:2]),
        "development_first_count": sum(row["clean_dynamic_first"] for row in rows[2:]),
        "development_second_count": sum(row["qualified_second"] for row in rows[2:]),
        "development_shin_count": sum(row["shin_first_or_simultaneous"] for row in rows[2:]),
        "all_safe": all(row["safe"] and row["collision_free"] for row in rows),
    }


def train(asset_root: Path, protocol_path: Path, output: Path, workers: int) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    parent_path = Path(str(protocol["parent_precontact_report"]))
    parent = json.loads(parent_path.read_text())
    if (
        protocol["schema"] != "rosclaw_soccer.rsi.r1_gated_receive_mixture_v117.protocol.v1"
        or protocol["partition"] != "CONSUMED_DEV_STATE_GATED_MIXTURE"
        or parent["report_hash"] != protocol["parent_precontact_hash"]
        or parent["report_hash"]
        != hash_json({key: value for key, value in parent.items() if key != "report_hash"})
        or parent["status"] != "REJECTED_NO_DEV_GAIN"
        or protocol["candidate_count"] != 13
        or protocol["rollout_count"] != 52
        or len(protocol["training_scenes"]) != 4
        or protocol["alternative_candidate_indices"] != [1, 11]
        or protocol["gate_biases"] != [-2.0, -4.0]
        or protocol["gate_slopes"] != [4.0, 8.0, 12.0]
        or not 1 <= workers <= 4
        or output.exists()
    ):
        raise ValueError("frozen measured-state mixture course required")
    root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(root):
        raise ValueError("external evidence output required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_gated_receive_mixture_v117.py",
            "scripts/rsi_r1_receiver_bridge_v71.py",
            "src/rosclaw_soccer/rsi/team_receive_gated_mixture_actor.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    retained = tuple(float(v) for v in protocol["retained_weights"])
    alternatives = [tuple(parent["candidates"][i]["weights"]) for i in (1, 11)]
    candidates = [(alternatives[0], (-12.0, 0.0, 0.0, 0.0))]
    candidates.extend(
        (alternative, (bias, slope, 0.0, 0.0))
        for alternative in alternatives
        for bias in protocol["gate_biases"]
        for slope in protocol["gate_slopes"]
    )
    output.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _candidate,
                asset_root,
                output / f"candidate-{index:02d}",
                protocol["training_scenes"],
                retained,
                alternative,
                gate,
                index,
            )
            for index, (alternative, gate) in enumerate(candidates)
        ]
        rows = [future.result() for future in futures]
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("mixture source changed during physics")
    baseline = rows[0]
    eligible = [
        row
        for row in rows[1:]
        if row["retained"]
        and row["all_safe"]
        and row["development_shin_count"] <= baseline["development_shin_count"]
        and (
            row["development_second_count"] > baseline["development_second_count"]
            or row["development_first_count"] > baseline["development_first_count"]
        )
    ]
    eligible.sort(
        key=lambda row: (
            -row["development_second_count"],
            -row["development_first_count"],
            row["development_shin_count"],
            row["index"],
        )
    )
    winner = eligible[0] if baseline["retained"] and eligible else None
    result = {
        "schema": "rosclaw_soccer.rsi.r1_gated_receive_mixture_v117.result.v1",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "parent_precontact_hash": parent["report_hash"],
        "source_hashes": sources,
        "actual_rollouts": len(rows) * len(protocol["training_scenes"]),
        "candidates": rows,
        "baseline_retained": baseline["retained"],
        "selected_index": None if winner is None else winner["index"],
        "status": "CONSUMED_DEV_IMPROVEMENT" if winner else "REJECTED_NO_DEV_GAIN",
        "fresh_evaluation_run": False,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    result["report_hash"] = hash_json(result)
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = train(args.asset_root, args.protocol, args.output, args.workers)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_index": result["selected_index"],
                "report_hash": result["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
