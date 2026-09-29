"""Online bounded receiver phase-policy search from actual six-G1 physical outcomes."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_receiver_bridge_v71 import run

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _weights(raw: np.ndarray[Any, Any]) -> tuple[float, float, float, float, float]:
    bounded = np.clip(raw, -2.0, 2.0)
    if bounded.shape != (5,) or not np.isfinite(bounded).all():
        raise ValueError("finite five-weight physical candidate required")
    return (
        float(bounded[0]),
        float(bounded[1]),
        float(bounded[2]),
        float(bounded[3]),
        float(bounded[4]),
    )


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
        navigation_profile="receive_phase",
        navigation_phase_weights=weights,
        capture_b6_microphysics=True,
    )
    protocol_path, report_path, trace_path = (
        output / "protocol.json",
        output / "report.json",
        output / "trace.npz",
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    actor = report["receive_phase_actor"]
    context = report["receive_body_context"]
    micro = report["b6_microphysics"]
    if (
        report["report_hash"] != hash_json({k: v for k, v in report.items() if k != "report_hash"})
        or report["protocol_hash"] != hash_bytes(protocol_path.read_bytes())
        or report["trace_hash"] != hash_bytes(trace_path.read_bytes())
        or protocol["navigation_profile"] != "receive_phase"
        or protocol["navigation_phase_weights"] != list(weights)
        or actor["contract_hash"] != protocol["navigation_contract_hash"]
        or context["contract_hash"] != actor["contract_hash"]
        or context["archive_hash"] != hash_bytes((output / "receive-body-context.npz").read_bytes())
        or context["sample_count"] != 500
        or micro["observer_fault"] is not False
        or protocol["activation_ceiling"] != "SIM_ONLY"
        or protocol["promotion_authorized"] is not False
    ):
        raise ValueError(f"unbound online receive episode: {output}")
    if micro["complete"] and micro["archive_hash"] != hash_bytes(
        (output / "b6-microphysics.npz").read_bytes()
    ):
        raise ValueError("receiver contact microphysics changed")
    with np.load(trace_path, allow_pickle=False) as trace:
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
            for index in own:
                index = int(index)
                if float(trace["time"][index]) - float(trace["time"][first]) < 0.20 or np.any(
                    trace["ball_nonfoot_contact_agent_code"][first : index + 1] != 0
                ):
                    continue
                speed = float(np.linalg.norm(trace["ball_velocity"][index, :3]))
                if speed >= 2.0:
                    second.append({"frame": index, "speed_mps": speed})
        shin = np.flatnonzero(
            (trace["ball_nonfoot_contact_agent_code"] == 4)
            & (trace["ball_nonfoot_contact_geom_id"] == 240)
        )
        shin_before_foot = bool(len(shin) and (first is None or int(shin[0]) < first))
    lateral_relative = None
    if micro["complete"]:
        with np.load(output / "b6-microphysics.npz", allow_pickle=False) as measured:
            indices = np.flatnonzero(measured["own_foot_normal_force_n"] > 1.0)
            if len(indices):
                lateral_relative = float(
                    measured["counterpart_minus_ball_velocity_world_mps"][int(indices[0]), 1]
                )
    safe = bool(report["result"]["safe"])
    collision_free = int(report["result"]["robot_robot_contact_count"]) == 0
    clean = bool((report["chain"] or {}).get("clean_transfer_observed"))
    dynamic = bool(incoming is not None and incoming >= 0.30)
    qualified = bool(safe and collision_free and clean and dynamic and second)
    reward = (
        -10.0 * float(not safe or not collision_free)
        + 2.0 * float(clean)
        + 2.0 * float(dynamic and first is not None)
        + 4.0 * float(qualified)
        + 4.0 * float(report["chain_success"])
        - 3.0 * float(shin_before_foot)
        - (0.0 if lateral_relative is None else min(abs(lateral_relative), 1.0))
    )
    return {
        "scene": scene["id"],
        "report_hash": report["report_hash"],
        "protocol_hash": report["protocol_hash"],
        "trace_hash": report["trace_hash"],
        "microphysics_hash": micro["archive_hash"],
        "safe": safe,
        "robot_collision_free": collision_free,
        "clean_transfer": clean,
        "dynamic_incoming": dynamic,
        "incoming_speed_mps": incoming,
        "first_receiver_foot_frame": first,
        "qualified_second_foot": second,
        "receiver_shin_before_foot": shin_before_foot,
        "first_foot_relative_lateral_velocity_mps": lateral_relative,
        "actor_active_frames": actor["active_frames"],
        "actor_peak_delta_mps": actor["peak_delta_mps"],
        "chain_success": bool(report["chain_success"]),
        "local_b6_passed": qualified,
        "reward": reward,
    }


def _candidate(
    asset_root: Path,
    output: Path,
    scenes: list[dict[str, Any]],
    weights: tuple[float, float, float, float, float],
    index: int,
) -> dict[str, Any]:
    rows = [_episode(asset_root, output / scene["id"], scene, weights) for scene in scenes]
    return {
        "index": index,
        "weights": list(weights),
        "scenes": rows,
        "reward": sum(float(row["reward"]) for row in rows),
        "safe_pair": all(row["safe"] and row["robot_collision_free"] for row in rows),
        "two_scene_b6_passed": all(row["local_b6_passed"] for row in rows),
    }


def train(asset_root: Path, protocol_path: Path, output: Path, workers: int = 3) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    scenes = [
        {"id": "s02", "ball_x_m": 2.24, "ball_y_m": -0.76, "seed": 512003},
        {"id": "s04", "ball_x_m": 2.32, "ball_y_m": -0.76, "seed": 512005},
    ]
    if (
        protocol["schema"] != "rosclaw_soccer.rsi.r1_online_receive_phase_v111.protocol.v1"
        or protocol["partition"] != "CONSUMED_DEV_ONLINE_POLICY_SEARCH"
        or protocol["training_scenes"] != scenes
        or protocol["parent_weights"] != [0.0] * 5
        or protocol["seed"] != 111
        or protocol["generation_0_candidates"] != 8
        or protocol["generation_0_sigma"] != 0.8
        or protocol["elite_count"] != 2
        or protocol["generation_1_candidates"] != 6
        or protocol["generation_1_sigma"] != 0.35
        or protocol["weight_absolute_cap"] != 2.0
        or protocol["rollout_count"] != 28
        or not 1 <= workers <= 4
        or output.exists()
    ):
        raise ValueError("frozen bounded online receive phase course required")
    root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(root):
        raise ValueError("external immutable online learning evidence required")
    import mujoco

    from rosclaw_soccer.training.continuous_competitive_match_growth import (
        build_continuous_competitive_fixture,
    )
    from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model

    fixture = build_continuous_competitive_fixture(asset_root)
    model = build_g1_multi_player_stadium_model(
        asset_root, players=fixture.players, spec=fixture.goal
    )
    if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, 240) != "red_finisher_right_shin":
        raise ValueError("receiver shin contact identity changed")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_online_receive_phase_v111.py",
            "scripts/rsi_r1_receiver_bridge_v71.py",
            "src/rosclaw_soccer/rsi/team_receive_phase_actor.py",
            "src/rosclaw_soccer/rsi/team_receive_body_tap.py",
            "src/rosclaw_soccer/skills/team/navigation_option.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
            "src/rosclaw_soccer/world/multi_player.py",
        )
    }
    output.mkdir(parents=True)
    rng = np.random.default_rng(protocol["seed"])
    first_weights = [(0.0,) * 5] + [_weights(rng.normal(0.0, 0.8, 5)) for _ in range(7)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _candidate,
                asset_root,
                output / f"generation-0-candidate-{i:02d}",
                scenes,
                weights,
                i,
            )
            for i, weights in enumerate(first_weights)
        ]
        first = [future.result() for future in futures]
    parent = first[0]
    parent_reproduced = bool(
        parent["scenes"][0]["local_b6_passed"]
        and parent["scenes"][0]["first_receiver_foot_frame"] is not None
        and abs(parent["scenes"][0]["first_receiver_foot_frame"] - 66) <= 2
        and any(
            abs(event["frame"] - 83) <= 2 for event in parent["scenes"][0]["qualified_second_foot"]
        )
    )
    # All learning updates are downstream of authenticated physical reports.
    ranked = sorted(
        first,
        key=lambda item: (item["safe_pair"], item["reward"], -item["index"]),
        reverse=True,
    )
    safe_ranked = [row for row in ranked if row["safe_pair"]]
    elite = (safe_ranked + [parent, parent])[:2]
    center = np.mean([row["weights"] for row in elite], axis=0)
    second_weights = [_weights(center + rng.normal(0.0, 0.35, 5)) for _ in range(6)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _candidate,
                asset_root,
                output / f"generation-1-candidate-{i:02d}",
                scenes,
                weights,
                i,
            )
            for i, weights in enumerate(second_weights)
        ]
        second = [future.result() for future in futures]
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("online receiver source changed during physical learning")
    eligible = [row for row in first[1:] + second if row["two_scene_b6_passed"]]
    eligible.sort(key=lambda row: (-row["reward"], row["index"]))
    selected = eligible[0] if eligible and parent_reproduced else None
    result = {
        "schema": "rosclaw_soccer.rsi.r1_online_receive_phase_v111.result.v1",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "source_hashes": sources,
        "actual_rollouts": 2 * (len(first) + len(second)),
        "parent_reproduced": parent_reproduced,
        "generation_0": first,
        "elite_indices": [row["index"] for row in elite],
        "updated_actor_center": center.tolist(),
        "generation_1": second,
        "status": (
            "INVALID_PARENT_REPRODUCTION"
            if not parent_reproduced
            else "CONSUMED_ONLINE_PHASE_PASSED"
            if selected
            else "REJECTED_TRAINING"
        ),
        "selected_weights": None if selected is None else selected["weights"],
        "fresh_evaluation_run": False,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    result["report_hash"] = hash_json(result)
    (output / "selection.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
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
                "parent_reproduced": result["parent_reproduced"],
                "selected_weights": result["selected_weights"],
                "report_hash": result["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
