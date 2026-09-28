"""Physical-search training of bounded G1 interception on consumed failure fronts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.rsi.team_adaptive_intercept_navigation import TeamAdaptiveInterceptNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture
from scripts.rsi_team_contextual_action_dataset import entry_features
from scripts.rsi_team_contextual_nav_fresh_exam import fresh_scenarios
from scripts.rsi_team_intercept_navigation_search import _score
from scripts.rsi_team_taskspace_first_touch import _run_one


def candidate_parameters(seed: int, count: int) -> list[dict[str, float]]:
    generator = np.random.default_rng(seed)
    candidates = [
        {
            "forward_gain": 0.4,
            "lateral_gain": 1.2,
            "target_gap_m": 0.55,
            "activation_max_gap_m": 1.4,
            "speed_cap_mps": 0.18,
        }
    ]
    while len(candidates) < count:
        values = {
            "forward_gain": float(generator.choice((-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8))),
            "lateral_gain": float(
                generator.choice((-1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6))
            ),
            "target_gap_m": float(generator.choice((0.3, 0.4, 0.5, 0.6, 0.7, 0.8))),
            "activation_max_gap_m": float(generator.choice((1.2, 1.5, 1.8, 2.0))),
            "speed_cap_mps": float(generator.choice((0.18, 0.22, 0.25))),
        }
        if values not in candidates:
            candidates.append(values)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-index", required=True, type=int)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("candidate output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    prior = json.loads(Path(protocol["failed_exam_report_path"]).read_text())
    if (
        protocol.get("schema") != "rsi_team_adaptive_intercept_search_protocol_v28"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("candidate_count") != 24
        or protocol.get("training_scene_indices") != [2, 3, 6, 8, 10, 11]
        or not 0 <= args.candidate_index < 24
        or prior.get("report_hash") != protocol["failed_exam_report_hash"]
        or hash_json({key: value for key, value in prior.items() if key != "report_hash"})
        != prior["report_hash"]
    ):
        raise ValueError("uncommitted SIM_ONLY physical-search protocol")
    fresh_protocol = json.loads(Path(protocol["failed_exam_protocol_path"]).read_text())
    parameters = candidate_parameters(protocol["parameter_seed"], 24)[args.candidate_index]
    qualification = qualify_g1_assets(args.asset_root)
    qualification.require_eligible()
    fixture = build_independent_three_vs_three_fixture(args.asset_root)
    policy_hash = hash_bytes(
        (args.asset_root / "policy/loco_mode/model/policy_29dof.pt").read_bytes()
    )
    config_hash = hash_bytes(
        (args.asset_root / "policy/loco_mode/config/LocoMode.yaml").read_bytes()
    )
    root = Path(__file__).parents[1]
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("rsi_team_taskspace_first_touch.py"),
        root / "src/rosclaw_soccer/rsi/team_adaptive_intercept_navigation.py",
        root / "src/rosclaw_soccer/rsi/taskspace_swing_probe.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    source_hashes = {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }
    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    scenarios = fresh_scenarios(fresh_protocol)
    for index in protocol["training_scene_indices"]:
        scenario = scenarios[index]
        navigation = TeamAdaptiveInterceptNavigation(
            agent_id=fresh_protocol["focal_agent_id"],
            foundation_hash=policy_hash,
            foundation_config_hash=config_hash,
            **parameters,
        )
        folder = args.output_dir / f"f{index:02d}"
        try:
            report = _run_one(
                mode="candidate",
                asset_root=args.asset_root,
                output_dir=folder,
                fixture=fixture,
                scenario=scenario,
                protocol=fresh_protocol,
                navigation_policy=navigation,
            )
        except ValueError as error:
            if str(error) != "incomplete paired motor episode":
                raise
            rows.append(
                {
                    "scene": f"f{index:02d}",
                    "status": "INCOMPLETE",
                    "safe": False,
                    "useful_pass": False,
                    "foot_contact_frames": [],
                    "minimum_foot_ball_gap_m": None,
                }
            )
            print(f"c{args.candidate_index:02d} f{index:02d} INCOMPLETE", flush=True)
            continue
        trace_path = folder / "candidate/taskspace_trace.npz"
        entry = entry_features(trace_path, 30)
        if entry["hash"] != prior["episodes"][index]["entry_hash"]:
            raise ValueError("interception changed pre-intervention physical state")
        outcome = _score(report, folder / "candidate/trajectory.npz")
        with np.load(trace_path, allow_pickle=False) as trace:
            feet = np.asarray(trace["pre_step_foot_link_position_w"])[:, 0]
            ball = np.asarray(trace["pre_step_ball_position_local_m"])[:, 0]
        minimum_gap = float(np.min(np.linalg.norm(feet - ball[:, None, :], axis=2)))
        outcome.update(
            {"scene": f"f{index:02d}", "status": "COMPLETE", "minimum_foot_ball_gap_m": minimum_gap}
        )
        rows.append(outcome)
        print(
            f"c{args.candidate_index:02d} f{index:02d} "
            f"safe={outcome['safe']} useful={outcome['useful_pass']} gap={minimum_gap:.3f}",
            flush=True,
        )
    if source_hashes != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("source changed during physical interception training")
    result = {
        "schema": "rsi_team_adaptive_intercept_candidate_v28",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "source_hashes": source_hashes,
        "asset_body_hash": qualification.body_hash,
        "candidate_index": args.candidate_index,
        "parameters": parameters,
        "rows": rows,
        "all_safe": all(row["safe"] for row in rows),
        "safe_contact_count": sum(bool(row["safe"] and row["foot_contact_frames"]) for row in rows),
        "useful_pass_count": sum(bool(row["useful_pass"]) for row in rows),
        "incomplete_count": sum(row["status"] != "COMPLETE" for row in rows),
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_ADAPTIVE_INTERCEPT="
        + json.dumps(
            {
                key: result[key]
                for key in (
                    "candidate_index",
                    "report_hash",
                    "all_safe",
                    "safe_contact_count",
                    "useful_pass_count",
                    "incomplete_count",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
