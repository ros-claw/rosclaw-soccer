"""Check one frozen physical-search interceptor on held-back consumed 3v3 scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.rsi.team_adaptive_intercept_navigation import TeamAdaptiveInterceptNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture
from scripts.rsi_team_contextual_action_dataset import entry_features
from scripts.rsi_team_contextual_nav_fresh_exam import fresh_scenarios
from scripts.rsi_team_intercept_navigation_search import _score
from scripts.rsi_team_taskspace_first_touch import _run_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("validation output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    model_path = Path(protocol["model_path"])
    model = json.loads(model_path.read_text(encoding="utf-8"))
    prior = json.loads(Path(protocol["failed_exam_report_path"]).read_text())
    if (
        protocol.get("schema") != "rsi_team_adaptive_intercept_validation_protocol_v28"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("validation_scene_indices") != [0, 1, 4, 5, 7, 9]
        or protocol.get("internal_gate") != {"minimum_safe": 6, "minimum_useful_pass": 3}
        or model.get("model_hash") != protocol["model_hash"]
        or hash_json({key: value for key, value in model.items() if key != "model_hash"})
        != model["model_hash"]
        or model.get("chosen_candidate_index") != 19
        or prior.get("report_hash") != protocol["failed_exam_report_hash"]
    ):
        raise ValueError("uncommitted SIM_ONLY validation protocol")
    fresh_protocol = json.loads(Path(protocol["failed_exam_protocol_path"]).read_text())
    qualification = qualify_g1_assets(args.asset_root)
    qualification.require_eligible()
    if qualification.body_hash != model["asset_body_hash"]:
        raise ValueError("trained interceptor body mismatch")
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
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    source_hashes = {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }
    args.output_dir.mkdir(parents=True)
    rows = []
    scenarios = fresh_scenarios(fresh_protocol)
    for index in protocol["validation_scene_indices"]:
        scenario = scenarios[index]
        navigation = TeamAdaptiveInterceptNavigation(
            agent_id=fresh_protocol["focal_agent_id"],
            foundation_hash=policy_hash,
            foundation_config_hash=config_hash,
            **model["parameters"],
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
                }
            )
            print(f"f{index:02d} INCOMPLETE", flush=True)
            continue
        trace_path = folder / "candidate/taskspace_trace.npz"
        entry = entry_features(trace_path, 30)
        if entry["hash"] != prior["episodes"][index]["entry_hash"]:
            raise ValueError("validation candidate changed paired entry state")
        outcome = _score(report, folder / "candidate/trajectory.npz")
        with np.load(trace_path, allow_pickle=False) as trace:
            feet = np.asarray(trace["pre_step_foot_link_position_w"])[:, 0]
            ball = np.asarray(trace["pre_step_ball_position_local_m"])[:, 0]
        outcome.update(
            {
                "scene": f"f{index:02d}",
                "status": "COMPLETE",
                "minimum_foot_ball_gap_m": float(
                    np.min(np.linalg.norm(feet - ball[:, None, :], axis=2))
                ),
            }
        )
        rows.append(outcome)
        print(f"f{index:02d} safe={outcome['safe']} useful={outcome['useful_pass']}", flush=True)
    if source_hashes != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("physical validation source changed")
    safe = sum(bool(row["safe"]) for row in rows)
    useful = sum(bool(row["useful_pass"]) for row in rows)
    report = {
        "schema": "rsi_team_adaptive_intercept_validation_report_v28",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "model_hash": model["model_hash"],
        "source_hashes": source_hashes,
        "asset_body_hash": qualification.body_hash,
        "rows": rows,
        "safe_count": safe,
        "safe_contact_count": sum(bool(row["safe"] and row["foot_contact_frames"]) for row in rows),
        "useful_pass_count": useful,
        "internal_gate_passed": safe >= 6 and useful >= 3,
        "fresh_holdout": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_ADAPTIVE_VALIDATION="
        + json.dumps(
            {
                key: report[key]
                for key in (
                    "report_hash",
                    "safe_count",
                    "safe_contact_count",
                    "useful_pass_count",
                    "internal_gate_passed",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
