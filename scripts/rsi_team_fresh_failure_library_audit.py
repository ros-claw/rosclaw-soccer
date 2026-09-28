"""Diagnose the failed fresh exam as consumed 3v3 development data, not a retest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.rsi.team_intercept_navigation import TeamInterceptNavigation
from rosclaw_soccer.rsi.team_phase_intercept_navigation import TeamPhaseInterceptNavigation
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
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("audit output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    prior_path = Path(protocol["failed_exam_report_path"])
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_fresh_failure_library_audit_protocol_v26"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or prior.get("schema") != "rsi_team_contextual_nav_fresh_exam_report_v25"
        or prior.get("gate_passed") is not False
        or prior.get("report_hash") != protocol["failed_exam_report_hash"]
        or hash_json({key: value for key, value in prior.items() if key != "report_hash"})
        != prior["report_hash"]
    ):
        raise ValueError("unsealed failed fresh exam required")
    fresh_protocol = json.loads(Path(protocol["failed_exam_protocol_path"]).read_text())
    if (
        hash_bytes(Path(protocol["failed_exam_protocol_path"]).read_bytes())
        != prior["protocol_hash"]
    ):
        raise ValueError("fresh course protocol changed")
    model = json.loads(Path(fresh_protocol["model_path"]).read_text())
    if model.get("model_hash") != fresh_protocol["model_hash"]:
        raise ValueError("different frozen action library")
    if args.arm not in model["arm_names"]:
        raise ValueError("arm not in failed exam's frozen training library")
    arm = model["arm_parameters"][args.arm]
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
        Path(__file__).with_name("rsi_team_contextual_nav_fresh_exam.py"),
        Path(__file__).with_name("rsi_team_taskspace_first_touch.py"),
        Path(__file__).with_name("rsi_team_intercept_navigation_search.py"),
        root / "src/rosclaw_soccer/rsi/team_intercept_navigation.py",
        root / "src/rosclaw_soccer/rsi/team_phase_intercept_navigation.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    source_hashes = {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }
    args.output_dir.mkdir(parents=True)
    rows = []
    for index, scenario in enumerate(fresh_scenarios(fresh_protocol)):
        nav_type = (
            TeamPhaseInterceptNavigation
            if arm["foot_selection"] == "phase"
            else TeamInterceptNavigation
        )
        navigation = nav_type(
            agent_id=fresh_protocol["focal_agent_id"],
            foundation_hash=policy_hash,
            foundation_config_hash=config_hash,
            forward_gain=float(arm["forward_gain"]),
            lateral_gain=float(arm["lateral_gain"]),
        )
        folder = args.output_dir / f"f{index:02d}"
        candidate = _run_one(
            mode="candidate",
            asset_root=args.asset_root,
            output_dir=folder,
            fixture=fixture,
            scenario=scenario,
            protocol=fresh_protocol,
            navigation_policy=navigation,
        )
        outcome = _score(candidate, folder / "candidate/trajectory.npz")
        entry = entry_features(folder / "candidate/taskspace_trace.npz", 30)
        if entry["hash"] != prior["episodes"][index]["entry_hash"]:
            raise ValueError("action-library audit changed the fresh entry state")
        outcome.update({"scenario_hash": scenario.scenario_hash, "entry_hash": entry["hash"]})
        rows.append(outcome)
        print(
            f"{args.arm} f{index:02d} safe={outcome['safe']} useful={outcome['useful_pass']}",
            flush=True,
        )
    if source_hashes != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("source changed during consumed physical audit")
    report = {
        "schema": "rsi_team_fresh_failure_library_audit_arm_v26",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "failed_fresh_report_hash": prior["report_hash"],
        "source_hashes": source_hashes,
        "asset_body_hash": qualification.body_hash,
        "arm": arm,
        "rows": rows,
        "safe_contact_count": sum(row["safe"] and row["foot_contact_frames"] for row in rows),
        "useful_pass_count": sum(row["useful_pass"] for row in rows),
        "unsafe_count": sum(not row["safe"] for row in rows),
        "fresh_holdout": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_FAILED_FRESH_LIBRARY="
        + json.dumps(
            {
                key: report[key]
                for key in (
                    "arm",
                    "safe_contact_count",
                    "useful_pass_count",
                    "unsafe_count",
                    "report_hash",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
