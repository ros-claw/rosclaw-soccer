"""SIM_ONLY motor-action expansion on the v21 development scenes it could not solve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.rsi.team_phase_intercept_navigation import TeamPhaseInterceptNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldScenario
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture
from scripts.rsi_team_contextual_action_dataset import courses_from_protocol, entry_features
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
        parser.error("output directory already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    arm = next((item for item in protocol["arms"] if item["name"] == args.arm), None)
    v21_protocol = json.loads(
        Path(protocol["development_protocol_path"]).read_text(encoding="utf-8")
    )
    v21_root = Path(protocol["development_result_root"])
    v21_arms = [item["name"] for item in v21_protocol["arms"]]
    v21_reports = {
        name: json.loads((v21_root / name / "report.json").read_text(encoding="utf-8"))
        for name in v21_arms
    }
    if (
        protocol.get("schema") != "rsi_team_motor_library_expansion_protocol_v22"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("frames") != 250
        or arm is None
        or len({item["name"] for item in protocol["arms"]}) != len(protocol["arms"])
        or v21_reports["parent"]["report_hash"] != protocol["v21_parent_report_hash"]
        or any(
            report["protocol_hash"]
            != hash_bytes(Path(protocol["development_protocol_path"]).read_bytes())
            for report in v21_reports.values()
        )
    ):
        raise ValueError("uncommitted or invalid SIM_ONLY expansion protocol")
    unresolved = [
        course["name"]
        for index, course in enumerate(courses_from_protocol(v21_protocol))
        if not any(v21_reports[name]["rows"][index]["useful_pass"] for name in v21_arms)
    ]
    if unresolved != protocol["uncovered_development_courses"] or len(unresolved) != 11:
        raise ValueError("uncovered development course commitment mismatch")
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
        Path(__file__).with_name("rsi_team_contextual_action_dataset.py"),
        Path(__file__).with_name("rsi_team_taskspace_first_touch.py"),
        Path(__file__).with_name("rsi_team_intercept_navigation_search.py"),
        root / "src/rosclaw_soccer/rsi/team_phase_intercept_navigation.py",
        root / "src/rosclaw_soccer/rsi/taskspace_swing_probe.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    source_hashes = {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }
    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for course in courses_from_protocol(v21_protocol):
        if course["name"] not in unresolved:
            continue
        scenario = IndependentTeamWorldScenario(
            course["scenario_id"],
            tuple(course["ball_initial_position_m"]),
            tuple(course["ball_initial_velocity_mps"]),
            course["seed"],
        )
        navigation = TeamPhaseInterceptNavigation(
            agent_id=protocol["focal_agent_id"],
            foundation_hash=policy_hash,
            foundation_config_hash=config_hash,
            forward_gain=0.4,
            lateral_gain=0.8,
        )
        episode_protocol = dict(protocol)
        episode_protocol["candidate_action"] = arm["candidate_action"]
        folder = args.output_dir / course["name"]
        report = _run_one(
            mode="candidate",
            asset_root=args.asset_root,
            output_dir=folder,
            fixture=fixture,
            scenario=scenario,
            protocol=episode_protocol,
            navigation_policy=navigation,
        )
        outcome = _score(report, folder / "candidate/trajectory.npz")
        outcome.update(
            {
                "course": course["name"],
                "entry": entry_features(folder / "candidate/taskspace_trace.npz", 30),
                "navigation_contract_hash": navigation.contract_hash,
            }
        )
        if (
            outcome["entry"]["hash"]
            != v21_reports["parent"]["rows"][int(course["name"][1:])]["entry"]["hash"]
        ):
            raise ValueError("motor action changed the predeclared entry observation")
        rows.append(outcome)
        print(
            f"{args.arm} {course['name']} safe={outcome['safe']} useful={outcome['useful_pass']}",
            flush=True,
        )
    if source_hashes != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("source changed during frozen physical motor expansion")
    summary = {
        "schema": "rsi_team_motor_library_expansion_arm_v22",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "source_hashes": source_hashes,
        "asset_body_hash": qualification.body_hash,
        "arm": arm,
        "rows": rows,
        "safe_contact_count": sum(bool(row["safe"] and row["foot_contact_frames"]) for row in rows),
        "useful_pass_count": sum(bool(row["useful_pass"]) for row in rows),
        "unsafe_count": sum(not row["safe"] for row in rows),
        "promotion_authorized": False,
    }
    summary["report_hash"] = hash_json(summary)
    (args.output_dir / "report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_MOTOR_EXPANSION="
        + json.dumps(
            {
                key: summary[key]
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
