"""Search bounded physical strike follow-through on consumed paired 3v3 scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.rsi.team_foot_velocity_chooser import TeamFootVelocityChooser
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture
from scripts.rsi_team_contextual_action_dataset import entry_features
from scripts.rsi_team_foot_velocity_fresh_exam import scenarios
from scripts.rsi_team_intercept_navigation_search import _score
from scripts.rsi_team_taskspace_first_touch import TeamSwingMotor, _run_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("bounded strike-search output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_strike_follow_through_protocol_v35"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("consumed_scene_indices") != [3, 4, 13, 16, 17, 20, 21, 23]
        or protocol.get("through_offsets_m") != [0.08, 0.16]
        or protocol.get("selection_gate") != {"maximum_unsafe": 1, "minimum_useful": 3}
    ):
        raise ValueError("invalid frozen consumed-scene search")
    reference_path = Path(protocol["reference_exam_path"])
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (
        reference.get("report_hash") != protocol["reference_exam_hash"]
        or hash_json({key: value for key, value in reference.items() if key != "report_hash"})
        != protocol["reference_exam_hash"]
        or reference.get("gate_passed") is not False
    ):
        raise ValueError("unsealed failed fresh exam required")
    source_protocol = json.loads(Path(protocol["reference_protocol_path"]).read_text())
    model_path = Path(protocol["model_path"])
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if (
        model.get("model_hash") != protocol["model_hash"]
        or hash_json({key: value for key, value in model.items() if key != "model_hash"})
        != protocol["model_hash"]
    ):
        raise ValueError("changed frozen navigation model")
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
        root / "src/rosclaw_soccer/rsi/team_foot_velocity_chooser.py",
        root / "src/rosclaw_soccer/rsi/taskspace_swing_probe.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    source_hashes = {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }
    physical_courses = scenarios(source_protocol)
    args.output_dir.mkdir(parents=True)
    variants = []
    for through in protocol["through_offsets_m"]:
        name = f"through_{int(round(through * 100)):02d}cm"
        rows: list[dict[str, Any]] = []
        for index in protocol["consumed_scene_indices"]:
            scene = physical_courses[index]
            parent_row = reference["rows"][index]
            if parent_row["scenario_hash"] != scene.scenario_hash:
                raise ValueError("consumed scene identity changed")
            navigation = TeamFootVelocityChooser(
                agent_id=protocol["focal_agent_id"],
                foundation_hash=policy_hash,
                foundation_config_hash=config_hash,
                model_path=model_path,
            )
            action = dict(source_protocol["candidate_action"])
            action["strike_through_m"] = through
            motor = TeamSwingMotor(
                protocol["focal_agent_id"], True, action, activation_selector=navigation
            )
            episode_protocol = dict(source_protocol)
            episode_protocol["candidate_action"] = action
            folder = args.output_dir / name / f"f{index:03d}"
            try:
                report = _run_one(
                    mode="candidate",
                    asset_root=args.asset_root,
                    output_dir=folder,
                    fixture=fixture,
                    scenario=scene,
                    protocol=episode_protocol,
                    navigation_policy=navigation,
                    motor_option=motor,
                )
                score = _score(report, folder / "candidate/trajectory.npz")
                entry = entry_features(folder / "candidate/taskspace_trace.npz", 30)
                if entry["hash"] != parent_row["parent_entry_hash"]:
                    raise ValueError("changed pre-intervention physical state")
                status = "COMPLETE"
            except ValueError as error:
                if str(error) != "incomplete paired motor episode":
                    raise
                score = {
                    "safe": False,
                    "foot_contact_frames": [],
                    "useful_pass": False,
                    "outgoing_ball_vx_mps": None,
                    "report_hash": None,
                }
                status = "INCOMPLETE"
            rows.append(
                {
                    "scene": f"f{index:03d}",
                    "scenario_hash": scene.scenario_hash,
                    "selected_arm": navigation.selected_arm,
                    "status": status,
                    "score": score,
                }
            )
            print(
                f"{name} f{index:03d} safe={score['safe']} "
                f"useful={score['useful_pass']} outgoing={score['outgoing_ball_vx_mps']}",
                flush=True,
            )
        variants.append(
            {
                "name": name,
                "strike_through_m": through,
                "rows": rows,
                "unsafe": sum(not row["score"]["safe"] for row in rows),
                "safe_contact": sum(
                    bool(row["score"]["safe"] and row["score"]["foot_contact_frames"])
                    for row in rows
                ),
                "useful": sum(row["score"]["useful_pass"] for row in rows),
            }
        )
    if source_hashes != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("physical source changed during bounded strike search")
    eligible = [
        variant
        for variant in variants
        if variant["unsafe"] <= protocol["selection_gate"]["maximum_unsafe"]
        and variant["useful"] >= protocol["selection_gate"]["minimum_useful"]
    ]
    selected = sorted(
        eligible,
        key=lambda item: (-item["useful"], item["unsafe"], -item["safe_contact"], item["name"]),
    )
    result = {
        "schema": "rsi_team_strike_follow_through_report_v35",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "reference_exam_hash": protocol["reference_exam_hash"],
        "model_hash": protocol["model_hash"],
        "body_hash": qualification.body_hash,
        "physical_source_hashes": source_hashes,
        "variants": variants,
        "selected_variant": None if not selected else selected[0]["name"],
        "fresh_exam_authorized": bool(selected),
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_STRIKE_FOLLOW_THROUGH="
        + json.dumps(
            {
                "report_hash": result["report_hash"],
                "selected_variant": result["selected_variant"],
                "fresh_exam_authorized": result["fresh_exam_authorized"],
                "variant_scores": [
                    {key: variant[key] for key in ("name", "unsafe", "safe_contact", "useful")}
                    for variant in variants
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
