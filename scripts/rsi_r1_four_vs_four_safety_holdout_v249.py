"""Predeclared bilateral 4v4 safety/ball-play holdout for SIM_ONLY braking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.active_team_probe import run_probe, validate_probe
from rosclaw_soccer.training.contact_control_profile import ContactControlProfile


def run(assets: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("new external simulation evidence directory required")
    profile = ContactControlProfile(
        separation_m=1.2,
        stiffness_scale=1.0,
        guard_margin_rad=0.04,
        outward_waist_braking_damping=16.0,
        outward_ankle_pitch_braking_damping=16.0,
    )
    courses = ((False, -0.08), (False, 0.08), (True, -0.08), (True, 0.08))
    output.mkdir(parents=True)
    rows = []
    for blue, offset in courses:
        destination = output / f"{'blue' if blue else 'red'}-{offset:+.2f}"
        run_probe(
            asset_root=assets,
            output=destination,
            active=True,
            duration=25.0,
            four_vs_four=True,
            blue_kickoff=blue,
            kickoff_offset_m=offset,
            basic_ball_play=True,
            bilateral_kick_options=True,
            anticipatory_contact=True,
            forward_receiver_lane=True,
            all_role_clearance=True,
            contact_control_profile=profile,
        )
        report = validate_probe(destination / "probe.json")
        with np.load(destination / "primary.npz", allow_pickle=False) as archive:
            nonfoot = int(np.count_nonzero(archive["ball_nonfoot_contact_agent_code"]))
        skills = [event["skill"] for event in report["assessment"]["events"]]
        rows.append(
            {
                "blue_kickoff": blue,
                "offset_m": offset,
                "probe_hash": report["report_hash"],
                "exact_replay": report["exact_replay"],
                "safe": report["results"][0]["safe"],
                "termination": report["termination"],
                "physical_pass_events": skills.count("pass"),
                "physical_shot_events": skills.count("shot"),
                "physical_save_events": skills.count("save"),
                "nonfoot_contact_frames": nonfoot,
                "physical_participants": sum(
                    row["physical_ball_contact_frames"] > 0 for row in report["engagement"]
                ),
            }
        )
    gate = {
        "exact_replay_all": all(row["exact_replay"] for row in rows),
        "safe_all": all(row["safe"] for row in rows),
        "duration_all": all(row["termination"]["reason"] == "TIME_LIMIT" for row in rows),
        "clean_foot_all": all(row["nonfoot_contact_frames"] == 0 for row in rows),
        "physical_pass_at_least_two": sum(row["physical_pass_events"] for row in rows) >= 2,
        "three_participants_each": all(row["physical_participants"] >= 3 for row in rows),
    }
    result = {
        "schema": "rosclaw_soccer.rsi.four_vs_four_safety_holdout_v249.v1",
        "source_hash": hash_bytes(Path(__file__).read_bytes()),
        "profile": profile.__dict__,
        "courses": rows,
        "gate": gate,
        "gate_passed": all(gate.values()),
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    result["report_hash"] = hash_json(result)
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.asset_root, args.output)
    print(json.dumps({"gate": result["gate"], "report_hash": result["report_hash"]}))


if __name__ == "__main__":
    main()
