"""Freeze a small causal SIM_ONLY navigation memory from paired physical episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.team_contextual_nav_policy import measured_entry_features
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_contextual_nav_train_protocol_v25"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("training_scenes") != 24
        or protocol.get("neighbors") != 3
    ):
        raise ValueError("uncommitted SIM_ONLY training protocol")
    reports = {}
    for dataset in protocol["datasets"]:
        for name, expected_hash in dataset["arm_report_hashes"].items():
            if name in reports:
                raise ValueError("duplicate arm")
            report = json.loads((Path(dataset["root"]) / name / "report.json").read_text())
            if (
                report.pop("report_hash", None) != expected_hash
                or hash_json(report) != expected_hash
            ):
                raise ValueError("unsealed physical training evidence")
            reports[name] = report
    if (
        len(reports) != 11
        or len({str(report["source_hashes"]) for report in reports.values()}) != 1
        or len({report["asset_body_hash"] for report in reports.values()}) != 1
        or any(len(report["rows"]) != 24 for report in reports.values())
    ):
        raise ValueError("mixed or incomplete physical development library")
    parent = reports["parent"]["rows"]
    for index in range(24):
        if len({report["rows"][index]["entry"]["hash"] for report in reports.values()}) != 1:
            raise ValueError("intervention changed causal entry state")
    features = []
    for row in parent:
        raw = np.asarray(row["entry"]["values"], dtype=float)
        ball = tuple(float(value) for value in raw[6:9])
        feet = tuple(tuple(float(value) for value in raw[start : start + 3]) for start in (12, 15))
        features.append(
            measured_entry_features(ball=ball, ball_vx=float(raw[9]), feet=feet)  # type: ignore[arg-type]
        )
    names = sorted(name for name in reports if name != "parent")
    root = Path(__file__).parents[1]
    paths = (
        Path(__file__),
        root / "src/rosclaw_soccer/rsi/team_contextual_nav_policy.py",
        root / "src/rosclaw_soccer/rsi/team_phase_intercept_navigation.py",
        root / "src/rosclaw_soccer/rsi/team_intercept_navigation.py",
    )
    model = {
        "schema": "rsi_team_contextual_navigation_memory_v25",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "physical_body_hash": reports["parent"]["asset_body_hash"],
        "physical_source_hashes": reports["parent"]["source_hashes"],
        "model_source_hashes": {
            str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in paths
        },
        "arm_names": names,
        "arm_parameters": {name: reports[name]["arm"] for name in names},
        "training_entry_hashes": [row["entry"]["hash"] for row in parent],
        "training_features": features,
        "training_outcomes": {
            name: [
                {
                    "safe": row["safe"],
                    "foot_contact": bool(row["foot_contact_frames"]),
                    "useful_pass": row["useful_pass"],
                }
                for row in reports[name]["rows"]
            ]
            for name in names
        },
        "reward_contract": {
            "unsafe": -5,
            "safe_no_contact": 0,
            "safe_contact": 1,
            "safe_useful_pass": 2,
            "all_three_neighbors_must_be_safe": True,
            "minimum_expected_reward_exclusive": 0,
        },
    }
    model["model_hash"] = hash_json(model)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_CONTEXTUAL_MODEL="
        + json.dumps(
            {"model_hash": model["model_hash"], "arm_count": len(names), "training_scenes": 24},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
