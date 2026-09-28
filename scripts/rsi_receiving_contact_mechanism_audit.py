"""Compare sealed eight-G1 receiving failures before designing the next actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_velocity_effects import receiving_velocity_effects

TRACES = {
    "prepost": ("rosclaw_soccer.rsi.receiving_full_prepost_capture_probe.v1", "full-prepost.npz"),
    "geometry": ("rosclaw_soccer.rsi.receiving_full_prepost_capture_probe.v1", "full-prepost.npz"),
    "brake": ("rosclaw_soccer.rsi.receiving_full_foot_brake_probe.v1", "full-foot-brake.npz"),
}


def audit(inputs: dict[str, Path], output: Path) -> dict[str, Any]:
    if output.exists() or output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("new external evidence file required")
    source_paths = {
        "audit": Path(__file__),
        "diagnostic": Path(__file__).resolve().parents[1]
        / "src/rosclaw_soccer/training/receiving_velocity_effects.py",
    }
    source_hashes = {name: hash_bytes(path.read_bytes()) for name, path in source_paths.items()}
    rows: dict[str, Any] = {}
    for label, (schema, filename) in TRACES.items():
        folder = inputs[label].resolve()
        report_path = folder / "report.json"
        trajectory = folder / filename
        report = json.loads(report_path.read_text(encoding="utf-8"))
        commitment = report.pop("report_hash")
        if (
            commitment != hash_json(report)
            or report["schema"] != schema
            or report["trace_hash"] != hash_bytes(trajectory.read_bytes())
            or report["promotion_authorized"] is not False
            or report["result"]["safe"] is not True
        ):
            raise ValueError(f"unqualified or altered {label} full-world evidence")
        agents = tuple(sorted(row["agent_id"] for row in report["result"]["qualities"]))
        if len(agents) != 8 or "red.finisher" not in agents:
            raise ValueError("complete eight-G1 focal roster required")
        with np.load(trajectory, allow_pickle=False) as archive:
            diagnostic = receiving_velocity_effects(
                {key: archive[key] for key in archive.files},
                agent_id="red.finisher",
                agent_code=agents.index("red.finisher") + 1,
            )
        rows[label] = {
            "report_hash": commitment,
            "report_file_hash": hash_bytes(report_path.read_bytes()),
            "trace_hash": report["trace_hash"],
            "diagnostic": diagnostic,
        }
    parent = rows["prepost"]["diagnostic"]
    brake = rows["brake"]["diagnostic"]
    report_out: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_contact_mechanism_audit.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hashes": source_hashes,
        "rows": rows,
        "brake_vs_prepost": {
            "parallel_speed_removed_delta_mps": brake["observed_parallel_speed_removed_mps"]
            - parent["observed_parallel_speed_removed_mps"],
            "outgoing_lateral_speed_abs_delta_mps": abs(brake["outgoing_lateral_mps"])
            - abs(parent["outgoing_lateral_mps"]),
            "tail_ball_speed_delta_mps": brake["tail_maximum_ball_speed_mps"]
            - parent["tail_maximum_ball_speed_mps"],
            "tail_foot_distance_delta_m": brake["tail_maximum_foot_distance_m"]
            - parent["tail_maximum_foot_distance_m"],
        },
        "all_physical_task_diagnostics_passed": all(
            row["diagnostic"]["task_contact_diagnostic_passed"] for row in rows.values()
        ),
        "promotion_authorized": False,
    }
    if {
        name: hash_bytes(path.read_bytes()) for name, path in source_paths.items()
    } != source_hashes:
        raise RuntimeError("source changed during eight-G1 evidence audit")
    report_out["report_hash"] = hash_json(report_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report_out, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in TRACES:
        parser.add_argument(f"--{label}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit({label: getattr(args, label) for label in TRACES}, args.output)
    print(json.dumps({"hash": report["report_hash"], **report["brake_vs_prepost"]}, sort_keys=True))


if __name__ == "__main__":
    main()
