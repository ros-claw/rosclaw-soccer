"""SIM_ONLY short eight-G1 student replay against its completed shared-world rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rsi_receiving_team_short_replay import replay

from rosclaw_soccer.providers.g1.qualified_receiving_student import QualifiedReceivingStudent
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def examine(
    *,
    asset_root: Path,
    captured: Path,
    warm_start: Path,
    training: Path,
    fresh: Path,
    full_world: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    previous: dict[str, Any] = json.loads(full_world.read_text(encoding="utf-8"))
    previous_hash = previous.pop("report_hash")
    reference_path = full_world.parent / "student.npz"
    bundle = QualifiedReceivingStudent.load(warm_start=warm_start, training=training, fresh=fresh)
    if (
        previous_hash != hash_json(previous)
        or previous["schema"] != "rosclaw_soccer.rsi.receiving_student_shared_world_exam.v1"
        or previous["model_hash"] != bundle.model_hash
        or previous["training_report_hash"] != bundle.training_report_hash
        or previous["fresh_report_hash"] != bundle.fresh_report_hash
        or previous["rows"][1]["label"] != "student"
        or previous["rows"][1]["trace_hash"] != hash_bytes(reference_path.read_bytes())
        or previous["shared_world_authoritative_reception"] is not False
    ):
        raise ValueError("sealed failed full eight-G1 student trajectory required")
    result = replay(
        asset_root=asset_root,
        captured=captured,
        output_dir=output_dir,
        student_bundle=bundle,
        reference_trace=reference_path,
    )
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("short-student wrapper source changed during physics")
    result.pop("report_hash")
    result["wrapper_source_hash"] = source_hash
    result["full_world_report_hash"] = previous_hash
    result["report_hash"] = hash_json(result)
    (output_dir / "report.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--captured", required=True, type=Path)
    parser.add_argument("--warm-start", required=True, type=Path)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--fresh", required=True, type=Path)
    parser.add_argument("--full-world", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    result = examine(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "report_hash",
                    "reference_first_foot_frame",
                    "replay_first_foot_frame",
                    "frame86_speed_error_mps",
                    "frame86_distance_error_m",
                    "training_proxy_fidelity_passed",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
