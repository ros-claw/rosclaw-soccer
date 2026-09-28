"""SIM_ONLY read-only motor tape of the failed full-eight-G1 receiving student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rsi_receiving_team_motor_capture import capture

from rosclaw_soccer.providers.g1.qualified_receiving_student import QualifiedReceivingStudent
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def collect(
    *,
    asset_root: Path,
    sonic_model_root: Path,
    captured: Path,
    prior: Path,
    warm_start: Path,
    training: Path,
    fresh: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    student = QualifiedReceivingStudent.load(warm_start=warm_start, training=training, fresh=fresh)
    report = capture(
        asset_root=asset_root,
        sonic_model_root=sonic_model_root,
        captured=captured,
        prior=prior,
        output_dir=output_dir,
        student_bundle=student,
    )
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("student motor capture wrapper changed during physics")
    report.pop("report_hash")
    report["wrapper_source_hash"] = source_hash
    report["report_hash"] = hash_json(report)
    (output_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--sonic-model-root", required=True, type=Path)
    parser.add_argument("--captured", required=True, type=Path)
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--warm-start", required=True, type=Path)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--fresh", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    report = collect(**vars(parser.parse_args()))
    print(
        json.dumps(
            {key: report[key] for key in ("report_hash", "trace_hash", "integration_hash")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
