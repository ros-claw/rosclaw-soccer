"""Read-only sustained-possession audit of sealed eight-G1 receiving traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_possession import evaluate_receiving_possession
from rosclaw_soccer.training.receiving_rollout import (
    explain_receiving_window,
    receiving_window,
)


def audit(*, evidence_dir: Path, output_dir: Path) -> dict[str, Any]:
    source = Path(__file__)
    evaluator = source.parents[1] / "src/rosclaw_soccer/training/receiving_possession.py"
    authoritative_source = source.parents[1] / "src/rosclaw_soccer/training/receiving_rollout.py"
    source_hash, evaluator_hash = (
        hash_bytes(source.read_bytes()),
        hash_bytes(evaluator.read_bytes()),
    )
    authoritative_hash = hash_bytes(authoritative_source.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY possession-audit directory required")
    payload: dict[str, Any] = json.loads((evidence_dir / "report.json").read_text(encoding="utf-8"))
    commitment = payload.pop("report_hash")
    if (
        commitment != hash_json(payload)
        or payload["schema"] != "rosclaw_soccer.rsi.receiving_ball_follow_probe.v1"
        or payload["promotion_authorized"] is not False
        or payload["fresh8_opened"] is not False
        or payload["course"]["agent_id"] != "red.finisher"
    ):
        raise ValueError("sealed consumed eight-G1 receiving evidence required")
    output_rows = []
    for label in ("parent", "follow075"):
        row = next(item for item in payload["rows"] if item["label"] == label)
        path = evidence_dir / f"{label}.npz"
        if row["trace_hash"] != hash_bytes(path.read_bytes()):
            raise ValueError("shared-world physical trace hash mismatch")
        qualities = row["result"]["qualities"]
        agent_ids = tuple(sorted(item["agent_id"] for item in qualities))
        agent_code = agent_ids.index("red.finisher") + 1
        with np.load(path, allow_pickle=False) as trace:
            full_trace = {key: np.asarray(trace[key]) for key in trace.files}
            measurement = evaluate_receiving_possession(
                ball_pose=np.asarray(full_trace["ball_pose"], dtype=np.float64),
                ball_velocity=np.asarray(full_trace["ball_velocity"], dtype=np.float64),
                pelvis_pose=np.asarray(full_trace["red_finisher_pelvis_pose"], dtype=np.float64),
                contact_agent_code=np.asarray(full_trace["ball_contact_agent_code"]),
                contact_foot_code=np.asarray(full_trace["ball_contact_foot_code"]),
                nonfoot_agent_code=np.asarray(full_trace["ball_nonfoot_contact_agent_code"]),
                agent_code=agent_code,
                body_safe=bool(row["safe"]),
            )
            _, authoritative = receiving_window(
                full_trace,
                agent_ids=agent_ids,
                agent_id="red.finisher",
                start=20,
                frames=100,
            )
            explanation = explain_receiving_window(
                full_trace,
                agent_ids=agent_ids,
                agent_id="red.finisher",
                start=20,
                frames=100,
            )
        output_rows.append(
            {
                "label": label,
                "source_trace_hash": row["trace_hash"],
                "auxiliary_pelvis_relative_measurement": measurement,
                "authoritative_receiving_window": authoritative,
                "authoritative_explanation": explanation,
            }
        )
    if (
        hash_bytes(source.read_bytes()) != source_hash
        or hash_bytes(evaluator.read_bytes()) != evaluator_hash
        or hash_bytes(authoritative_source.read_bytes()) != authoritative_hash
    ):
        raise RuntimeError("possession evaluator changed during physical evidence audit")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_possession_audit.v2",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "evaluator_hash": evaluator_hash,
        "authoritative_source_hash": authoritative_hash,
        "source_report_hash": commitment,
        "rows": output_rows,
        "fresh8_opened": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    report = audit(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "rows": [
                    {
                        "label": row["label"],
                        "auxiliary": row["auxiliary_pelvis_relative_measurement"],
                        "authoritative": row["authoritative_receiving_window"],
                        "failed_criteria": row["authoritative_explanation"]["failed_criteria"],
                    }
                    for row in report["rows"]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
