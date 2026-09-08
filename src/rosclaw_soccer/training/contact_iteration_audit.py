"""Reconstruct iteration outcomes from hash-bound complete match trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.active_team_probe import validate_probe
from rosclaw_soccer.training.owned_contact_learning import reward


def audit(*, sources: tuple[Path, ...], output: Path) -> dict[str, Any]:
    if output.exists() or not sources or len(set(sources)) != len(sources):
        raise ValueError("audit needs unique sources and a fresh output")
    rows = []
    for path in sources:
        report = validate_probe(path)
        with np.load(path.parent / "primary.npz", allow_pickle=False) as archive:
            time = archive["time"]
            option = archive["option_agent_code"]
            contact = archive["ball_contact_agent_code"]
            foot = np.isin(archive["ball_contact_effector_code"], (1, 2))
            neural_contacts = (
                (option > 0) & (option == contact) & foot & (archive["ball_contact_force_n"] > 0)
            )
            collisions = archive["robot_robot_contact_count"] > 0
            rows.append(
                {
                    "source": str(path),
                    "source_hash": hash_bytes(path.read_bytes()),
                    "implementation_hash": hash_json(report["implementation"]),
                    "duration_sec": float(time[-1]),
                    "safe": report["results"][0]["safe"],
                    "all_individual_bodies_safe": all(
                        q["safe"] for q in report["results"][0]["qualities"]
                    ),
                    "robot_collision_frames": int(collisions.sum()),
                    "neural_option_frames": int(np.count_nonzero(option)),
                    "neural_foot_contact_frames": int(neural_contacts.sum()),
                    "causal_passes": sum(
                        x["physical_receive_confirmed"] for x in report["causal_pass_feedback"]
                    ),
                    "qualified_pass_score": reward(report)[1],
                    "termination": report["termination"]["reason"],
                    "exact_replay": report["exact_replay"],
                    "failure_feedback": report["causal_pass_feedback"],
                }
            )
    result = {
        "schema_version": "rosclaw_soccer.contact_iteration_audit.v1",
        "rows": rows,
        "paired_trials": len(rows),
        "physical_trajectories": 2 * len(rows),
        "qualified_pass_total": sum(r["qualified_pass_score"] for r in rows),
        "implementation_revision_count": len({r["implementation_hash"] for r in rows}),
        "cross_revision_rows_are_not_matched_ablations": True,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
    }
    result["report_hash"] = hash_json(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(sources=tuple(args.source), output=args.output)))
