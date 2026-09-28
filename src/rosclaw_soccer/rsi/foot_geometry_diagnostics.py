"""Mine audited SIM_ONLY first-touch geometry; never grants motor authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_vector_first_touch
from rosclaw_soccer.sim.contracts import hash_json

RIGHT_KNEE_INDEX = G1_DDS_JOINT_NAMES.index("right_knee_joint")


def mine_foot_geometry(folder: Path) -> dict[str, Any]:
    audit = audit_vector_first_touch(folder)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if report.get("foot_geometry_body_names") != [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_knee_link",
        "right_knee_link",
    ]:
        raise ValueError("audited foot geometry required")
    with (
        np.load(folder / "body_trace.npz", allow_pickle=False) as body,
        np.load(folder / "trace.npz", allow_pickle=False) as physics,
    ):
        positions = body["foot_geometry_position_before_step_m"]
        velocities = body["foot_geometry_velocity_before_step_m_s"]
        ball = body["ball_position_before_step_m"]
        joint = body["joint_position_rad"]
        force = physics["ball_body_contact_force_peak_n"]
        episodes = []
        for lane, row in enumerate(report["environments"]):
            contact = row["first_contact_frame"]
            if contact is None:
                episodes.append(
                    {
                        "lane": lane,
                        "incoming": row["course"]["ball_vx_m_s"] < 0,
                        "first_contact_frame": None,
                        "first_contact_kind": "MISS",
                    }
                )
                continue
            active = np.flatnonzero(force[contact, lane] > 1.0).tolist()
            first_kind = "FOOT" if active and set(active) <= {0, 1} else "NONFOOT"
            foot_minus_knee = float(positions[contact, lane, 1, 0] - positions[contact, lane, 3, 0])
            window = slice(max(0, contact - 30), contact)
            knee_angle = joint[window, lane, RIGHT_KNEE_INDEX]
            foot_forward = positions[window, lane, 1, 0] - positions[window, lane, 3, 0]
            correlation = (
                float(np.corrcoef(knee_angle, foot_forward)[0, 1])
                if len(knee_angle) >= 3
                and np.std(knee_angle) > 1e-6
                and np.std(foot_forward) > 1e-6
                else None
            )
            episodes.append(
                {
                    "lane": lane,
                    "incoming": row["course"]["ball_vx_m_s"] < 0,
                    "first_contact_frame": contact,
                    "first_contact_kind": first_kind,
                    "eventual_clean_foot_only": bool(
                        row["contact_body_indices"] and set(row["contact_body_indices"]) <= {0, 1}
                    ),
                    "right_foot_minus_knee_x_m": foot_minus_knee,
                    "right_foot_minus_ball_x_m": float(
                        positions[contact, lane, 1, 0] - ball[contact, lane, 0]
                    ),
                    "right_knee_minus_ball_x_m": float(
                        positions[contact, lane, 3, 0] - ball[contact, lane, 0]
                    ),
                    "right_foot_vx_m_s": float(velocities[contact, lane, 1, 0]),
                    "right_knee_angle_rad": float(joint[contact, lane, RIGHT_KNEE_INDEX]),
                    "precontact_knee_vs_foot_forward_correlation": correlation,
                }
            )
    incoming = [
        row for row in episodes if row["incoming"] and row["first_contact_frame"] is not None
    ]
    if not incoming:
        raise ValueError("incoming contact episodes required for shin-first diagnosis")
    result: dict[str, Any] = {
        "schema": "rsi_isaac_foot_geometry_diagnostics_v1",
        "activation_ceiling": "SIM_ONLY",
        "partition": "CONSUMED_DEV",
        "source_report_hash": report["report_hash"],
        "source_audit_hash": audit["report_hash"],
        "episode_count": len(episodes),
        "incoming_contact_count": len(incoming),
        "incoming_foot_behind_knee_count": sum(
            row["right_foot_minus_knee_x_m"] < 0 for row in incoming
        ),
        "incoming_first_nonfoot_count": sum(
            row["first_contact_kind"] == "NONFOOT" for row in incoming
        ),
        "episodes": episodes,
        "causal_claim_authorized": False,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("diagnostic output already exists")
    result = mine_foot_geometry(args.folder)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
