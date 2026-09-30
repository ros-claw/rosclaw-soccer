"""Audit PhysX collision-group smoke against same-source local G1 controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.independent_first_touch_bank import post_contact_displacement
from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_vector_first_touch
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def audit(overlap_root: Path, separated_root: Path, local_roots: list[Path]) -> dict[str, Any]:
    overlap = json.loads((overlap_root / "report.json").read_text(encoding="utf-8"))
    separated = json.loads((separated_root / "report.json").read_text(encoding="utf-8"))
    separated_audit = audit_vector_first_touch(separated_root)
    locals_ = [
        json.loads((root / "report.json").read_text(encoding="utf-8")) for root in local_roots
    ]
    local_audits = [audit_vector_first_touch(root) for root in local_roots]
    groups = [
        {
            "includes": [f"/World/Env{i}"],
            "filters": [
                f"/World/diagnostic_collisions/group{i}",
                "/World/diagnostic_collisions/global_group",
            ],
        }
        for i in range(2)
    ]
    if (
        overlap.get("schema") != "rsi_isaac_vector_first_touch_smoke_v1"
        or overlap.get("activation_ceiling") != "SIM_ONLY"
        or overlap.get("learning_authorized") is not False
        or overlap.get("promotion_authorized") is not False
        or overlap.get("report_hash")
        != hash_json({k: v for k, v in overlap.items() if k != "report_hash"})
        or overlap.get("trace_hash") != hash_bytes((overlap_root / "trace.npz").read_bytes())
        or overlap.get("body_trace_hash")
        != hash_bytes((overlap_root / "body_trace.npz").read_bytes())
        or overlap.get("overlap_collision_filtered_diagnostic") is not True
        or overlap.get("lane_spacing_m") != 0.0
        or overlap.get("collision_filter_contract")
        != {"groups": groups, "physics_scene_inverted_filter": True}
        or len(overlap.get("environments", [])) != 2
        or separated.get("overlap_collision_filtered_diagnostic") is not False
        or separated.get("lane_spacing_m") != 8.0
        or separated_audit["independent_physical_episode_count"] != 2
        or any(
            local_audit["independent_physical_episode_count"] != 1 for local_audit in local_audits
        )
        or any(r["source_hash"] != overlap["source_hash"] for r in [separated, *locals_])
        or any(r["asset_hash"] != overlap["asset_hash"] for r in [separated, *locals_])
        or any(
            r["sonic_qualification_hash"] != overlap["sonic_qualification_hash"]
            for r in [separated, *locals_]
        )
    ):
        raise ValueError("unbound PhysX collision-filter diagnostic")
    with np.load(overlap_root / "trace.npz", allow_pickle=False) as trace:
        position = trace["ball_position_m"]
        force = trace["ball_body_contact_force_peak_n"]
        spin = trace["ball_angular_velocity_rad_s"]
        if (
            position.shape != (300, 2, 3)
            or force.shape != (300, 2, 6)
            or spin.shape != (300, 2, 3)
            or not all(np.isfinite(value).all() for value in (position, force, spin))
            or np.any(force < 0)
        ):
            raise ValueError("invalid overlap physical trace")
        rows = []
        for i in range(2):
            episode = overlap["environments"][i]
            separated_episode = separated["environments"][i]
            local_episode = locals_[i]["environments"][0]
            first_frames = np.flatnonzero(np.max(force[:, i], axis=1) > 1.0)
            first = int(first_frames[0]) if len(first_frames) else None
            bodies = np.flatnonzero(np.max(force[:, i], axis=0) > 1.0).tolist()
            if (
                episode["lane_y_m"] != 0.0
                or episode["first_contact_frame"] != first
                or episode["contact_body_indices"] != bodies
                or episode["course"] != separated_episode["course"]
                or episode["course"] != local_episode["course"]
                or not np.allclose(episode["ball_final_local_xyz_m"], position[-1, i], atol=1e-5)
                or not 0.65 <= episode["minimum_pelvis_z_m"] <= 1.5
            ):
                raise ValueError(f"overlap contact ledger drift: {i}")
            overlap_disp = post_contact_displacement(position[:, i], first)
            with np.load(local_roots[i] / "trace.npz", allow_pickle=False) as local_trace:
                local_disp = post_contact_displacement(
                    local_trace["ball_position_m"][:, 0], local_episode["first_contact_frame"]
                )
            with np.load(separated_root / "trace.npz", allow_pickle=False) as separate_trace:
                separated_disp = post_contact_displacement(
                    separate_trace["ball_position_m"][:, i],
                    separated_episode["first_contact_frame"],
                )
            equivalent = bool(
                first == local_episode["first_contact_frame"]
                and bodies == local_episode["contact_body_indices"]
                and abs(episode["minimum_pelvis_z_m"] - local_episode["minimum_pelvis_z_m"]) <= 0.02
                and overlap_disp["forward_60_m"] is not None
                and local_disp["forward_60_m"] is not None
                and overlap_disp["lateral_60_m"] is not None
                and local_disp["lateral_60_m"] is not None
                and abs(overlap_disp["forward_60_m"] - local_disp["forward_60_m"]) <= 0.10
                and abs(overlap_disp["lateral_60_m"] - local_disp["lateral_60_m"]) <= 0.10
            )
            rows.append(
                {
                    "lane": i,
                    "overlap_first_contact_frame": first,
                    "local_first_contact_frame": local_episode["first_contact_frame"],
                    "overlap_contact_body_indices": bodies,
                    "local_contact_body_indices": local_episode["contact_body_indices"],
                    "overlap_displacement": overlap_disp,
                    "local_displacement": local_disp,
                    "separated_displacement": separated_disp,
                    "equivalent_to_local": equivalent,
                }
            )
    report: dict[str, Any] = {
        "schema": "rsi_physx_overlap_filter_smoke_audit_v1",
        "activation_ceiling": "SIM_ONLY",
        "overlap_report_hash": overlap["report_hash"],
        "separated_report_hash": separated["report_hash"],
        "local_report_hashes": [r["report_hash"] for r in locals_],
        "lanes": rows,
        "equivalence_gate_passed": all(row["equivalent_to_local"] for row in rows),
        "batched_training_authorized": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-root", required=True, type=Path)
    parser.add_argument("--separated-root", required=True, type=Path)
    parser.add_argument("--local-root", required=True, nargs=2, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or len({args.overlap_root, args.separated_root, *args.local_root}) != 4:
        parser.error("new output and four distinct evidence roots required")
    report = audit(args.overlap_root, args.separated_root, args.local_root)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"OVERLAP_FILTER_AUDIT={report['report_hash']}", flush=True)
    print(f"EQUIVALENCE_GATE_PASSED={report['equivalence_gate_passed']}", flush=True)


if __name__ == "__main__":
    main()
