"""Authenticated precontact Isaac snapshots for SIM_ONLY short-horizon research.

Future frozen-parent targets are included solely for replay-equivalence diagnosis.
They are privileged hindsight, never an online policy observation or promotion truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_vector_first_touch
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "rsi_isaac_first_touch_snapshot_bank_v1"
LEAD_FRAMES = 30
WINDOW_FRAMES = 80


def _extract(
    folders: tuple[Path, ...],
    *,
    lead_frames: int = LEAD_FRAMES,
    window_frames: int = WINDOW_FRAMES,
    fixed_start_frame: int | None = None,
    partition: str = "CONSUMED_DEV",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not folders or len(set(folders)) != len(folders):
        raise ValueError("nonempty distinct audited source folders required")
    if (
        type(lead_frames) is not int
        or type(window_frames) is not int
        or not 15 <= lead_frames < window_frames <= 150
        or partition not in {"CONSUMED_DEV", "SEALED_HOLDOUT"}
        or (
            fixed_start_frame is not None
            and (type(fixed_start_frame) is not int or fixed_start_frame < 1)
        )
    ):
        raise ValueError("invalid bounded precontact snapshot horizon")
    rows: list[dict[str, Any]] = []
    samples: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "root_pose_local_xyzw_m",
            "root_velocity_world",
            "joint_position_rad",
            "joint_velocity_rad_s",
            "ball_position_local_m",
            "ball_linear_velocity_m_s",
            "ball_angular_velocity_rad_s",
            "foot_geometry_position_local_m",
            "foot_geometry_velocity_m_s",
            "privileged_parent_joint_targets_rad",
            "reference_contact_force_n",
            "reference_ball_position_local_m",
            "reference_root_pose_local_xyzw_m",
        )
    }
    identities: set[tuple[str, str, str]] = set()
    for folder in folders:
        audit = audit_vector_first_touch(folder)
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        if (
            report.get("schema") != "rsi_isaac_vector_first_touch_smoke_v1"
            or report.get("foot_geometry_body_names")
            != [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_knee_link",
                "right_knee_link",
            ]
            or report.get("torch_batch_plan_only") is not True
        ):
            raise ValueError("qualified G1 foot geometry and frozen SONIC required")
        identities.add(
            (report["source_hash"], report["asset_hash"], report["sonic_qualification_hash"])
        )
        with (
            np.load(folder / "body_trace.npz", allow_pickle=False) as body,
            np.load(folder / "trace.npz", allow_pickle=False) as physics,
        ):
            for lane, entry in enumerate(report["environments"]):
                first = entry["first_contact_frame"]
                if entry["course"]["ball_vx_m_s"] >= 0 or first is None:
                    continue
                start = first - lead_frames if fixed_start_frame is None else fixed_start_frame
                if (
                    start < 1
                    or first - start < lead_frames
                    or start + window_frames > report["frames"]
                ):
                    raise ValueError("incoming first contact outside snapshot horizon")
                lane_y = float(entry["lane_y_m"])
                root = body["root_pose_xyzw_m"][start, lane].copy()
                root[1] -= lane_y
                ball = body["ball_position_before_step_m"][start, lane].copy()
                ball[1] -= lane_y
                feet = body["foot_geometry_position_before_step_m"][start, lane].copy()
                feet[:, 1] -= lane_y
                future_ball = physics["ball_position_m"][start : start + window_frames, lane].copy()
                future_ball[:, 1] -= lane_y
                future_root = body["root_pose_xyzw_m"][start : start + window_frames, lane].copy()
                future_root[:, 1] -= lane_y
                values = {
                    "root_pose_local_xyzw_m": root,
                    "root_velocity_world": body["root_velocity_world"][start, lane].copy(),
                    "joint_position_rad": body["joint_position_rad"][start, lane].copy(),
                    "joint_velocity_rad_s": body["joint_velocity_rad_s"][start, lane].copy(),
                    "ball_position_local_m": ball,
                    "ball_linear_velocity_m_s": body["ball_linear_velocity_before_step_m_s"][
                        start, lane
                    ].copy(),
                    "ball_angular_velocity_rad_s": physics["ball_angular_velocity_rad_s"][
                        start - 1, lane
                    ].copy(),
                    "foot_geometry_position_local_m": feet,
                    "foot_geometry_velocity_m_s": body["foot_geometry_velocity_before_step_m_s"][
                        start, lane
                    ].copy(),
                    "privileged_parent_joint_targets_rad": body["joint_target_rad"][
                        start : start + window_frames, lane
                    ].copy(),
                    "reference_contact_force_n": physics["ball_body_contact_force_peak_n"][
                        start : start + window_frames, lane
                    ].copy(),
                    "reference_ball_position_local_m": future_ball,
                    "reference_root_pose_local_xyzw_m": future_root,
                }
                for key, value in values.items():
                    if not np.isfinite(value).all():
                        raise ValueError("nonfinite authenticated snapshot")
                    samples[key].append(value)
                rows.append(
                    {
                        "source_folder": str(folder.resolve()),
                        "source_report_hash": report["report_hash"],
                        "source_audit_hash": audit["report_hash"],
                        "lane": lane,
                        "start_frame": start,
                        "first_contact_offset": first - start,
                        "course": entry["course"],
                        "eventual_clean_foot_only": bool(
                            entry["contact_body_indices"]
                            and set(entry["contact_body_indices"]) <= {0, 1}
                        ),
                    }
                )
    if len(identities) != 1 or not rows:
        raise ValueError("heterogeneous or empty snapshot collection")
    arrays = {key: np.stack(values) for key, values in samples.items()}
    manifest = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "partition": partition,
        "source_identity": list(next(iter(identities))),
        "lead_frames": lead_frames,
        "window_frames": window_frames,
        "snapshot_count": len(rows),
        "snapshots": rows,
        "privileged_future_targets_diagnostic_only": True,
        "learning_authorized": False,
        "promotion_authorized": False,
    }
    if fixed_start_frame is not None:
        manifest["fixed_start_frame"] = fixed_start_frame
    return manifest, arrays


def build_snapshot_bank(
    folders: tuple[Path, ...],
    output: Path,
    *,
    lead_frames: int = LEAD_FRAMES,
    window_frames: int = WINDOW_FRAMES,
    fixed_start_frame: int | None = None,
    partition: str = "CONSUMED_DEV",
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("immutable snapshot bank output already exists")
    manifest, arrays = _extract(
        folders,
        lead_frames=lead_frames,
        window_frames=window_frames,
        fixed_start_frame=fixed_start_frame,
        partition=partition,
    )
    output.mkdir(parents=True)
    archive = output / "snapshots.npz"
    np.savez_compressed(archive, **arrays)  # type: ignore[arg-type]
    manifest["archive_hash"] = hash_bytes(archive.read_bytes())
    manifest["manifest_hash"] = hash_json(manifest)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def audit_snapshot_bank(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    archive = output / "snapshots.npz"
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("activation_ceiling") != "SIM_ONLY"
        or manifest.get("learning_authorized") is not False
        or manifest.get("promotion_authorized") is not False
        or manifest.get("privileged_future_targets_diagnostic_only") is not True
        or manifest.get("partition") not in {"CONSUMED_DEV", "SEALED_HOLDOUT"}
        or manifest.get("manifest_hash")
        != hash_json({k: v for k, v in manifest.items() if k != "manifest_hash"})
        or manifest.get("archive_hash") != hash_bytes(archive.read_bytes())
    ):
        raise ValueError("unauthenticated snapshot bank")
    # A source folder can supply multiple lanes; rebuild once per unique source.
    source_folders = tuple(
        dict.fromkeys(Path(row["source_folder"]) for row in manifest["snapshots"])
    )
    expected_manifest, expected_arrays = _extract(
        source_folders,
        lead_frames=manifest["lead_frames"],
        window_frames=manifest["window_frames"],
        fixed_start_frame=manifest.get("fixed_start_frame"),
        partition=manifest["partition"],
    )
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise ValueError("snapshot source commitment changed")
    with np.load(archive, allow_pickle=False) as stored:
        if set(stored.files) != set(expected_arrays) or any(
            not np.array_equal(stored[key], expected) for key, expected in expected_arrays.items()
        ):
            raise ValueError("snapshot state differs from authenticated precontact physics")
    return {
        "schema": "rsi_isaac_first_touch_snapshot_bank_audit_v1",
        "activation_ceiling": "SIM_ONLY",
        "manifest_hash": manifest["manifest_hash"],
        "snapshot_count": manifest["snapshot_count"],
        "reference_future_is_privileged": True,
        "promotion_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="*", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--lead-frames", type=int, default=LEAD_FRAMES)
    parser.add_argument("--window-frames", type=int, default=WINDOW_FRAMES)
    parser.add_argument("--fixed-start-frame", type=int)
    parser.add_argument(
        "--partition", choices=("CONSUMED_DEV", "SEALED_HOLDOUT"), default="CONSUMED_DEV"
    )
    args = parser.parse_args()
    if args.audit_only:
        if args.folders:
            parser.error("audit-only reads the committed output directory")
        result = audit_snapshot_bank(args.output)
    else:
        result = build_snapshot_bank(
            tuple(args.folders),
            args.output,
            lead_frames=args.lead_frames,
            window_frames=args.window_frames,
            fixed_start_frame=args.fixed_start_frame,
            partition=args.partition,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
