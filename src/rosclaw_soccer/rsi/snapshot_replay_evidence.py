"""Independent fail-closed audit of SIM_ONLY Isaac snapshot replay equivalence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.first_touch_snapshot_bank import audit_snapshot_bank
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def projected_probe_residual(
    desired: np.ndarray[Any, Any],
    baseline_target: np.ndarray[Any, Any],
    joint_limits: np.ndarray[Any, Any],
    joint_indices: list[int],
) -> np.ndarray[Any, Any]:
    """Recompute the bounded target delta after the physical joint-limit shield."""
    from rosclaw_soccer.rsi.first_touch_candidate import project_residual_target

    if (
        desired.shape != (3,)
        or baseline_target.shape != (29,)
        or joint_limits.shape != (3, 2)
        or len(joint_indices) != 3
        or not np.isfinite(desired).all()
        or not np.isfinite(baseline_target).all()
        or not np.isfinite(joint_limits).all()
    ):
        raise ValueError("invalid bounded snapshot probe inputs")
    return np.asarray(
        [
            project_residual_target(
                float(baseline_target[joint_index]),
                float(value),
                float(joint_limits[output_index, 0]),
                float(joint_limits[output_index, 1]),
            )[0]
            - float(baseline_target[joint_index])
            for output_index, (joint_index, value) in enumerate(
                zip(joint_indices, desired, strict=True)
            )
        ]
    )


def audit_snapshot_replay(
    folder: Path, *, snapshot_bank: Path, candidate_path: Path | None = None
) -> dict[str, Any]:
    bank_audit = audit_snapshot_bank(snapshot_bank)
    manifest = json.loads((snapshot_bank / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    trace_path = folder / "replay.npz"
    count = report.get("sample_count")
    start = report.get("start_index")
    shared_hash = report.get("shared_candidate_hash")
    probe = bool(report.get("knee_extension_probe", False) or shared_hash is not None)
    phase_target = report.get("phase_target_frames")
    phase_probe = phase_target is not None
    closed_loop = report.get("closed_loop_sonic", False)
    if (
        report.get("schema") != "rsi_isaac_first_touch_snapshot_replay_v1"
        or report.get("activation_ceiling") != "SIM_ONLY"
        or report.get("learning_authorized") is not False
        or report.get("promotion_authorized") is not False
        or report.get("privileged_future_targets_diagnostic_only") is not True
        or report.get("snapshot_bank_manifest_hash") != bank_audit["manifest_hash"]
        or report.get("asset_hash") != manifest["source_identity"][1]
        or report.get("foundation_qualification_hash") != manifest["source_identity"][2]
        or report.get("runner_source_hash", "").startswith("sha256:") is not True
        or type(report.get("knee_extension_probe", False)) is not bool
        or (
            shared_hash is not None
            and (report.get("knee_extension_probe") is True or not isinstance(shared_hash, str))
        )
        or (candidate_path is None) != (shared_hash is None)
        or (phase_probe and (probe or not closed_loop))
        or (not phase_probe and report.get("phase_ramp_frames") is not None)
        or type(closed_loop) is not bool
        or type(count) is not int
        or not 2 <= count <= 16
        or type(start) is not int
        or start < 0
        or start + count > manifest["snapshot_count"]
        or report.get("window_frames") != manifest["window_frames"]
        or not isinstance(report.get("rows"), list)
        or len(report["rows"]) != count
        or (
            "probe_applied_frames" in report
            and (
                not isinstance(report["probe_applied_frames"], list)
                or len(report["probe_applied_frames"]) != count
            )
        )
        or report.get("trace_hash") != hash_bytes(trace_path.read_bytes())
        or report.get("report_hash")
        != hash_json({k: v for k, v in report.items() if k != "report_hash"})
    ):
        raise ValueError("unauthenticated Isaac snapshot replay")
    frames = manifest["window_frames"]
    lead = manifest["lead_frames"]
    with (
        np.load(snapshot_bank / "snapshots.npz", allow_pickle=False) as bank,
        np.load(trace_path, allow_pickle=False) as replay,
    ):
        shapes = {
            "initial_root_pose_local_xyzw_m": (count, 7),
            "initial_root_velocity_world": (count, 6),
            "initial_joint_position_rad": (count, 29),
            "initial_joint_velocity_rad_s": (count, 29),
            "initial_ball_position_local_m": (count, 3),
            "initial_ball_velocity_world": (count, 6),
            "observed_ball_position_local_m": (frames, count, 3),
            "observed_root_pose_local_xyzw_m": (frames, count, 7),
            "observed_ball_body_contact_force_peak_n": (frames, count, 6),
        }
        if "knee_extension_probe" in report:
            shapes.update(
                pre_step_root_pose_local_xyzw_m=(frames, count, 7),
                pre_step_joint_position_rad=(frames, count, 29),
                pre_step_joint_velocity_rad_s=(frames, count, 29),
                pre_step_ball_position_local_m=(frames, count, 3),
                pre_step_ball_linear_velocity_m_s=(frames, count, 3),
                applied_probe_residual_rad=(frames, count, 3),
            )
        if "closed_loop_sonic" in report:
            shapes["predicted_baseline_joint_target_rad"] = (frames, count, 29)
        if phase_probe:
            shapes["applied_sonic_phase_offset_frames"] = (frames, count)
        if report.get("probe_joint_limits_recorded") is True:
            shapes["probe_joint_position_limits_rad"] = (count, 3, 2)
        if set(replay.files) != set(shapes) or any(
            replay[key].shape != shape or not np.isfinite(replay[key]).all()
            for key, shape in shapes.items()
        ):
            raise ValueError("invalid snapshot replay trace shape or finite values")
        expected = {
            "root_pose_m": (
                replay["initial_root_pose_local_xyzw_m"],
                bank["root_pose_local_xyzw_m"][start : start + count],
            ),
            "root_velocity_m_s": (
                replay["initial_root_velocity_world"],
                bank["root_velocity_world"][start : start + count],
            ),
            "joint_position_rad": (
                replay["initial_joint_position_rad"],
                bank["joint_position_rad"][start : start + count],
            ),
            "joint_velocity_rad_s": (
                replay["initial_joint_velocity_rad_s"],
                bank["joint_velocity_rad_s"][start : start + count],
            ),
            "ball_position_m": (
                replay["initial_ball_position_local_m"],
                bank["ball_position_local_m"][start : start + count],
            ),
            "ball_velocity_m_s": (
                replay["initial_ball_velocity_world"],
                np.concatenate(
                    (
                        bank["ball_linear_velocity_m_s"][start : start + count],
                        bank["ball_angular_velocity_rad_s"][start : start + count],
                    ),
                    axis=1,
                ),
            ),
        }
        if set(report.get("initial_state_max_absolute_error", {})) != set(expected):
            raise ValueError("incomplete initial-state restore ledger")
        max_initial = 0.0
        for key, (actual, source) in expected.items():
            delta = np.max(np.abs(actual - source), axis=1)
            if not np.allclose(
                delta, report["initial_state_max_absolute_error"][key], atol=1e-5, rtol=0
            ):
                raise ValueError("initial-state restore ledger differs from physical trace")
            max_initial = max(max_initial, float(np.max(delta)))
        observed_ball = replay["observed_ball_position_local_m"]
        reference_ball = bank["reference_ball_position_local_m"][start : start + count].transpose(
            1, 0, 2
        )
        observed_root = replay["observed_root_pose_local_xyzw_m"]
        reference_root = bank["reference_root_pose_local_xyzw_m"][start : start + count].transpose(
            1, 0, 2
        )
        observed_force = replay["observed_ball_body_contact_force_peak_n"]
        reference_force = bank["reference_contact_force_n"][start : start + count].transpose(
            1, 0, 2
        )
        if np.any(observed_force < 0):
            raise ValueError("negative physical contact force")
        if phase_probe:
            from rosclaw_soccer.rsi.sonic_phase_probe import phase_offset_frames

            if report.get("phase_ramp_frames") != 20:
                raise ValueError("snapshot phase schedule changed")
            expected_phase = np.asarray(
                [phase_offset_frames(frame, phase_target) for frame in range(frames)]
            )
            if not np.allclose(
                replay["applied_sonic_phase_offset_frames"],
                expected_phase[:, None],
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError("snapshot phase action differs from bounded schedule")
        if "knee_extension_probe" in report:
            if shared_hash is not None and report.get("probe_joint_limits_recorded") is not True:
                raise ValueError("shared motor candidate lacks physical joint limit evidence")
            if report.get("probe_joint_limits_recorded") is True:
                joint_limits = replay["probe_joint_position_limits_rad"]
                if np.any(joint_limits[:, :, 0] >= joint_limits[:, :, 1]):
                    raise ValueError("invalid bounded joint limit trace")
            if "closed_loop_sonic" in report:
                target = replay["predicted_baseline_joint_target_rad"]
                parent = bank["privileged_parent_joint_targets_rad"][
                    start : start + count
                ].transpose(1, 0, 2)
                first_error = float(np.max(np.abs(target[0] - parent[0])))
                if (
                    not np.isclose(
                        report.get("max_parent_target_error_at_snapshot_rad"),
                        first_error,
                        atol=1e-6,
                        rtol=0,
                    )
                    or not np.isfinite(report.get("warmup_max_target_error_rad", np.nan))
                    or report["warmup_max_target_error_rad"] > 1e-3
                    or (closed_loop and first_error > 1e-3)
                    or (not closed_loop and not np.allclose(target, parent, atol=1e-5, rtol=0))
                ):
                    raise ValueError("snapshot SONIC controller target commitment invalid")
            if (
                not np.allclose(
                    replay["pre_step_ball_position_local_m"][0],
                    replay["initial_ball_position_local_m"],
                    atol=1e-5,
                    rtol=0,
                )
                or not np.allclose(
                    replay["pre_step_ball_position_local_m"][1:],
                    observed_ball[:-1],
                    atol=1e-5,
                    rtol=0,
                )
                or not np.allclose(
                    replay["pre_step_root_pose_local_xyzw_m"][0],
                    replay["initial_root_pose_local_xyzw_m"],
                    atol=1e-5,
                    rtol=0,
                )
                or not np.allclose(
                    replay["pre_step_root_pose_local_xyzw_m"][1:],
                    observed_root[:-1],
                    atol=1e-5,
                    rtol=0,
                )
            ):
                raise ValueError("snapshot intervention pre-step state misaligned")
            if probe:
                from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
                from rosclaw_soccer.rsi.temporal_first_touch_policy import (
                    JOINT_NAMES,
                    knee_extension_probe_weights,
                    temporal_residual,
                )

                if shared_hash is None:
                    weights = knee_extension_probe_weights()[0]
                else:
                    from rosclaw_soccer.rsi.snapshot_shared_temporal_policy import load_candidate

                    if candidate_path is None:
                        raise ValueError("shared snapshot intervention lacks candidate")
                    candidate_hash, weights = load_candidate(
                        candidate_path, bank_manifest_hash=bank_audit["manifest_hash"]
                    )
                    if candidate_hash != shared_hash:
                        raise ValueError("shared snapshot intervention candidate changed")
                applied_counts = []
                joint_indices = [G1_DDS_JOINT_NAMES.index(name) for name in JOINT_NAMES]
                for lane in range(count):
                    active = np.flatnonzero(np.max(observed_force[:, lane], axis=1) > 1.0)
                    first = int(active[0]) if len(active) else None
                    applied_count = 0
                    for frame in range(frames):
                        relative = (
                            replay["pre_step_ball_position_local_m"][frame, lane]
                            - replay["pre_step_root_pose_local_xyzw_m"][frame, lane, :3]
                        )
                        expected_action = (
                            temporal_residual(
                                weights,
                                ball_relative_xyz_m=(
                                    float(relative[0]),
                                    float(relative[1]),
                                    float(relative[2]),
                                ),
                                ball_vx_m_s=float(
                                    replay["pre_step_ball_linear_velocity_m_s"][frame, lane, 0]
                                ),
                                joint_position_rad=replay["pre_step_joint_position_rad"][
                                    frame, lane
                                ],
                                joint_velocity_rad_s=replay["pre_step_joint_velocity_rad_s"][
                                    frame, lane
                                ],
                            )
                            if first is None or frame <= first
                            else np.zeros(3)
                        )
                        actual_action = replay["applied_probe_residual_rad"][frame, lane]
                        desired_nonzero = bool(np.any(expected_action))
                        if report.get("probe_joint_limits_recorded") is True:
                            expected_action = projected_probe_residual(
                                expected_action,
                                replay["predicted_baseline_joint_target_rad"][frame, lane],
                                joint_limits[lane],
                                joint_indices,
                            )
                        if desired_nonzero:
                            applied_count += 1
                        if not np.allclose(actual_action, expected_action, atol=1e-5, rtol=0):
                            raise ValueError(
                                "snapshot intervention differs from bounded causal probe"
                            )
                    applied_counts.append(applied_count)
                if report["probe_applied_frames"] != applied_counts:
                    raise ValueError("snapshot intervention application count invalid")
            elif (
                np.any(np.abs(replay["applied_probe_residual_rad"]) > 1e-8)
                or report["probe_applied_frames"] != [0] * count
            ):
                raise ValueError("inactive snapshot probe changed parent action")
        ball_error = np.linalg.norm(observed_ball - reference_ball, axis=2)
        root_error = np.linalg.norm(observed_root[:-1, :, :3] - reference_root[1:, :, :3], axis=2)
        first_equal = 0
        class_equal = 0
        for lane, row in enumerate(report["rows"]):
            source = manifest["snapshots"][start + lane]
            observed_active = np.flatnonzero(np.max(observed_force[:, lane], axis=1) > 1.0)
            reference_active = np.flatnonzero(np.max(reference_force[:, lane], axis=1) > 1.0)
            observed_first = int(observed_active[0]) if len(observed_active) else None
            reference_first = int(reference_active[0]) if len(reference_active) else None
            observed_bodies = np.flatnonzero(np.max(observed_force[:, lane], axis=0) > 1.0).tolist()
            reference_bodies = np.flatnonzero(
                np.max(reference_force[:, lane], axis=0) > 1.0
            ).tolist()
            if (
                row.get("source_report_hash") != source["source_report_hash"]
                or row.get("source_lane") != source["lane"]
                or row.get("observed_first_contact_offset") != observed_first
                or row.get("reference_first_contact_offset") != reference_first
                or row.get("observed_contact_body_indices") != observed_bodies
                or row.get("reference_contact_body_indices") != reference_bodies
                or not np.isclose(
                    row.get("precontact_max_ball_position_error_m"),
                    np.max(ball_error[:lead, lane]),
                    atol=1e-6,
                    rtol=0,
                )
                or not np.isclose(
                    row.get("precontact_max_root_position_error_m"),
                    np.max(root_error[:lead, lane]),
                    atol=1e-6,
                    rtol=0,
                )
            ):
                raise ValueError("snapshot replay report differs from physical trace")
            first_equal += observed_first == reference_first
            class_equal += observed_bodies == reference_bodies
        max_ball = float(np.max(ball_error[:lead]))
        max_root = float(np.max(root_error[:lead]))
    parent_equivalent = bool(
        not probe
        and not phase_probe
        and max_initial <= 1e-4
        and max_ball <= 0.005
        and max_root <= 0.01
        and first_equal == count
        and class_equal == count
    )
    qualified = bool(closed_loop and parent_equivalent)
    result: dict[str, Any] = {
        "schema": "rsi_isaac_first_touch_snapshot_replay_audit_v1",
        "activation_ceiling": "SIM_ONLY",
        "source_report_hash": report["report_hash"],
        "snapshot_bank_manifest_hash": bank_audit["manifest_hash"],
        "sample_count": count,
        "max_initial_state_error": max_initial,
        "max_precontact_ball_position_error_m": max_ball,
        "max_precontact_root_position_error_m": max_root,
        "first_contact_frame_equal_count": first_equal,
        "contact_body_class_equal_count": class_equal,
        "parent_replay_equivalent": parent_equivalent,
        "short_horizon_training_surrogate_qualified": qualified,
        "intervention_action_audited": probe or phase_probe,
        "phase_target_frames": phase_target,
        "shared_candidate_hash": shared_hash,
        "closed_loop_sonic": closed_loop,
        "intervention_training_qualified": False,
        "future_parent_target_is_privileged": True,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--snapshot-bank", required=True, type=Path)
    parser.add_argument("--shared-candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("immutable replay audit output already exists")
    result = audit_snapshot_replay(
        args.folder, snapshot_bank=args.snapshot_bank, candidate_path=args.shared_candidate
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
