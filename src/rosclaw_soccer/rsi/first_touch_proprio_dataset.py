"""Build episode-split, precontact-only SIM_ONLY G1 ball/body learning data.

The ball trace is sampled after each 20 ms control frame, while the body trace
is sampled before it. Shift ball samples by one frame before joining; the
unshifted arrays silently leak future physics into a purported observation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_vector_first_touch
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

LEG_JOINT_INDICES = (0, 3, 4, 6, 9, 10)
FEATURE_NAMES = (
    "ball_relative_x_m",
    "ball_relative_y_m",
    "ball_relative_z_m",
    "ball_velocity_x_m_s",
    "ball_velocity_y_m_s",
    *(f"root_quaternion_xyzw_{axis}" for axis in "xyzw"),
    *(f"root_velocity_{axis}" for axis in ("vx", "vy", "vz", "wx", "wy", "wz")),
    *(f"leg_joint_position_{index}_rad" for index in LEG_JOINT_INDICES),
    *(f"leg_joint_velocity_{index}_rad_s" for index in LEG_JOINT_INDICES),
    "navigation_speed_mps",
)


def aligned_episode_features(
    *,
    ball_position_after_step: np.ndarray,
    initial_ball_xyz_m: tuple[float, float, float],
    initial_ball_vx_m_s: float,
    root_pose_xyzw_m: np.ndarray,
    root_velocity_world: np.ndarray,
    joint_position_rad: np.ndarray,
    joint_velocity_rad_s: np.ndarray,
    navigation_speed_mps: np.ndarray,
) -> np.ndarray:
    """Align observations at frame start, without future ball or reward leakage."""
    frames = len(ball_position_after_step)
    if (
        frames < 2
        or ball_position_after_step.shape != (frames, 3)
        or root_pose_xyzw_m.shape != (frames, 7)
        or root_velocity_world.shape != (frames, 6)
        or any(array.shape != (frames, 29) for array in (joint_position_rad, joint_velocity_rad_s))
        or navigation_speed_mps.shape != (frames,)
        or not all(
            np.isfinite(array).all()
            for array in (
                ball_position_after_step,
                root_pose_xyzw_m,
                root_velocity_world,
                joint_position_rad,
                joint_velocity_rad_s,
                navigation_speed_mps,
            )
        )
        or not np.isfinite(initial_ball_vx_m_s)
        or not np.isfinite(initial_ball_xyz_m).all()
    ):
        raise ValueError("invalid aligned body/ball episode")
    ball_before = np.concatenate(
        (np.asarray(initial_ball_xyz_m, dtype=np.float64)[None, :], ball_position_after_step[:-1])
    )
    velocity_xy = np.empty((frames, 2), dtype=np.float64)
    velocity_xy[0] = (initial_ball_vx_m_s, 0.0)
    velocity_xy[1:] = (ball_before[1:, :2] - ball_before[:-1, :2]) / 0.02
    result: np.ndarray = np.concatenate(
        (
            ball_before - root_pose_xyzw_m[:, :3],
            velocity_xy,
            root_pose_xyzw_m[:, 3:7],
            root_velocity_world,
            joint_position_rad[:, LEG_JOINT_INDICES],
            joint_velocity_rad_s[:, LEG_JOINT_INDICES],
            navigation_speed_mps[:, None],
        ),
        axis=1,
    ).astype(np.float32)
    if result.shape != (frames, len(FEATURE_NAMES)) or not np.isfinite(result).all():
        raise ValueError("nonfinite or changed proprioceptive feature contract")
    return result


def build_dataset(folders: tuple[Path, ...], output_dir: Path) -> dict[str, Any]:
    """Persist audited train-only episodes; never split correlated frames as samples."""
    if len(folders) < 2 or output_dir.exists() or len(set(folders)) != len(folders):
        raise ValueError("at least two distinct sources and a new output directory required")
    episodes: list[np.ndarray] = []
    action_targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    first_frames: list[int] = []
    seeds: list[int] = []
    lanes: list[int] = []
    sources = []
    foundation: tuple[str, str, int] | None = None
    for folder in folders:
        audit = audit_vector_first_touch(folder)
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        seed = report.get("training_course_seed")
        if (
            type(seed) is not int
            or seed in seeds
            or "body_trace_hash" not in report
            or report.get("navigation_speed_mps", 1.4) != 1.4
            or "near_ball_gap_m" in report
            or report.get("torch_batch_plan_only") is not True
        ):
            raise ValueError("dataset requires distinct frozen 1.4 m/s train-only body traces")
        current_foundation = (
            report["asset_hash"],
            report["sonic_qualification_hash"],
            report["frames"],
        )
        if foundation is not None and foundation != current_foundation:
            raise ValueError("dataset foundation, asset or horizon changed")
        foundation = current_foundation
        with (
            np.load(folder / "trace.npz", allow_pickle=False) as physics,
            np.load(folder / "body_trace.npz", allow_pickle=False) as body,
        ):
            ball = physics["ball_position_m"]
            for i, row in enumerate(report["environments"]):
                course = row["course"]
                episode = aligned_episode_features(
                    ball_position_after_step=ball[:, i],
                    initial_ball_xyz_m=(
                        course["ball_x_m"],
                        row["lane_y_m"] + course["ball_y_local_m"],
                        0.13,
                    ),
                    initial_ball_vx_m_s=course["ball_vx_m_s"],
                    root_pose_xyzw_m=body["root_pose_xyzw_m"][:, i],
                    root_velocity_world=body["root_velocity_world"][:, i],
                    joint_position_rad=body["joint_position_rad"][:, i],
                    joint_velocity_rad_s=body["joint_velocity_rad_s"][:, i],
                    navigation_speed_mps=body["navigation_speed_mps"][:, i],
                )
                first = row["first_contact_frame"]
                bodies = set(row["contact_body_indices"])
                episodes.append(episode)
                action_targets.append(body["joint_target_rad"][:, i, LEG_JOINT_INDICES])
                masks.append(
                    np.arange(report["frames"]) < (first if first is not None else report["frames"])
                )
                labels.append(1 if bodies and bodies <= {0, 1} else 0 if bodies else -1)
                first_frames.append(first if first is not None else -1)
                seeds.append(seed)
                lanes.append(i)
        sources.append(
            {
                "training_course_seed": seed,
                "report_hash": report["report_hash"],
                "audit_hash": audit["report_hash"],
                "body_trace_hash": report["body_trace_hash"],
            }
        )
    if len({source["training_course_seed"] for source in sources}) != len(sources):
        raise ValueError("training course seed repeated")
    feature_array = np.stack(episodes)
    mask_array = np.stack(masks)
    if not np.all(mask_array[:, 0]) or not np.isfinite(feature_array).all():
        raise ValueError("invalid precontact masks or features")
    output_dir.mkdir(parents=True, exist_ok=False)
    data_path = output_dir / "episodes.npz"
    np.savez_compressed(
        data_path,
        features=feature_array,
        action_target_rad=np.stack(action_targets).astype(np.float32),
        precontact_mask=mask_array,
        contact_label=np.asarray(labels, dtype=np.int8),
        first_contact_frame=np.asarray(first_frames, dtype=np.int32),
        training_course_seed=np.asarray(seeds, dtype=np.int64),
        environment=np.asarray(lanes, dtype=np.int32),
    )
    manifest = {
        "schema": "rsi_isaac_first_touch_proprio_dataset_v1",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "fresh_opened": False,
        "feature_names": list(FEATURE_NAMES),
        "action_joint_indices": list(LEG_JOINT_INDICES),
        "policy_input_excludes_current_action": True,
        "feature_alignment": (
            "body and ball both before frame step; ball shifted from post-step trace"
        ),
        "episode_count": len(episodes),
        "clean_foot_episode_count": labels.count(1),
        "nonfoot_contact_episode_count": labels.count(0),
        "no_contact_episode_count": labels.count(-1),
        "source_count": len(sources),
        "sources": sources,
        "builder_source_hash": hash_bytes(Path(__file__).read_bytes()),
        "data_hash": hash_bytes(data_path.read_bytes()),
    }
    manifest["manifest_hash"] = hash_json(manifest)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_dataset(tuple(args.folders), args.output_dir)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
