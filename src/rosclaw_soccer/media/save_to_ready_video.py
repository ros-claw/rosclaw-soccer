"""Evidence-bound reel for the strict save-to-ready successor portfolio."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from rosclaw_soccer.media.three_role_save_portfolio_video import (
    _Clip,
    _ffmpeg_command,
    _Frame,
    _probe,
    _segment,
    _write_frames,
    _write_labels,
)
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.dynamic_corner_save import validate_dynamic_corner_evidence
from rosclaw_soccer.training.save_to_ready_successor import (
    validate_save_to_ready_successor_evidence,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_three_player_stadium_model

_CLAIM = "STRICT_SAVE_ABSORB_REENGAGE_GOALKEEPER_READY_PORTFOLIO"


def validate_save_to_ready_video_manifest(path: Path) -> dict[str, Any]:
    """Validate video bytes, source trajectories and SIM_ONLY authority."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("save-to-ready video manifest must be an object")
    expected = payload.pop("manifest_hash", None)
    if expected != hash_json(payload):
        raise ValueError("save-to-ready video manifest integrity mismatch")
    video_value = payload.get("video_path")
    source_files = payload.get("source_files")
    if not isinstance(video_value, str) or not isinstance(source_files, dict):
        raise ValueError("save-to-ready video bindings are invalid")
    video = Path(video_value).expanduser().resolve()
    if not video.is_file() or hash_bytes(video.read_bytes()) != payload.get("video_hash"):
        raise ValueError("save-to-ready video hash changed")
    for source_value, source_hash in source_files.items():
        source = Path(source_value).expanduser().resolve()
        if not source.is_file() or hash_bytes(source.read_bytes()) != source_hash:
            raise ValueError("save-to-ready video source binding changed")
    numbers = tuple(
        payload.get(name) for name in ("fps", "width", "height", "frame_count", "duration_sec")
    )
    if (
        payload.get("schema_version") != "rosclaw_soccer.save_to_ready_video.v1"
        or payload.get("claim") != _CLAIM
        or payload.get("case_count") != 4
        or payload.get("strict_replay") is not True
        or payload.get("all_post_probe_goalkeeper_ready") is not True
        or payload.get("visualization_only") is not True
        or payload.get("pixels_used_for_scoring") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
        or payload.get("commercial_use_allowed") is not False
        or any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in numbers
        )
    ):
        raise ValueError("save-to-ready video authority contract is invalid")
    payload["manifest_hash"] = expected
    return cast(dict[str, Any], payload)


def render_save_to_ready_video(
    *,
    evidence_path: Path,
    parent_evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 60,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render save, active recovery, new command and readiness for four lanes."""

    evidence_file = evidence_path.expanduser().resolve()
    parent_file = parent_evidence_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    manifest_path = output.with_suffix(".json")
    if (
        output.exists()
        or manifest_path.exists()
        or output.suffix.lower() != ".mp4"
        or output == checkout
        or checkout in output.parents
        or not 20 <= fps <= 60
        or not 1280 <= width <= 3840
        or not 720 <= height <= 2160
    ):
        raise ValueError("save-to-ready video output contract is invalid")
    evidence = validate_save_to_ready_successor_evidence(evidence_file)
    parent = validate_dynamic_corner_evidence(parent_file)
    artifacts = cast(dict[str, Any], evidence["artifacts"])
    if hash_bytes(parent_file.read_bytes()) != artifacts.get(
        "parent_evidence_file_hash"
    ) or parent.get("report_hash") != artifacts.get("parent_report_hash"):
        raise ValueError("save-to-ready parent evidence binding changed")
    cases = cast(dict[str, dict[str, Any]], evidence["cases"])
    request_path = evidence_file.parent / "request.json"
    parent_request_path = parent_file.parent / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    parent_request = json.loads(parent_request_path.read_text(encoding="utf-8"))
    goal_specs = parent_request.get("lane_goal_specs")
    if not isinstance(goal_specs, dict) or set(goal_specs) != set(cases):
        raise ValueError("save-to-ready goal contracts are incomplete")
    trajectories: dict[str, dict[str, np.ndarray]] = {}
    goals: dict[str, G1TrainingGoalSpec] = {}
    source_files = {
        str(evidence_file): hash_bytes(evidence_file.read_bytes()),
        str(request_path): hash_bytes(request_path.read_bytes()),
        str(parent_file): hash_bytes(parent_file.read_bytes()),
        str(parent_request_path): hash_bytes(parent_request_path.read_bytes()),
    }
    required = {
        "time",
        "ball_pose",
        "passer_pelvis_pose",
        "passer_joint_position",
        "shooter_pelvis_pose",
        "shooter_joint_position",
        "goalkeeper_pelvis_pose",
        "goalkeeper_joint_position",
        "goalkeeper_foot_contact",
    }
    for lane_id, case in cases.items():
        name = case.get("trajectory_file")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("save-to-ready trajectory name is invalid")
        trajectory_path = evidence_file.parent / name
        with np.load(trajectory_path, allow_pickle=False) as archive:
            trajectory = {key: np.asarray(archive[key]) for key in archive.files}
        if not required <= set(trajectory):
            raise ValueError("save-to-ready trajectory is incomplete")
        goal = G1TrainingGoalSpec(**goal_specs[lane_id])
        if abs(goal.width_m - 7.32) > 1.0e-9 or abs(goal.height_m - 2.44) > 1.0e-9:
            raise ValueError("save-to-ready video requires a regulation goal")
        trajectories[lane_id] = trajectory
        goals[lane_id] = goal
        source_files[str(trajectory_path)] = hash_bytes(trajectory_path.read_bytes())
    clips = _timeline(cases, trajectories, fps)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for save-to-ready video")
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        if qualification.body_hash != request.get("body_hash"):
            raise ValueError("save-to-ready Body hash changed")
        import mujoco

        first_lane = next(iter(cases))
        goal = goals[first_lane]
        model = build_g1_three_player_stadium_model(
            asset_root.expanduser().resolve(),
            passer_origin_m=(5.10, -0.16406006503921598, 0.0),
            goalkeeper_origin_m=(goal.plane_x_m - 0.48, 0.0, 0.0),
            spec=goal,
        )
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-save-to-ready-") as temp:
                labels = _write_labels(Path(temp), clips)
                process = subprocess.Popen(
                    _ffmpeg_command(
                        ffmpeg=ffmpeg,
                        output=output,
                        width=width,
                        height=height,
                        fps=fps,
                        clips=clips,
                        labels=labels,
                    ),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                if process.stdin is None:
                    raise RuntimeError("save-to-ready raw-video pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        renderer=renderer,
                        trajectories=trajectories,
                        clips=clips,
                        goal=goal,
                        stream=cast(BinaryIO, process.stdin),
                    )
                except BaseException:
                    process.stdin.close()
                    process.kill()
                    process.wait()
                    raise
                process.stdin.close()
                stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
                if process.wait():
                    raise RuntimeError(f"save-to-ready ffmpeg failed: {stderr[-3000:]}")
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl
    video_probe = _probe(ffprobe, output)
    frame_count = sum(len(clip.frames) for clip in clips)
    if (
        video_probe["width"] != width
        or video_probe["height"] != height
        or video_probe["fps"] != fps
        or abs(video_probe["frame_count"] - frame_count) > 1
    ):
        raise RuntimeError("save-to-ready encoded video contract changed")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.save_to_ready_video.v1",
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_files": source_files,
        "claim": _CLAIM,
        "strict_replay": True,
        "case_count": len(cases),
        "all_post_probe_goalkeeper_ready": evidence["portfolio_gates"][
            "all_post_probe_goalkeeper_ready"
        ],
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps,
        "clips": [{"label": clip.label, "frame_count": len(clip.frames)} for clip in clips],
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
    }
    manifest["manifest_hash"] = hash_json(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    validate_save_to_ready_video_manifest(manifest_path)
    return manifest


def _timeline(
    cases: dict[str, dict[str, Any]],
    trajectories: dict[str, dict[str, np.ndarray]],
    fps: int,
) -> tuple[_Clip, ...]:
    clips: list[_Clip] = []
    first_lane = next(iter(cases))
    start = float(trajectories[first_lane]["time"][0])
    clips.append(
        _Clip(
            "S104 SAVE → ABSORB → RE-ENGAGE → GOALKEEPER READY · 4 STRICT LANES",
            tuple(_Frame(first_lane, start, "wide_goal") for _ in range(round(1.5 * fps))),
        )
    )
    for index, (lane_id, case) in enumerate(cases.items(), start=1):
        result = cast(dict[str, Any], case["result"])
        takeoff = cast(dict[str, Any], case["takeoff"])
        takeoff_metrics = cast(dict[str, Any], takeoff["metrics"])
        successor = cast(dict[str, Any], case["successor"])
        probe = cast(dict[str, Any], successor["probe"])
        probe_metrics = cast(dict[str, Any], probe["metrics"])
        pre_metrics = cast(dict[str, Any], successor["pre_probe_ready"]["metrics"])
        post_metrics = cast(dict[str, Any], successor["post_probe_ready"]["metrics"])
        shot_time = float(result["shot_contact_time_sec"])
        save_time = float(result["goalkeeper_glove_contact_time_sec"])
        landing_time = float(takeoff_metrics["landing_time_sec"])
        probe_start = float(probe_metrics["start_sec"])
        probe_stop = float(probe_metrics["stop_sec"])
        end = float(trajectories[lane_id]["time"][-1])
        side = str(result["goalkeeper_glove_contact_side"]).upper()
        command = float(probe_metrics["expected_command_mps"])
        displacement = float(probe_metrics["signed_displacement_m"])
        pre_linear_speed = float(pre_metrics["maximum_root_linear_speed_mps"])
        post_support = float(post_metrics["bilateral_support_fraction"])
        post_angular_speed = float(post_metrics["maximum_root_angular_speed_rad_s"])
        clips.append(
            _Clip(
                f"LANE {index}/4 · LIVE STRIKE → {side} GLOVE SAVE · NO RESET",
                _segment(lane_id, shot_time - 0.45, save_time + 0.28, 0.85, "chain", fps),
            )
        )
        clips.append(
            _Clip(
                f"TRUE AIRBORNE CONTACT → FOOT LANDING · t={landing_time:.2f}s",
                _segment(lane_id, save_time - 0.14, landing_time + 0.30, 0.30, "hero", fps),
            )
        )
        clips.append(
            _Clip(
                f"ABSORB + WALK OFF IMPULSE · PRE-READY {pre_linear_speed:.4f} m/s",
                _segment(lane_id, save_time + 0.35, probe_start - 0.10, 4.0, "keeper_front", fps),
            )
        )
        clips.append(
            _Clip(
                f"NEW LATERAL COMMAND {command:+.2f} m/s → {displacement:+.3f} m RESPONSE",
                _segment(lane_id, probe_start - 0.25, probe_stop + 0.65, 0.70, "keeper_hero", fps),
            )
        )
        clips.append(
            _Clip(
                f"GOALKEEPER READY AGAIN · {post_support:.0%} DOUBLE SUPPORT · "
                f"ω={post_angular_speed:.4f} rad/s",
                _segment(lane_id, end - 1.0, end, 1.0, "keeper_front", fps),
            )
        )
    last_lane = next(reversed(cases))
    end = float(trajectories[last_lane]["time"][-1])
    clips.append(
        _Clip(
            "4/4 × STRICT REPLAY · SAVE IS NOT TERMINAL · THE NEXT ACTION REMAINS AVAILABLE",
            tuple(_Frame(last_lane, end, "wide_goal") for _ in range(round(2.0 * fps))),
        )
    )
    return tuple(clips)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--parent-evidence", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = render_save_to_ready_video(
        evidence_path=args.evidence,
        parent_evidence_path=args.parent_evidence,
        asset_root=args.asset_root,
        output_path=args.output,
        source_checkout=args.source_checkout,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["render_save_to_ready_video", "validate_save_to_ready_video_manifest"]
