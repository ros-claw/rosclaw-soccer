"""Render a 4-course verified FRESH physical exam as a SIM-only development film."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import IO, Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.media.rsi_sonic_ball_goal_video import Clip, _camera, _timeline
from rosclaw_soccer.media.trajectory_render import (
    escape_filtergraph_option,
    sample_g1_ball_trajectory,
)
from rosclaw_soccer.rsi.verify_sonic_contact_holdout import verify_holdout
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _showcase_camera(
    mujoco: Any, view: str, pelvis: NDArray[np.float64], ball: NDArray[np.float64]
) -> Any:
    """Keep athlete, ball and goal in the continuous evidence shot."""

    if view != "follow":
        return _camera(mujoco, view, pelvis, ball)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (3.10, 0.15, 0.62)
    camera.distance = 6.15
    camera.azimuth = 118.0
    camera.elevation = -10.0
    return camera


def render(
    *,
    holdout: Path,
    stadium_assets: Path,
    output: Path,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    verification = verify_holdout(holdout, stadium_assets=stadium_assets)
    if (
        not verification["contact_independently_reconstructed"]
        or verification["candidate_foot_goals"] < 2
        or verification["physical_execution_count"] != 8
    ):
        raise ValueError("at least two independently verified fresh foot goals required")
    ball_x_values = {float(row["ball_xy_m"][0]) for row in verification["results"]}
    if len(ball_x_values) != 1:
        raise ValueError("one physical ball-distance course per film required")
    ball_x_m = ball_x_values.pop()
    goal_y_values = tuple(
        float(row["ball_xy_m"][1])
        for row in verification["results"]
        if row["candidate"] and row["foot_first_goal"]
    )
    resolved = output.expanduser().resolve()
    checkout = Path(__file__).resolve().parents[3]
    if (
        resolved.exists()
        or resolved.with_suffix(".json").exists()
        or checkout in resolved.parents
        or resolved.suffix.lower() != ".mp4"
        or (width, height) not in ((1280, 720), (1920, 1080))
        or fps != 30
    ):
        raise ValueError("new external 720p/1080p MP4 path required")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe required")
    import mujoco

    from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
    from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

    model = build_g1_stadium_model(
        stadium_assets, G1TrainingGoalSpec(ball_radius_m=0.11, ball_mass_kg=0.43)
    )
    model.opt.timestep = 0.002
    video_model_hash = compiled_model_hash(model)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    data = mujoco.MjData(model)
    trajectories: dict[float, dict[str, NDArray[np.float64]]] = {}
    shots: list[tuple[float, Clip]] = []
    report_hashes = []
    for y in goal_y_values:
        course = holdout / f"x{round(1000 * ball_x_m):04d}-y{round(1000 * y):04d}-candidate"
        report = json.loads((course / "report.json").read_text(encoding="utf-8"))
        if video_model_hash != report["physics_hash"]:
            raise ValueError("video world differs from holdout physical evidence")
        with np.load(course / "trajectory.npz", allow_pickle=False) as archive:
            q = archive["qpos"].copy()
        trajectory = {
            "time": (np.arange(len(q), dtype=np.float64) + 1) * 0.02,
            "pelvis_pose": q[:, :7],
            "joint_position": q[:, 7:36],
            "ball_pose": q[:, 36:43],
        }
        trajectories[y] = trajectory
        report_hashes.append(hash_bytes((course / "report.json").read_bytes()))
        contact_sec = float(report["first_robot_ball_contact"]["time_sec"])
        goal_sec = float(report["goal_frame"]) * 0.02
        if y == goal_y_values[0]:
            shots.append((y, Clip("opening", 0.0, 0.0, 0.0, "wide")))
        shots.extend(
            (
                (y, Clip("continuous_run_and_goal", 0.0, goal_sec + 0.5, 1.0, "follow")),
                (
                    y,
                    Clip(
                        "foot_contact_slow",
                        max(0.0, contact_sec - 0.22),
                        contact_sec + 0.45,
                        0.55,
                        "contact",
                    ),
                ),
                (
                    y,
                    Clip("goal_slow", goal_sec - 0.35, goal_sec + 0.30, 0.65, "goal"),
                ),
            )
        )
        if y == goal_y_values[-1]:
            shots.append((y, Clip("closing", 6.0, 6.0, 0.0, "wide")))
    timelines = [_timeline(clip, fps) for _, clip in shots]
    frame_count = sum(len(timeline) for timeline in timelines)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    caption = escape_filtergraph_option(
        "SIM ONLY | FRESH "
        f"{verification['candidate_foot_goals']}/4 VS PARENT {verification['parent_foot_goals']}/4"
        " | FOOT-FIRST GOALS | NOT PROMOTED"
    )
    results = {
        (float(row["ball_xy_m"][1]), bool(row["candidate"])): row for row in verification["results"]
    }
    filters = [
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        "text=ROSCLAW SOCCER | PHYSICAL RSI:fontsize=30:fontcolor=white:x=38:y=30:"
        "box=1:boxcolor=black@0.42:boxborderw=12",
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
        f"text={caption}:fontsize=18:fontcolor=white:x=38:y=h-58:"
        "box=1:boxcolor=black@0.42:boxborderw=9",
    ]
    frame_offset = 0
    for index, y in enumerate(goal_y_values, start=1):
        course_frames = sum(
            len(timeline)
            for (course_y, _), timeline in zip(shots, timelines, strict=True)
            if course_y == y
        )
        parent = results[(y, False)]
        candidate = results[(y, True)]
        parent_label = "PARENT GOAL" if parent["foot_first_goal"] else "PARENT MISS"
        label = escape_filtergraph_option(
            f"SHOT {index:02d} | BALL {ball_x_m:.2f} M / Y +{y:.2f} M | {parent_label}"
            f" | CANDIDATE {candidate['peak_ball_speed_mps']:.2f} M/S"
        )
        start_sec = frame_offset / fps
        end_sec = (frame_offset + course_frames) / fps
        filters.append(
            "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
            f"text={label}:fontsize=19:fontcolor=white:x=w-tw-38:y=106:"
            "box=1:boxcolor=black@0.42:boxborderw=9:"
            f"enable='between(t,{start_sec:.3f},{end_sec:.3f})'"
        )
        frame_offset += course_frames
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-vf",
        ",".join(filters),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(resolved),
    ]
    renderer = mujoco.Renderer(model, width=width, height=height)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        if process.stdin is None:
            raise RuntimeError("ffmpeg pipe unavailable")
        ffmpeg_input: IO[bytes] = process.stdin
        for (y, clip), timeline in zip(shots, timelines, strict=True):
            trajectory = trajectories[y]
            for simulation_time in timeline:
                _, pelvis, joints, ball = sample_g1_ball_trajectory(trajectory, simulation_time)
                data.qpos[:7] = pelvis
                data.qpos[7:36] = joints
                data.qpos[36:43] = ball
                mujoco.mj_forward(model, data)
                renderer.update_scene(
                    data,
                    camera=_showcase_camera(
                        mujoco,
                        clip.view,
                        np.asarray(pelvis, dtype=np.float64),
                        np.asarray(ball, dtype=np.float64),
                    ),
                )
                ffmpeg_input.write(np.ascontiguousarray(renderer.render()).tobytes())
        ffmpeg_input.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg encoding failed: {stderr[-2000:]}")
    except BaseException:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        raise
    finally:
        renderer.close()
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames",
            "-of",
            "json",
            str(resolved),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    if (
        int(stream["width"]) != width
        or int(stream["height"]) != height
        or int(stream["nb_frames"]) != frame_count
    ):
        raise RuntimeError("encoded video dimensions or frame count differ")
    manifest = {
        "schema": "rosclaw_soccer.rsi.sonic_contact_holdout_video.v1",
        "video_path": str(resolved),
        "video_hash": hash_bytes(resolved.read_bytes()),
        "holdout_verification_hash": verification["verification_hash"],
        "source_report_hashes": report_hashes,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps,
        "shots": [{"ball_x_m": ball_x_m, "ball_y_m": y, **asdict(clip)} for y, clip in shots],
        "pixels_used_for_scoring": False,
        "visualization_only": True,
        "promotion_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    manifest["manifest_hash"] = hash_json(manifest)
    with resolved.with_suffix(".json").open("x", encoding="utf-8") as stream_file:
        json.dump(manifest, stream_file, sort_keys=True, indent=2, allow_nan=False)
        stream_file.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--stadium-assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resolution", choices=("720p", "1080p"), default="720p")
    args = parser.parse_args()
    width, height = (1280, 720) if args.resolution == "720p" else (1920, 1080)
    print(
        json.dumps(
            render(
                holdout=args.holdout,
                stadium_assets=args.stadium_assets,
                output=args.output,
                width=width,
                height=height,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
