"""Evidence-downstream cinematic replay of a verified frozen-SONIC goal."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.media.trajectory_render import sample_g1_ball_trajectory
from rosclaw_soccer.rsi.verify_sonic_ball_goal import verify_sonic_ball_goal
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


@dataclass(frozen=True)
class Clip:
    name: str
    start_sec: float
    end_sec: float
    speed: float
    view: str


CLIPS = (
    Clip("opening", 0.0, 0.8, 0.0, "wide"),
    Clip("continuous_run_and_goal", 0.0, 4.6, 1.0, "follow"),
    Clip("foot_ball_contact_slow", 1.15, 1.95, 0.40, "contact"),
    Clip("whole_ball_goal_slow", 3.25, 4.30, 0.55, "goal"),
    Clip("recovery", 4.3, 5.98, 1.0, "recovery"),
    Clip("closing", 5.98, 5.98, 0.0, "wide"),
)


def _timeline(clip: Clip, fps: int) -> NDArray[np.float64]:
    if clip.speed == 0.0:
        seconds = 0.8 if clip.name == "opening" else 1.0
        return np.full(round(seconds * fps), clip.start_sec)
    duration = (clip.end_sec - clip.start_sec) / clip.speed
    count = max(1, round(duration * fps))
    return clip.start_sec + np.arange(count, dtype=np.float64) * clip.speed / fps


def _camera(mujoco: Any, view: str, pelvis: NDArray[np.float64], ball: NDArray[np.float64]) -> Any:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    if view == "wide":
        camera.lookat[:] = (2.5, 0.0, 0.58)
        camera.distance, camera.azimuth, camera.elevation = 8.0, 100.0, -10.0
    elif view == "follow":
        camera.lookat[:] = (min(3.3, 0.6 * pelvis[0] + 0.4 * ball[0]), 0.25, 0.6)
        camera.distance, camera.azimuth, camera.elevation = 5.2, 105.0, -9.0
    elif view == "contact":
        camera.lookat[:] = (ball[0], ball[1], 0.48)
        camera.distance, camera.azimuth, camera.elevation = 3.35, 90.0, -6.0
    elif view == "goal":
        camera.lookat[:] = (4.65, 0.28, 0.56)
        camera.distance, camera.azimuth, camera.elevation = 5.4, 145.0, -8.0
    elif view == "recovery":
        camera.lookat[:] = (pelvis[0], pelvis[1], 0.67)
        camera.distance, camera.azimuth, camera.elevation = 3.9, 108.0, -8.0
    else:
        raise ValueError("unknown evidence camera")
    return camera


def render(
    *,
    primary: Path,
    replay: Path,
    stadium_assets: Path,
    output: Path,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> dict[str, Any]:
    verification = verify_sonic_ball_goal(primary, replay)
    if not verification["strict_replay"] or not verification["whole_ball_goal_crossed"]:
        raise ValueError("video requires a physically verified goal and strict replay")
    checkout = Path(__file__).resolve().parents[3]
    resolved = output.expanduser().resolve()
    manifest_path = resolved.with_suffix(".json")
    if (
        checkout in resolved.parents
        or resolved.exists()
        or manifest_path.exists()
        or resolved.suffix.lower() != ".mp4"
        or fps not in (30, 60)
        or (width, height) not in ((1280, 720), (1920, 1080))
    ):
        raise ValueError("new external 720p/1080p MP4 output required")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required")
    report = json.loads((primary / "report.json").read_text(encoding="utf-8"))
    with np.load(primary / "trajectory.npz", allow_pickle=False) as archive:
        q = archive["qpos"].copy()
    trajectory = {
        "time": (np.arange(len(q), dtype=np.float64) + 1) * 0.02,
        "pelvis_pose": q[:, :7],
        "joint_position": q[:, 7:36],
        "ball_pose": q[:, 36:43],
    }
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
    from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

    model = build_g1_stadium_model(
        stadium_assets, G1TrainingGoalSpec(ball_radius_m=0.11, ball_mass_kg=0.43)
    )
    model.opt.timestep = 0.002
    if compiled_model_hash(model) != report["physics_hash"]:
        raise ValueError("video model differs from physical goal evidence")
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    data = mujoco.MjData(model)
    timelines = tuple(_timeline(clip, fps) for clip in CLIPS)
    count = sum(len(timeline) for timeline in timelines)
    resolved.parent.mkdir(parents=True, exist_ok=True)
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
            raise RuntimeError("ffmpeg rawvideo pipe unavailable")
        ffmpeg_input: IO[bytes] = process.stdin
        for clip, timeline in zip(CLIPS, timelines, strict=True):
            for simulation_time in timeline:
                _, pelvis, joints, ball = sample_g1_ball_trajectory(trajectory, simulation_time)
                data.qpos[:7] = pelvis
                data.qpos[7:36] = joints
                data.qpos[36:43] = ball
                mujoco.mj_forward(model, data)
                renderer.update_scene(
                    data,
                    camera=_camera(
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
            "stream=width,height,nb_frames,r_frame_rate",
            "-of",
            "json",
            str(resolved),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream_info = json.loads(probe.stdout)["streams"][0]
    if (
        stream_info["width"] != width
        or stream_info["height"] != height
        or int(stream_info["nb_frames"]) != count
    ):
        raise RuntimeError("encoded video dimensions or frame count differ from manifest")
    manifest: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.sonic_ball_goal_video.v1",
        "video_path": str(resolved),
        "video_hash": hash_bytes(resolved.read_bytes()),
        "primary_report_hash": hash_bytes((primary / "report.json").read_bytes()),
        "replay_report_hash": hash_bytes((replay / "report.json").read_bytes()),
        "verification_hash": verification["verification_hash"],
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": count,
        "duration_sec": count / fps,
        "clips": [asdict(clip) for clip in CLIPS],
        "training_goal_width_m": 2.4,
        "training_goal_height_m": 1.6,
        "frozen_sonic_parent_only": True,
        "learned_contact_actor": False,
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    manifest["manifest_hash"] = hash_json(manifest)
    with manifest_path.open("x", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, sort_keys=True, indent=2, allow_nan=False)
        manifest_file.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--stadium-assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resolution", choices=("720p", "1080p"), default="720p")
    args = parser.parse_args()
    width, height = (1280, 720) if args.resolution == "720p" else (1920, 1080)
    print(
        json.dumps(
            render(
                primary=args.primary,
                replay=args.replay,
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
