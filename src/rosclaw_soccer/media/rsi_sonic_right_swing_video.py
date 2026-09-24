"""Evidence-downstream before/after film of a fresh right-foot swing exam."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import IO, Any

import numpy as np

from rosclaw_soccer.media.rsi_sonic_contact_holdout_video import _showcase_camera
from rosclaw_soccer.media.trajectory_render import (
    escape_filtergraph_option,
    sample_g1_ball_trajectory,
)
from rosclaw_soccer.rsi.verify_sonic_right_swing_holdout import COURSES, verify
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def render(
    *, root: Path, stadium_assets: Path, probe_source: Path, output: Path, resolution: str
) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    verification = verify(root, stadium_assets=stadium_assets, probe_source=probe_source)
    checkout = Path(__file__).resolve().parents[3]
    output = output.expanduser().resolve()
    width, height = {"720p": (1280, 720), "1080p": (1920, 1080)}[resolution]
    if (
        checkout in output.parents
        or output.exists()
        or output.with_suffix(".json").exists()
        or output.suffix.lower() != ".mp4"
    ):
        raise ValueError("new external MP4 path required")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe required")
    import mujoco

    from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
    from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

    model = build_g1_stadium_model(
        stadium_assets, G1TrainingGoalSpec(ball_radius_m=0.11, ball_mass_kg=0.43)
    )
    model.opt.timestep = 0.002
    physics_hash = compiled_model_hash(model)
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    renderer = mujoco.Renderer(model, width=width, height=height)
    fps = 30
    frames_per_course = 126  # 4.2 s; complete uncut run through goal capture.
    rows = []
    for y, _ in COURSES:
        for candidate in (False, True):
            mode = "candidate" if candidate else "parent"
            course = root / f"rsi-sonic-right-swing-fresh-x218-y{round(y * 1000)}-{mode}-20260924"
            report = json.loads((course / "report.json").read_text(encoding="utf-8"))
            if report["physics_hash"] != physics_hash:
                raise ValueError("film physics model differs from verified exam")
            with np.load(course / "trajectory.npz", allow_pickle=False) as archive:
                q = archive["qpos"].copy()
            rows.append(
                (
                    y,
                    candidate,
                    report,
                    {
                        "time": (np.arange(len(q), dtype=np.float64) + 1) * 0.02,
                        "pelvis_pose": q[:, :7],
                        "joint_position": q[:, 7:36],
                        "ball_pose": q[:, 36:43],
                    },
                )
            )
    filters = [
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        "text=ROSCLAW SOCCER | RIGHT-FOOT PHYSICAL LEARNING:fontsize=29:fontcolor=white:"
        "x=35:y=30:box=1:boxcolor=black@0.48:boxborderw=11",
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
        "text=SIM ONLY | 4 PAIRED FRESH COURSES | FOOT-FIRST GOALS | NOT PROMOTED:"
        "fontsize=18:fontcolor=white:x=35:y=h-55:box=1:boxcolor=black@0.48:boxborderw=9",
    ]
    for index, (y, candidate, report, _) in enumerate(rows):
        mode = "LEARNED RIGHT-LEG RESIDUAL" if candidate else "FROZEN BODY BASELINE"
        caption = escape_filtergraph_option(
            f"PAIR {index // 2 + 1:02d} | BALL Y +{y:.3f} M | {mode}"
            f" | BALL {report['peak_ball_speed_mps']:.2f} M/S"
        )
        start = index * frames_per_course / fps
        end = (index + 1) * frames_per_course / fps
        filters.append(
            "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
            f"text={caption}:fontsize=20:fontcolor=white:x=35:y=96:"
            "box=1:boxcolor=black@0.48:boxborderw=9:"
            f"enable='between(t,{start:.3f},{end:.3f})'"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
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
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        if process.stdin is None:
            raise RuntimeError("video encoder pipe unavailable")
        stream: IO[bytes] = process.stdin
        for _, _, _, trajectory in rows:
            for frame in range(frames_per_course):
                _, pelvis, joints, ball = sample_g1_ball_trajectory(trajectory, frame / fps)
                data.qpos[:7], data.qpos[7:36], data.qpos[36:43] = pelvis, joints, ball
                mujoco.mj_forward(model, data)
                renderer.update_scene(
                    data,
                    camera=_showcase_camera(
                        mujoco,
                        "follow",
                        np.asarray(pelvis, dtype=np.float64),
                        np.asarray(ball, dtype=np.float64),
                    ),
                )
                stream.write(np.ascontiguousarray(renderer.render()).tobytes())
        stream.close()
        error = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        if process.wait() != 0:
            raise RuntimeError(f"video encoder failed: {error[-500:]}")
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
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream_info = json.loads(probe.stdout)["streams"][0]
    if (
        int(stream_info["width"]) != width
        or int(stream_info["height"]) != height
        or int(stream_info["nb_frames"]) != len(rows) * frames_per_course
    ):
        raise RuntimeError("encoded film dimensions or frame count differ")
    manifest: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.sonic_right_swing_video.v1",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "verification_hash": verification["verification_hash"],
        "width": width,
        "height": height,
        "fps": fps,
        "frames": len(rows) * frames_per_course,
        "duration_sec": len(rows) * frames_per_course / fps,
    }
    manifest["manifest_hash"] = hash_json(manifest)
    output.with_suffix(".json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--stadium-assets", required=True, type=Path)
    parser.add_argument("--probe-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resolution", choices=("720p", "1080p"), default="1080p")
    print(json.dumps(render(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
