"""Render the S101 lateral athlete CPU trajectories as a long-form reel."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from rosclaw_soccer.media.three_role_save_portfolio_video import _pose, _probe
from rosclaw_soccer.media.trajectory_render import escape_filtergraph_option
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.world.field import build_g1_stadium_model

_CLAIM = "S101_BILATERAL_LATERAL_ATHLETE_EXPERT_CPU_MUJOCO_REPLAY"


@dataclass(frozen=True)
class _Clip:
    route: dict[str, Any]
    frame_times_sec: tuple[float, ...]
    label: str


def _load_bound_json(path: Path, *, hash_key: str, label: str) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    claimed = payload.get(hash_key)
    value = dict(payload)
    value.pop(hash_key, None)
    if claimed != hash_json(value):
        raise ValueError(f"{label} content hash is invalid")
    return cast(dict[str, Any], payload)


def validate_lateral_athlete_video_manifest(path: Path) -> dict[str, Any]:
    payload = _load_bound_json(path, hash_key="manifest_hash", label="lateral athlete video")
    source_files = payload.get("source_files")
    video_value = payload.get("video_path")
    if not isinstance(source_files, dict) or not isinstance(video_value, str):
        raise ValueError("lateral athlete video bindings are invalid")
    video = Path(video_value).expanduser().resolve()
    if not video.is_file() or hash_bytes(video.read_bytes()) != payload.get("video_hash"):
        raise ValueError("lateral athlete video hash changed")
    for name, expected in source_files.items():
        source = Path(name).expanduser().resolve()
        if not source.is_file() or hash_bytes(source.read_bytes()) != expected:
            raise ValueError("lateral athlete video source changed")
    if (
        payload.get("schema_version") != "rosclaw_soccer.lateral_athlete_video.v1"
        or payload.get("claim") != _CLAIM
        or payload.get("source_evidence_passed") is not True
        or payload.get("visualization_only") is not True
        or payload.get("pixels_used_for_scoring") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
        or payload.get("commercial_use_allowed") is not False
    ):
        raise ValueError("lateral athlete video authority contract is invalid")
    return payload


def _select_routes(trajectory: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    routes = trajectory.get("candidate")
    if not isinstance(routes, list):
        raise ValueError("lateral athlete candidate trajectories are missing")
    selected: list[dict[str, Any]] = []
    for target in (-2.0, 2.0, -1.5, 1.5):
        matches = [
            row
            for row in routes
            if isinstance(row, dict)
            and math.isclose(float(row.get("target_lateral_m", math.inf)), target)
            and row.get("passed") is True
        ]
        if not matches:
            raise ValueError("lateral athlete video route is missing")
        selected.append(min(matches, key=lambda row: int(row["seed"])))
    return tuple(selected)


def _timeline(*, routes: tuple[dict[str, Any], ...], fps: int) -> tuple[_Clip, ...]:
    clips: list[_Clip] = []
    playback_rate = 1.25
    for index, route in enumerate(routes, start=1):
        settle = float(route["settled_time_sec"])
        simulation_duration = min(9.5, settle + 1.0)
        frame_count = int(math.ceil(simulation_duration / playback_rate * fps))
        times = tuple(
            min(simulation_duration, frame * playback_rate / fps) for frame in range(frame_count)
        )
        target = float(route["target_lateral_m"])
        side = "LEFT" if target < 0.0 else "RIGHT"
        clips.append(
            _Clip(
                route=route,
                frame_times_sec=times,
                label=(
                    f"RUN {index}/4 · {side} {abs(target):.1f} M · "
                    f"ERROR {100.0 * float(route['endpoint_error_m']):.2f} CM · "
                    f"SETTLE {settle:.2f} S"
                ),
            )
        )
    return tuple(clips)


def _interpolate_qpos(route: dict[str, Any], time_sec: float) -> np.ndarray:
    trajectory = route.get("trajectory")
    if not isinstance(trajectory, list) or len(trajectory) < 2:
        raise ValueError("lateral athlete qpos trajectory is missing")
    control_dt = float(trajectory[1]["time_sec"]) - float(trajectory[0]["time_sec"])
    position = max(0.0, time_sec / control_dt)
    lower = min(int(math.floor(position)), len(trajectory) - 1)
    upper = min(lower + 1, len(trajectory) - 1)
    ratio = position - lower
    left: np.ndarray = np.asarray(trajectory[lower]["qpos"], dtype=np.float64)
    right: np.ndarray = np.asarray(trajectory[upper]["qpos"], dtype=np.float64)
    if left.shape != right.shape or left.shape[0] < 43:
        raise ValueError("lateral athlete qpos shape is invalid")
    value: np.ndarray = left + ratio * (right - left)
    value[:7] = _pose(left[:7], right[:7], ratio)
    return value


def _write_frames(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    clips: tuple[_Clip, ...],
    stream: BinaryIO,
) -> None:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.lookat[:] = (4.52, 0.0, 0.72)
    camera.distance = 7.2
    camera.azimuth = 178.0
    camera.elevation = -7.5
    for clip in clips:
        target = float(clip.route["target_lateral_m"])
        camera.lookat[1] = 0.45 * target
        for time_sec in clip.frame_times_sec:
            qpos = _interpolate_qpos(clip.route, time_sec)
            data.qpos[:] = qpos
            # The regulation football is a visualization-only target marker.
            # It never changes the already-scored stored G1 trajectory.
            data.qpos[36:43] = (4.52, target, 0.115, 1.0, 0.0, 0.0, 0.0)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = np.asarray(renderer.render(), dtype=np.uint8)
            stream.write(frame.tobytes())


def _write_labels(root: Path, clips: tuple[_Clip, ...]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for index, clip in enumerate(clips):
        path = root / f"label-{index}.txt"
        path.write_text(clip.label, encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def _ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    width: int,
    height: int,
    fps: int,
    clips: tuple[_Clip, ...],
    labels: tuple[Path, ...],
) -> list[str]:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font_option = f"fontfile={escape_filtergraph_option(str(font))}:" if font.is_file() else ""
    scale = height / 720.0
    left = round(30 * scale)
    filters = [
        f"drawbox=x=0:y=0:w=iw:h={round(116 * scale)}:color=0x030711@0.84:t=fill",
        f"drawbox=x=0:y=h-{round(62 * scale)}:w=iw:h={round(62 * scale)}:"
        "color=0x030711@0.84:t=fill",
        f"drawtext={font_option}text='ROSClaw Soccer · S101 LATERAL ATHLETE EXPERT':"
        f"expansion=none:x={left}:y={round(13 * scale)}:fontsize={round(32 * scale)}:"
        "fontcolor=white",
        f"drawtext={font_option}text='4xA6000 DISTILLATION · CPU MUJOCO TRUTH · "
        "SIM ONLY · PIXELS NEVER SCORE':"
        f"expansion=none:x={left}:y=h-{round(40 * scale)}:fontsize={round(18 * scale)}:"
        "fontcolor=0x8DD8FF",
    ]
    offset = 0.0
    for label, clip in zip(labels, clips, strict=True):
        end = offset + len(clip.frame_times_sec) / fps
        filters.append(
            f"drawtext={font_option}textfile={escape_filtergraph_option(str(label))}:"
            f"expansion=none:x={left}:y={round(61 * scale)}:fontsize={round(20 * scale)}:"
            f"fontcolor=0x65F59A:enable='between(t,{offset:.6f},{end:.6f})'"
        )
        offset = end
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
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


def render_lateral_athlete_video(
    *,
    exam_report_path: Path,
    trajectory_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render four bilateral routes without giving pixels scoring authority."""

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
        raise ValueError("lateral athlete video output contract is invalid")
    exam_file = exam_report_path.expanduser().resolve()
    trajectory_file = trajectory_path.expanduser().resolve()
    exam = _load_bound_json(exam_file, hash_key="report_hash", label="lateral athlete exam")
    trajectory = _load_bound_json(
        trajectory_file, hash_key="trajectory_hash", label="lateral athlete trajectories"
    )
    if not (
        exam.get("passed") is True
        and exam.get("physics_backend") == "mujoco_cpu"
        and exam.get("activation_ceiling") == "SIM_ONLY"
        and exam.get("hardware_command_sent") is False
        and exam.get("video_used_for_scoring") is False
        and trajectory.get("exam_report_hash") == exam.get("report_hash")
        and trajectory.get("activation_ceiling") == "SIM_ONLY"
    ):
        raise ValueError("lateral athlete source evidence is not render eligible")
    routes = _select_routes(trajectory)
    clips = _timeline(routes=routes, fps=fps)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for lateral athlete video")
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        output.parent.mkdir(parents=True, exist_ok=True)
        import mujoco

        model = build_g1_stadium_model(asset_root)
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-lateral-athlete-") as temp:
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
                    raise RuntimeError("lateral athlete ffmpeg pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        renderer=renderer,
                        clips=clips,
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
                    raise RuntimeError(f"lateral athlete ffmpeg failed: {stderr[-3000:]}")
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl
    probe = _probe(ffprobe, output)
    frame_count = sum(len(clip.frame_times_sec) for clip in clips)
    if (
        probe["width"] != width
        or probe["height"] != height
        or probe["fps"] != fps
        or abs(probe["frame_count"] - frame_count) > 1
    ):
        raise RuntimeError("lateral athlete encoded video contract changed")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.lateral_athlete_video.v1",
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_files": {
            str(exam_file): hash_bytes(exam_file.read_bytes()),
            str(trajectory_file): hash_bytes(trajectory_file.read_bytes()),
        },
        "claim": _CLAIM,
        "source_evidence_passed": True,
        "case_count": len(clips),
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps,
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
    }
    manifest["manifest_hash"] = hash_json(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exam-report", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    args = parser.parse_args()
    manifest = render_lateral_athlete_video(
        exam_report_path=args.exam_report,
        trajectory_path=args.trajectory,
        asset_root=args.asset_root,
        output_path=args.output,
        source_checkout=args.source_checkout,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["render_lateral_athlete_video", "validate_lateral_athlete_video_manifest"]
