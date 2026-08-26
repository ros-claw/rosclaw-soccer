"""Render the S102 neural dive-command CPU MuJoCo evidence reel."""

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
from numpy.typing import NDArray

from rosclaw_soccer.media.three_role_save_portfolio_video import _pose, _probe
from rosclaw_soccer.media.trajectory_render import escape_filtergraph_option
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.dive_athlete_cpu_exam import (
    validate_dive_athlete_cpu_exam_report,
)
from rosclaw_soccer.world.field import build_g1_stadium_model

_CLAIM = "S102_NEURAL_DIVE_COMMAND_CPU_MUJOCO_REPLAY_NO_BALL_CONTACT_CLAIM"


@dataclass(frozen=True)
class _Clip:
    case_id: str
    direction_index: int
    label: str
    frame_source_time_sec: tuple[float, ...]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_dive_athlete_video_manifest(path: Path) -> dict[str, Any]:
    """Validate the encoded video and every scored-state source binding."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dive athlete video manifest must be an object")
    expected = payload.pop("manifest_hash", None)
    if expected != hash_json(payload):
        raise ValueError("dive athlete video manifest integrity mismatch")
    video_value = payload.get("video_path")
    source_files = payload.get("source_files")
    if not isinstance(video_value, str) or not isinstance(source_files, dict):
        raise ValueError("dive athlete video source bindings are invalid")
    video = Path(video_value).expanduser().resolve()
    if not video.is_file() or hash_bytes(video.read_bytes()) != payload.get("video_hash"):
        raise ValueError("dive athlete video hash changed")
    for source_value, source_hash in source_files.items():
        source = Path(source_value).expanduser().resolve()
        if not source.is_file() or hash_bytes(source.read_bytes()) != source_hash:
            raise ValueError("dive athlete video source hash changed")
    numbers = tuple(
        payload.get(name) for name in ("fps", "width", "height", "frame_count", "duration_sec")
    )
    if (
        payload.get("schema_version") != "rosclaw_soccer.dive_athlete_video.v1"
        or payload.get("claim") != _CLAIM
        or payload.get("case_count") != 4
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
        raise ValueError("dive athlete video authority contract is invalid")
    payload["manifest_hash"] = expected
    return cast(dict[str, Any], payload)


def _load_state(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "time",
            "direction",
            "commanded_joint_position",
            "achieved_joint_position",
            "pelvis_pose",
            "root_velocity",
        }
        if set(archive.files) != required:
            raise ValueError("dive athlete state trajectory arrays changed")
        values = {name: np.asarray(archive[name]) for name in archive.files}
    time = np.asarray(values["time"], dtype=np.float64)
    achieved = np.asarray(values["achieved_joint_position"], dtype=np.float64)
    pelvis = np.asarray(values["pelvis_pose"], dtype=np.float64)
    if (
        time.ndim != 1
        or time.size < 80
        or not np.all(np.diff(time) > 0.0)
        or achieved.shape != (2, time.size, 29)
        or pelvis.shape != (2, time.size, 7)
        or not np.all(np.isfinite(achieved))
        or not np.all(np.isfinite(pelvis))
        or not np.array_equal(values["direction"], np.asarray((-1, 1), dtype=np.int64))
    ):
        raise ValueError("dive athlete state trajectory is invalid")
    return values


def _timeline(
    *, states: dict[str, dict[str, NDArray[Any]]], evidence: dict[str, Any], fps: int
) -> tuple[_Clip, ...]:
    clips: list[_Clip] = []
    order = (
        ("s92-left-outer", 0, "LEFT · OUTER TARGET"),
        ("s92-left-outer", 1, "RIGHT · OUTER TARGET"),
        ("s92-left-inner", 0, "LEFT · INNER TARGET"),
        ("s92-left-inner", 1, "RIGHT · INNER TARGET"),
    )
    display_duration_sec = 3.2
    lead_hold_sec = 0.34
    playback_rate = 0.74
    for index, (case_id, direction_index, title) in enumerate(order, start=1):
        time = np.asarray(states[case_id]["time"], dtype=np.float64)
        source_times = tuple(
            float(
                np.clip(
                    (frame / fps - lead_hold_sec) * playback_rate,
                    0.0,
                    float(time[-1]),
                )
            )
            for frame in range(round(display_duration_sec * fps))
        )
        outcome = evidence["case_reports"][case_id]["outcomes"][direction_index]
        clips.append(
            _Clip(
                case_id=case_id,
                direction_index=direction_index,
                label=(
                    f"RUN {index}/4 · {title} · "
                    f"SHIFT {100.0 * abs(float(outcome['final_lateral_displacement_m'])):.1f} CM · "
                    "0 JOINT-LIMIT ERROR"
                ),
                frame_source_time_sec=source_times,
            )
        )
    return tuple(clips)


def _interpolate_state(
    state: dict[str, NDArray[Any]], *, direction_index: int, time_sec: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    time = np.asarray(state["time"], dtype=np.float64)
    upper = min(int(np.searchsorted(time, time_sec, side="right")), time.size - 1)
    lower = max(0, upper - 1)
    interval = float(time[upper] - time[lower])
    ratio = 0.0 if interval <= 0.0 else (time_sec - float(time[lower])) / interval
    pelvis_values = np.asarray(state["pelvis_pose"], dtype=np.float64)
    joint_values = np.asarray(state["achieved_joint_position"], dtype=np.float64)
    pelvis = _pose(
        pelvis_values[direction_index, lower],
        pelvis_values[direction_index, upper],
        ratio,
    )
    joint = joint_values[direction_index, lower] + ratio * (
        joint_values[direction_index, upper] - joint_values[direction_index, lower]
    )
    return np.asarray(pelvis, dtype=np.float64), np.asarray(joint, dtype=np.float64)


def _write_frames(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    states: dict[str, dict[str, NDArray[Any]]],
    clips: tuple[_Clip, ...],
    stream: BinaryIO,
) -> None:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.lookat[:] = (4.52, 0.0, 0.76)
    camera.distance = 5.8
    camera.azimuth = 178.0
    camera.elevation = -7.0
    for clip in clips:
        state = states[clip.case_id]
        for time_sec in clip.frame_source_time_sec:
            pelvis, joint = _interpolate_state(
                state,
                direction_index=clip.direction_index,
                time_sec=time_sec,
            )
            data.qpos[:7] = pelvis
            data.qpos[7:36] = joint
            data.qpos[36:43] = (-20.0, 0.0, 0.115, 1.0, 0.0, 0.0, 0.0)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            camera.lookat[1] = 0.22 * float(pelvis[1])
            renderer.update_scene(data, camera=camera)
            stream.write(np.asarray(renderer.render(), dtype=np.uint8).tobytes())


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
        f"drawtext={font_option}text='ROSClaw Soccer · S102 NEURAL DIVE ATHLETE':"
        f"expansion=none:x={left}:y={round(13 * scale)}:fontsize={round(32 * scale)}:"
        "fontcolor=white",
        f"drawtext={font_option}text='4xA6000 LEARNING · CPU MUJOCO STATE REPLAY · "
        "COMMAND EXAM · NO BALL-CONTACT CLAIM':"
        f"expansion=none:x={left}:y=h-{round(40 * scale)}:fontsize={round(18 * scale)}:"
        "fontcolor=0x8DD8FF",
    ]
    offset = 0.0
    for label, clip in zip(labels, clips, strict=True):
        end = offset + len(clip.frame_source_time_sec) / fps
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


def render_dive_athlete_video(
    *,
    exam_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 60,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render scored states after validating all non-pixel evidence bindings."""

    exam_file = exam_path.expanduser().resolve()
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
        raise ValueError("dive athlete video output contract is invalid")
    evidence = validate_dive_athlete_cpu_exam_report(exam_file)
    cases = cast(dict[str, dict[str, Any]], evidence["case_reports"])
    source_files = {str(exam_file): hash_bytes(exam_file.read_bytes())}
    states: dict[str, dict[str, NDArray[Any]]] = {}
    for case_id, case in cases.items():
        state_path = exam_file.parent / f"{case_id}-state.npz"
        if hash_bytes(state_path.read_bytes()) != case.get("state_trajectory_hash"):
            raise ValueError("dive athlete video state binding changed")
        states[case_id] = _load_state(state_path)
        source_files[str(state_path)] = hash_bytes(state_path.read_bytes())
    clips = _timeline(states=states, evidence=evidence, fps=fps)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for dive athlete video")
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        import mujoco

        model = build_g1_stadium_model(asset_root.expanduser().resolve())
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-dive-athlete-") as temp:
                label_paths: list[Path] = []
                for index, clip in enumerate(clips):
                    label_path = Path(temp) / f"label-{index}.txt"
                    label_path.write_text(clip.label, encoding="utf-8")
                    label_paths.append(label_path)
                process = subprocess.Popen(
                    _ffmpeg_command(
                        ffmpeg=ffmpeg,
                        output=output,
                        width=width,
                        height=height,
                        fps=fps,
                        clips=clips,
                        labels=tuple(label_paths),
                    ),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                if process.stdin is None:
                    raise RuntimeError("dive athlete ffmpeg pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        renderer=renderer,
                        states=states,
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
                    raise RuntimeError(f"dive athlete ffmpeg failed: {stderr[-3000:]}")
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl
    frame_count = sum(len(clip.frame_source_time_sec) for clip in clips)
    probe = _probe(ffprobe, output)
    if (
        probe["width"] != width
        or probe["height"] != height
        or probe["fps"] != fps
        or abs(probe["frame_count"] - frame_count) > 1
    ):
        raise RuntimeError("dive athlete encoded video contract changed")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.dive_athlete_video.v1",
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_files": source_files,
        "source_exam_report_hash": evidence["report_hash"],
        "claim": _CLAIM,
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
    _atomic_json(manifest_path, manifest)
    return validate_dive_athlete_video_manifest(manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exam", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    manifest = render_dive_athlete_video(
        exam_path=args.exam,
        asset_root=args.asset_root,
        output_path=args.output,
        source_checkout=args.source_checkout,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["render_dive_athlete_video", "validate_dive_athlete_video_manifest"]
