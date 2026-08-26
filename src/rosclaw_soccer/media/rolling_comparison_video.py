"""Side-by-side evidence video for legacy sliding and corrected rolling."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.media.trajectory_render import escape_filtergraph_option
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

_RESOLUTIONS = {"720p": (1280, 720), "1080p": (1920, 1080)}


@dataclass(frozen=True)
class RollingComparisonVideoResult:
    output_path: str
    manifest_path: str
    video_hash: str
    audit_hash: str
    request_hash: str
    trajectory_hash: str
    renderer_hash: str
    fps: int
    width: int
    height: int
    frame_count: int
    duration_sec: float
    legacy_median_slip_ratio: float
    corrected_median_slip_ratio: float
    source_audit_passed: bool
    visualization_only: bool = True
    pixels_used_for_scoring: bool = False
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.rolling_comparison_video.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Frame:
    simulation_time_sec: float
    view: str


def render_rolling_comparison_video(
    *,
    audit_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
    resolution: str = "1080p",
) -> RollingComparisonVideoResult:
    """Render an immutable rolling audit without using pixels for scoring."""

    checkout = source_checkout.expanduser().resolve()
    audit_file = audit_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    for path, label in ((audit_file, "audit"), (output, "video")):
        if path == checkout or checkout in path.parents:
            raise ValueError(f"rolling comparison {label} must remain outside source checkout")
    if output.suffix.lower() != ".mp4":
        raise ValueError("rolling comparison output must use .mp4")
    if not 10 <= fps <= 60:
        raise ValueError("rolling comparison fps must be in [10, 60]")
    try:
        width, height = _RESOLUTIONS[resolution]
    except KeyError as error:
        raise ValueError("rolling comparison resolution must be 720p or 1080p") from error
    if width % 2:
        raise ValueError("rolling comparison width must be even")
    manifest = output.with_suffix(".json")
    if output.exists() or manifest.exists():
        raise FileExistsError("rolling comparison output or manifest already exists")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for rolling comparison video")

    audit = _load_json(audit_file)
    request_path = audit_file.parent / "request.json"
    trajectory_path = audit_file.parent / "rolling-audit-trajectory.npz"
    if audit.get("passed") is not True or audit.get("strict_replay") is not True:
        raise ValueError("rolling comparison audit is not a passing strict replay")
    if (
        audit.get("activation_ceiling") != "SIM_ONLY"
        or audit.get("physics_authority") != "CPU_MUJOCO"
    ):
        raise ValueError("rolling comparison audit lacks SIM_ONLY CPU MuJoCo authority")
    if (
        audit.get("hardware_command_sent") is not False
        or audit.get("pixels_used_for_scoring") is not False
    ):
        raise ValueError("rolling comparison audit violates its authority boundary")
    if audit.get("request_hash") != _file_hash(request_path):
        raise ValueError("rolling comparison request hash mismatch")
    if audit.get("trajectory_hash") != _file_hash(trajectory_path):
        raise ValueError("rolling comparison trajectory hash mismatch")
    request = _load_json(request_path)
    trajectory = _load_trajectory(trajectory_path)
    legacy = _metrics(audit, "legacy")
    corrected = _metrics(audit, "corrected")
    if legacy.get("passed") is not False or corrected.get("passed") is not True:
        raise ValueError("rolling comparison audit does not isolate the expected regression")
    frames = _timeline(fps, float(request["duration_sec"]))

    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco

        goal = G1TrainingGoalSpec(
            plane_x_m=7.5,
            width_m=3.0,
            height_m=2.0,
            depth_m=1.2,
            target_y_m=0.89,
            target_z_m=0.115,
            precision_radius_m=0.10,
            ball_contact_sliding_friction=0.10,
            ball_sliding_friction=0.10,
        )
        models = (
            build_g1_stadium_model(asset_root, goal),
            build_g1_stadium_model(asset_root, goal),
        )
        half = width // 2
        for model in models:
            model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), half)
            model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = tuple(mujoco.MjData(model) for model in models)
        renderers = tuple(mujoco.Renderer(model, height=height, width=half) for model in models)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-soccer-roll-video-"):
                process = subprocess.Popen(
                    _ffmpeg_command(
                        ffmpeg=ffmpeg,
                        output=output,
                        fps=fps,
                        width=width,
                        height=height,
                        legacy=legacy,
                        corrected=corrected,
                    ),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                if process.stdin is None:
                    raise RuntimeError("rolling comparison ffmpeg pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        models=models,
                        data=data,
                        renderers=renderers,
                        trajectory=trajectory,
                        frames=frames,
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
                    raise RuntimeError(f"rolling comparison ffmpeg failed: {stderr[-2000:]}")
        finally:
            for renderer in renderers:
                renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    probe = _probe(ffprobe, output)
    expected_duration = len(frames) / fps
    if (
        (probe["width"], probe["height"]) != (width, height)
        or probe["fps"] != fps
        or probe["frame_count"] != len(frames)
        or abs(probe["duration_sec"] - expected_duration) > 1.0 / fps + 1e-6
    ):
        raise RuntimeError("rolling comparison encoded timeline does not match its contract")
    value = RollingComparisonVideoResult(
        output_path=str(output),
        manifest_path=str(manifest),
        video_hash=_file_hash(output),
        audit_hash=_file_hash(audit_file),
        request_hash=_file_hash(request_path),
        trajectory_hash=_file_hash(trajectory_path),
        renderer_hash=_renderer_hash(),
        fps=fps,
        width=width,
        height=height,
        frame_count=len(frames),
        duration_sec=probe["duration_sec"],
        legacy_median_slip_ratio=float(legacy["median_slip_ratio"]),
        corrected_median_slip_ratio=float(corrected["median_slip_ratio"]),
        source_audit_passed=True,
    )
    manifest.write_text(
        json.dumps(value.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def _timeline(fps: int, duration_sec: float) -> tuple[_Frame, ...]:
    intro = tuple(_Frame(0.0, "wide") for _ in range(round(1.20 * fps)))
    wide = tuple(
        _Frame(min(duration_sec, index / fps * 0.50), "wide")
        for index in range(math.ceil(duration_sec / 0.50 * fps))
    )
    close_duration = min(1.60, duration_sec)
    close = tuple(
        _Frame(min(close_duration, index / fps * 0.50), "close")
        for index in range(math.ceil(close_duration / 0.50 * fps))
    )
    finale = tuple(_Frame(duration_sec, "wide") for _ in range(round(2.0 * fps)))
    return intro + wide + close + finale


def _write_frames(
    *,
    mujoco: Any,
    models: tuple[Any, Any],
    data: tuple[Any, Any],
    renderers: tuple[Any, Any],
    trajectory: dict[str, NDArray[np.float64]],
    frames: tuple[_Frame, ...],
    stream: BinaryIO,
) -> None:
    cameras = (mujoco.MjvCamera(), mujoco.MjvCamera())
    for camera in cameras:
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    joints = tuple(
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")) for model in models
    )
    qpos = tuple(int(model.jnt_qposadr[joint]) for model, joint in zip(models, joints, strict=True))
    for frame in frames:
        images: list[NDArray[np.uint8]] = []
        for index, name in enumerate(("legacy", "corrected")):
            pose = _sample_pose(
                trajectory[f"{name}_time"],
                trajectory[f"{name}_ball_pose"],
                frame.simulation_time_sec,
            )
            data[index].qpos[:] = models[index].qpos0
            data[index].qpos[qpos[index] : qpos[index] + 7] = pose
            mujoco.mj_forward(models[index], data[index])
            camera = cameras[index]
            if frame.view == "close":
                camera.lookat[:] = pose[:3]
                camera.lookat[2] = 0.18
                camera.distance, camera.azimuth, camera.elevation = 1.25, 90.0, -10.0
            else:
                camera.lookat[:] = (2.35, 3.0, 0.18)
                camera.distance, camera.azimuth, camera.elevation = 5.1, 90.0, -10.0
            renderers[index].update_scene(data[index], camera=camera)
            images.append(np.asarray(renderers[index].render(), dtype=np.uint8).copy())
        stream.write(np.ascontiguousarray(np.concatenate(images, axis=1)).tobytes())


def _sample_pose(
    time: NDArray[np.float64],
    pose: NDArray[np.float64],
    simulation_time: float,
) -> NDArray[np.float64]:
    upper = int(np.searchsorted(time, simulation_time, side="right"))
    if upper <= 0:
        return np.asarray(pose[0], dtype=np.float64).copy()
    if upper >= len(time):
        return np.asarray(pose[-1], dtype=np.float64).copy()
    lower = upper - 1
    ratio = float((simulation_time - time[lower]) / (time[upper] - time[lower]))
    value: NDArray[np.float64] = np.empty(7, dtype=np.float64)
    value[:3] = pose[lower, :3] + ratio * (pose[upper, :3] - pose[lower, :3])
    value[3:] = _slerp(pose[lower, 3:], pose[upper, 3:], ratio)
    return value


def _slerp(
    left: NDArray[np.float64], right: NDArray[np.float64], ratio: float
) -> NDArray[np.float64]:
    start = left / np.linalg.norm(left)
    end = right / np.linalg.norm(right)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end, dot = -end, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = start + ratio * (end - start)
        return np.asarray(result / np.linalg.norm(result), dtype=np.float64)
    angle = math.acos(dot)
    scale = math.sin(angle)
    return np.asarray(
        math.sin((1.0 - ratio) * angle) / scale * start + math.sin(ratio * angle) / scale * end,
        dtype=np.float64,
    )


def _ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    fps: int,
    width: int,
    height: int,
    legacy: dict[str, Any],
    corrected: dict[str, Any],
) -> list[str]:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font_option = f"fontfile={escape_filtergraph_option(str(font))}:" if font.is_file() else ""
    scale = height / 720.0
    top = round(112 * scale)
    bottom = round(70 * scale)
    left_title = escape_filtergraph_option(
        f"LEGACY · SLIP {100 * float(legacy['median_slip_ratio']):.1f}% · FAIL"
    )
    right_title = escape_filtergraph_option(
        f"CORRECTED · SLIP {100 * float(corrected['median_slip_ratio']):.2f}% · PASS"
    )
    title = escape_filtergraph_option("SAME MEASURED PASS STATE · ONLY ANGULAR DAMPING CHANGED")
    footer = escape_filtergraph_option(
        "6D PHYSICS SCORING · STRICT CPU MUJOCO REPLAY · SIM ONLY · PIXELS DO NOT SCORE"
    )
    filters = (
        f"drawbox=x=0:y=0:w=iw:h={top}:color=0x030711@0.88:t=fill,"
        f"drawbox=x=0:y=h-{bottom}:w=iw:h={bottom}:color=0x030711@0.88:t=fill,"
        f"drawbox=x=w/2-2:y=0:w=4:h=ih:color=white@0.75:t=fill,"
        f"drawtext={font_option}text={title}:expansion=none:x=(w-text_w)/2:y={round(10 * scale)}:"
        f"fontsize={round(27 * scale)}:fontcolor=white,"
        f"drawtext={font_option}text={left_title}:expansion=none:"
        f"x={round(25 * scale)}:y={round(58 * scale)}:"
        f"fontsize={round(21 * scale)}:fontcolor=0xFF7D7D,"
        f"drawtext={font_option}text={right_title}:expansion=none:"
        f"x=w/2+{round(25 * scale)}:y={round(58 * scale)}:"
        f"fontsize={round(21 * scale)}:fontcolor=0x65F59A,"
        f"drawtext={font_option}text={footer}:expansion=none:"
        f"x=(w-text_w)/2:y=h-{round(45 * scale)}:"
        f"fontsize={round(18 * scale)}:fontcolor=0x8DD8FF"
    )
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
        filters,
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


def _load_trajectory(path: Path) -> dict[str, NDArray[np.float64]]:
    with np.load(path, allow_pickle=False) as archive:
        value = {name: np.asarray(archive[name], dtype=np.float64) for name in archive.files}
    for prefix in ("legacy", "corrected"):
        time = value.get(f"{prefix}_time")
        pose = value.get(f"{prefix}_ball_pose")
        velocity = value.get(f"{prefix}_ball_velocity")
        if time is None or pose is None or velocity is None:
            raise ValueError("rolling comparison trajectory is missing state")
        if time.ndim != 1 or pose.shape != (len(time), 7) or velocity.shape != (len(time), 6):
            raise ValueError("rolling comparison trajectory shapes are invalid")
        if not all(np.all(np.isfinite(array)) for array in (time, pose, velocity)):
            raise ValueError("rolling comparison trajectory contains non-finite state")
    return value


def _metrics(audit: dict[str, Any], name: str) -> dict[str, Any]:
    value = audit.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"rolling comparison {name} metrics are missing")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or not 1 <= path.stat().st_size <= 16 * 1024 * 1024:
        raise ValueError("rolling comparison JSON is missing or oversized")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("rolling comparison JSON must be an object")
    return value


def _probe(ffprobe: str, path: Path) -> dict[str, Any]:
    process = subprocess.run(
        (
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames:format=duration",
            "-of",
            "json",
            str(path),
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=120.0,
    )
    value = json.loads(process.stdout)
    stream = value["streams"][0]
    numerator, denominator = str(stream["avg_frame_rate"]).split("/", maxsplit=1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": int(round(int(numerator) / int(denominator))),
        "frame_count": int(stream["nb_read_frames"]),
        "duration_sec": float(value["format"]["duration"]),
    }


def _renderer_hash() -> str:
    return str(
        hash_json(
            {
                "renderer": _file_hash(Path(__file__)),
                "rolling": _file_hash(
                    Path(__file__).resolve().parents[1] / "physics/rolling_authenticity.py"
                ),
                "field": _file_hash(Path(__file__).resolve().parents[1] / "world/field.py"),
                "python": platform.python_version(),
                "numpy": np.__version__,
            }
        )
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


__all__ = ["RollingComparisonVideoResult", "render_rolling_comparison_video"]
