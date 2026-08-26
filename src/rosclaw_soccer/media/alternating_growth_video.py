"""Evidence-downstream showcase for alternating passer and shooter Growth rounds."""

from __future__ import annotations

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

from rosclaw_soccer.media.three_role_save_portfolio_video import (
    _add_ball_trail,
    _id,
    _joint_qpos,
    _lerp,
    _pose,
    _probe,
)
from rosclaw_soccer.media.trajectory_render import escape_filtergraph_option
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.dynamic_lead_pass_evidence import (
    validate_dynamic_lead_pass_evidence,
)
from rosclaw_soccer.training.upper_corner_strike_evidence import (
    validate_upper_corner_strike_evidence,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_coupled_stadium_model

_CLAIM = "ALTERNATING_TEAM_GROWTH_PASSER_THEN_SHOOTER"


@dataclass(frozen=True)
class _Frame:
    case_id: str
    simulation_time_sec: float
    view: str


@dataclass(frozen=True)
class _Clip:
    label: str
    frames: tuple[_Frame, ...]


def validate_alternating_growth_video_manifest(path: Path) -> dict[str, Any]:
    """Fail closed if a rendered video or any physics source changes."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("alternating Growth video manifest must be an object")
    manifest_hash = payload.pop("manifest_hash", None)
    try:
        source_files = payload.get("source_files")
        video_value = payload.get("video_path")
        if not isinstance(source_files, dict) or not isinstance(video_value, str):
            raise ValueError("alternating Growth video bindings are invalid")
        video = Path(video_value).expanduser().resolve()
        if not video.is_file() or hash_bytes(video.read_bytes()) != payload.get("video_hash"):
            raise ValueError("alternating Growth video hash changed")
        for source_value, expected in source_files.items():
            source = Path(source_value).expanduser().resolve()
            if not source.is_file() or hash_bytes(source.read_bytes()) != expected:
                raise ValueError("alternating Growth video source binding changed")
        if (
            payload.get("schema_version") != "rosclaw_soccer.alternating_growth_video.v1"
            or payload.get("claim") != _CLAIM
            or payload.get("case_count") != 4
            or payload.get("source_evidence_passed") is not True
            or payload.get("visualization_only") is not True
            or payload.get("pixels_used_for_scoring") is not False
            or payload.get("promotion_eligible") is not False
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("commercial_use_allowed") is not False
            or manifest_hash != hash_json(payload)
        ):
            raise ValueError("alternating Growth video authority contract is invalid")
    finally:
        if manifest_hash is not None:
            payload["manifest_hash"] = manifest_hash
    return payload


def render_alternating_growth_video(
    *,
    dynamic_pass_evidence_path: Path,
    upper_corner_evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render four passing/shooting cases without giving pixels metric authority."""

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
        raise ValueError("alternating Growth video output contract is invalid")
    dynamic_path = dynamic_pass_evidence_path.expanduser().resolve()
    upper_path = upper_corner_evidence_path.expanduser().resolve()
    dynamic = validate_dynamic_lead_pass_evidence(dynamic_path)
    upper = validate_upper_corner_strike_evidence(upper_path)
    source_files = {
        str(dynamic_path): hash_bytes(dynamic_path.read_bytes()),
        str(upper_path): hash_bytes(upper_path.read_bytes()),
    }
    trajectories: dict[str, dict[str, np.ndarray]] = {}
    results: dict[str, dict[str, Any]] = {}
    for case_id, case in cast(dict[str, dict[str, Any]], dynamic["holdouts"]).items():
        name = case.get("candidate_trajectory_file")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("dynamic-pass video trajectory name is invalid")
        trajectory_path = dynamic_path.parent / name
        source_files[str(trajectory_path)] = hash_bytes(trajectory_path.read_bytes())
        trajectories["pass-" + case_id] = _load_two_player_trajectory(trajectory_path)
        results["pass-" + case_id] = cast(dict[str, Any], case["candidate"])["result"]
    for lane_id, lane in cast(dict[str, dict[str, Any]], upper["lanes"]).items():
        name = lane.get("nominal_trajectory_file")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("upper-corner video trajectory name is invalid")
        trajectory_path = upper_path.parent / name
        source_files[str(trajectory_path)] = hash_bytes(trajectory_path.read_bytes())
        trajectories["strike-" + lane_id] = _load_two_player_trajectory(trajectory_path)
        results["strike-" + lane_id] = cast(dict[str, Any], lane["discovery"])["result"]
    if len(trajectories) != 4:
        raise ValueError("alternating Growth video requires exactly four cases")

    goal = G1TrainingGoalSpec(
        plane_x_m=5.0,
        width_m=7.32,
        height_m=2.44,
        depth_m=2.0,
        post_radius_m=0.06,
        target_y_m=3.429,
        target_z_m=1.75,
        precision_radius_m=0.12,
        regulation_field_enabled=True,
    )
    clips = _timeline(trajectories=trajectories, results=results, fps=fps)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for alternating Growth video")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    output.parent.mkdir(parents=True, exist_ok=True)
    first = trajectories[next(iter(trajectories))]
    passer_origin = tuple(float(value) for value in first["passer_pelvis_pose"][0, :3])
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco

        model = build_g1_coupled_stadium_model(
            asset_root,
            passer_origin_m=cast(tuple[float, float, float], passer_origin),
            spec=goal,
        )
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-alternating-growth-") as temp:
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
                    raise RuntimeError("alternating Growth raw-video pipe is unavailable")
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
                    raise RuntimeError(f"alternating Growth ffmpeg failed: {stderr[-3000:]}")
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl
    probe = _probe(ffprobe, output)
    frame_count = sum(len(clip.frames) for clip in clips)
    if (
        probe["width"] != width
        or probe["height"] != height
        or probe["fps"] != fps
        or abs(probe["frame_count"] - frame_count) > 1
    ):
        raise RuntimeError("alternating Growth encoded video contract changed")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.alternating_growth_video.v1",
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_files": source_files,
        "claim": _CLAIM,
        "case_count": len(trajectories),
        "source_evidence_passed": True,
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
    validate_alternating_growth_video_manifest(manifest_path)
    return manifest


def _load_two_player_trajectory(path: Path) -> dict[str, np.ndarray]:
    required = {
        "time",
        "ball_pose",
        "passer_pelvis_pose",
        "passer_joint_position",
        "shooter_pelvis_pose",
        "shooter_joint_position",
    }
    with np.load(path, allow_pickle=False) as archive:
        trajectory = {name: np.asarray(archive[name]) for name in archive.files}
    if not required <= set(trajectory):
        raise ValueError("alternating Growth trajectory is incomplete")
    length = len(trajectory["time"])
    if length < 2 or any(len(trajectory[name]) != length for name in required):
        raise ValueError("alternating Growth trajectory arrays are misaligned")
    if any(not np.all(np.isfinite(trajectory[name])) for name in required):
        raise ValueError("alternating Growth trajectory contains non-finite values")
    return trajectory


def _timeline(
    *,
    trajectories: dict[str, dict[str, np.ndarray]],
    results: dict[str, dict[str, Any]],
    fps: int,
) -> tuple[_Clip, ...]:
    first_id = next(iter(trajectories))
    first_time = float(trajectories[first_id]["time"][0])
    clips: list[_Clip] = [
        _Clip(
            "ONE TEAM · ONE PLASTIC ROLE PER ROUND · STRICT PHYSICS HOLDOUTS",
            tuple(_Frame(first_id, first_time, "wide") for _ in range(round(1.5 * fps))),
        )
    ]
    for case_id in (key for key in trajectories if key.startswith("pass-")):
        result = results[case_id]
        pass_time = float(result["pass_contact_time_sec"])
        shot_time = float(result["shot_contact_time_sec"])
        error_cm = 100.0 * float(result["pass_delivery_error_m"])
        clips.append(
            _Clip(
                f"MOVING RECEIVER · LEARNED LEAD PASS · DELIVERY ERROR {error_cm:.1f} cm",
                _segment(case_id, pass_time - 0.65, shot_time + 0.50, 0.82, "chain", fps),
            )
        )
    for case_id in (key for key in trajectories if key.startswith("strike-")):
        result = results[case_id]
        shot_time = float(result["shot_contact_time_sec"])
        crossing_y = float(result["goal_crossing_y_m"])
        crossing_z = float(result["goal_crossing_z_m"])
        clips.append(
            _Clip(
                f"REGULATION UPPER CORNER · CROSSING y={crossing_y:+.2f} z={crossing_z:.2f} m",
                _segment(case_id, shot_time - 0.85, shot_time + 1.10, 0.70, "goal", fps),
            )
        )
        clips.append(
            _Clip(
                "BOUNDED CONTACT-TORQUE MUSCLE MEMORY · SLOW-MOTION REPLAY",
                _segment(case_id, shot_time - 0.22, shot_time + 0.72, 0.30, "hero", fps),
            )
        )
    last_id = next(reversed(trajectories))
    last_time = float(trajectories[last_id]["time"][-1])
    clips.append(
        _Clip(
            "PASSER GREW · SHOOTER GREW · UNSAFE GOALKEEPER CANDIDATE STAYED FROZEN",
            tuple(_Frame(last_id, last_time, "wide") for _ in range(round(1.8 * fps))),
        )
    )
    return tuple(clips)


def _segment(
    case_id: str,
    start: float,
    end: float,
    speed: float,
    view: str,
    fps: int,
) -> tuple[_Frame, ...]:
    if end <= start or speed <= 0.0:
        raise ValueError("alternating Growth video segment is invalid")
    count = max(1, int(math.ceil((end - start) / speed * fps)))
    return tuple(
        _Frame(case_id, min(end, start + index / fps * speed), view) for index in range(count)
    )


def _sample(trajectory: dict[str, np.ndarray], time_sec: float) -> dict[str, Any]:
    time = trajectory["time"]
    upper = int(np.searchsorted(time, time_sec, side="right"))
    if upper <= 0:
        lower = upper = 0
        ratio = 0.0
    elif upper >= len(time):
        lower = upper = len(time) - 1
        ratio = 0.0
    else:
        lower = upper - 1
        ratio = float((time_sec - time[lower]) / (time[upper] - time[lower]))
    sampled: dict[str, Any] = {"index": upper if ratio >= 0.5 else lower}
    for role in ("passer", "shooter"):
        sampled[f"{role}_pelvis_pose"] = _pose(
            trajectory[f"{role}_pelvis_pose"][lower],
            trajectory[f"{role}_pelvis_pose"][upper],
            ratio,
        )
        sampled[f"{role}_joint_position"] = _lerp(
            trajectory[f"{role}_joint_position"][lower],
            trajectory[f"{role}_joint_position"][upper],
            ratio,
        )
    sampled["ball_pose"] = _pose(
        trajectory["ball_pose"][lower], trajectory["ball_pose"][upper], ratio
    )
    return sampled


def _write_frames(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    trajectories: dict[str, dict[str, np.ndarray]],
    clips: tuple[_Clip, ...],
    goal: G1TrainingGoalSpec,
    stream: BinaryIO,
) -> None:
    ball_body = _id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_joint = int(model.body_jntadr[ball_body])
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    passer_joint = _id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, "passer_floating_base_joint")
    base_qpos = {"shooter": 0, "passer": int(model.jnt_qposadr[passer_joint])}
    joints = {
        "shooter": _joint_qpos(mujoco, model, ""),
        "passer": _joint_qpos(mujoco, model, "passer_"),
    }
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    for clip in clips:
        for frame in clip.frames:
            trajectory = trajectories[frame.case_id]
            sampled = _sample(trajectory, frame.simulation_time_sec)
            data.qpos[:] = model.qpos0
            for role in ("shooter", "passer"):
                data.qpos[base_qpos[role] : base_qpos[role] + 7] = sampled[f"{role}_pelvis_pose"]
                data.qpos[joints[role]] = sampled[f"{role}_joint_position"]
            data.qpos[ball_qpos : ball_qpos + 7] = sampled["ball_pose"]
            mujoco.mj_forward(model, data)
            _set_camera(camera, frame.view, sampled, goal)
            renderer.update_scene(data, camera=camera)
            _add_ball_trail(mujoco, renderer.scene, trajectory, int(sampled["index"]))
            stream.write(np.ascontiguousarray(renderer.render().copy()).tobytes())


def _set_camera(camera: Any, view: str, sampled: dict[str, Any], goal: G1TrainingGoalSpec) -> None:
    ball = np.asarray(sampled["ball_pose"], dtype=np.float64)
    passer = np.asarray(sampled["passer_pelvis_pose"], dtype=np.float64)
    shooter = np.asarray(sampled["shooter_pelvis_pose"], dtype=np.float64)
    if view == "chain":
        camera.lookat[:] = 0.35 * passer[:3] + 0.35 * shooter[:3] + 0.30 * ball[:3]
        camera.lookat[2] = 0.72
        camera.distance, camera.azimuth, camera.elevation = 7.4, 111.0, -8.0
    elif view == "goal":
        camera.lookat[:] = (goal.plane_x_m - 0.55, float(ball[1]), 1.25)
        camera.distance, camera.azimuth, camera.elevation = 7.1, 174.0, -5.0
    elif view == "hero":
        camera.lookat[:] = 0.55 * shooter[:3] + 0.45 * ball[:3]
        camera.lookat[2] = 0.90
        camera.distance, camera.azimuth, camera.elevation = 5.4, 135.0, -7.0
    else:
        camera.lookat[:] = (3.35, 0.0, 0.80)
        camera.distance, camera.azimuth, camera.elevation = 11.5, 94.0, -11.0


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
        f"drawbox=x=0:y=0:w=iw:h={round(116 * scale)}:color=0x030711@0.82:t=fill",
        f"drawbox=x=0:y=h-{round(62 * scale)}:w=iw:h={round(62 * scale)}:"
        "color=0x030711@0.82:t=fill",
        f"drawtext={font_option}text='ROSClaw Soccer · ALTERNATING TEAM GROWTH':"
        f"expansion=none:x={left}:y={round(13 * scale)}:fontsize={round(32 * scale)}:"
        "fontcolor=white",
        f"drawtext={font_option}text='CPU MUJOCO TRUTH · REGULATION GOAL · "
        "SIM ONLY · PIXELS NEVER SCORE':"
        f"expansion=none:x={left}:y=h-{round(40 * scale)}:fontsize={round(18 * scale)}:"
        "fontcolor=0x8DD8FF",
    ]
    offset = 0.0
    for label, clip in zip(labels, clips, strict=True):
        end = offset + len(clip.frames) / fps
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


__all__ = [
    "render_alternating_growth_video",
    "validate_alternating_growth_video_manifest",
]
