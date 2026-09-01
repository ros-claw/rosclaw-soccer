"""Render an evidence-downstream before/after First Touch growth video."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
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
from rosclaw_soccer.media.trajectory_render import append_sphere, escape_filtergraph_option
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_coupled_stadium_model

_CLAIM = "MATCHED_FIRST_TOUCH_LOCAL_ACQUISITION_BEFORE_AFTER"


@dataclass(frozen=True)
class _Frame:
    case_id: str
    simulation_time_sec: float
    view: str


@dataclass(frozen=True)
class _Clip:
    label: str
    frames: tuple[_Frame, ...]


def _load_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("First Touch video source report must be an object")
    declared = value.pop("report_hash", None)
    try:
        if (
            declared != hash_json(value)
            or value.get("schema_version") != "rosclaw_soccer.first_touch_physics_evidence.v1"
        ):
            raise ValueError("First Touch video source report is not content bound")
    finally:
        value["report_hash"] = declared
    return cast(dict[str, Any], value)


def _load_exam(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("First Touch paired exam must be an object")
    declared = value.pop("exam_hash", None)
    try:
        if (
            declared != hash_json(value)
            or value.get("schema_version") != "rosclaw_soccer.first_touch_paired_growth_exam.v1"
            or value.get("status") != "PASS_PAIRED_ACQUISITION"
            or value.get("deterministic_candidate_replay") is not True
        ):
            raise ValueError("First Touch paired exam has not passed")
    finally:
        value["exam_hash"] = declared
    return cast(dict[str, Any], value)


def _load_trajectory(path: Path) -> dict[str, np.ndarray]:
    required = {
        "time",
        "ball_pose",
        "ball_contact_role",
        "passer_pelvis_pose",
        "passer_joint_position",
        "shooter_pelvis_pose",
        "shooter_joint_position",
    }
    with np.load(path, allow_pickle=False) as archive:
        value = {name: np.asarray(archive[name]) for name in archive.files}
    if not required <= value.keys():
        raise ValueError("First Touch render trajectory is incomplete")
    length = len(value["time"])
    if length < 2 or any(len(value[name]) != length for name in required):
        raise ValueError("First Touch render trajectory arrays are misaligned")
    if any(not np.all(np.isfinite(value[name])) for name in required):
        raise ValueError("First Touch render trajectory is non-finite")
    return value


def validate_first_touch_growth_video_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("First Touch video manifest must be an object")
    declared = payload.pop("manifest_hash", None)
    try:
        sources = payload.get("source_files")
        video_value = payload.get("video_path")
        if not isinstance(sources, dict) or not isinstance(video_value, str):
            raise ValueError("First Touch video bindings are invalid")
        video = Path(video_value).expanduser().resolve()
        if not video.is_file() or hash_bytes(video.read_bytes()) != payload.get("video_hash"):
            raise ValueError("First Touch video changed")
        for source_value, expected_hash in sources.items():
            source = Path(source_value).expanduser().resolve()
            if not source.is_file() or hash_bytes(source.read_bytes()) != expected_hash:
                raise ValueError("First Touch video source changed")
        if (
            payload.get("schema_version") != "rosclaw_soccer.first_touch_growth_video.v1"
            or payload.get("claim") != _CLAIM
            or payload.get("source_exam_passed") is not True
            or payload.get("visualization_only") is not True
            or payload.get("pixels_used_for_scoring") is not False
            or payload.get("promotion_eligible") is not False
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or declared != hash_json(payload)
        ):
            raise ValueError("First Touch video authority contract is invalid")
    finally:
        payload["manifest_hash"] = declared
    return cast(dict[str, Any], payload)


def render_first_touch_growth_video(
    *,
    paired_exam_path: Path,
    baseline_report_path: Path,
    candidate_report_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
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
        raise ValueError("First Touch video output contract is invalid")
    exam_path = paired_exam_path.expanduser().resolve()
    baseline_path = baseline_report_path.expanduser().resolve()
    candidate_path = candidate_report_path.expanduser().resolve()
    exam = _load_exam(exam_path)
    reports = {
        "baseline": _load_report(baseline_path),
        "candidate": _load_report(candidate_path),
    }
    if (
        exam["baseline"]["report_hash"] != reports["baseline"]["report_hash"]
        or exam["candidate"]["report_hash"] != reports["candidate"]["report_hash"]
    ):
        raise ValueError("First Touch video reports do not belong to the paired exam")
    source_files = {
        str(exam_path): hash_bytes(exam_path.read_bytes()),
        str(baseline_path): hash_bytes(baseline_path.read_bytes()),
        str(candidate_path): hash_bytes(candidate_path.read_bytes()),
    }
    trajectories: dict[str, dict[str, np.ndarray]] = {}
    for case_id, report_path in (("baseline", baseline_path), ("candidate", candidate_path)):
        name = reports[case_id]["physics"]["trajectory_artifact"]
        trajectory_path = report_path.parent / name
        if (
            hash_bytes(trajectory_path.read_bytes())
            != reports[case_id]["physics"]["trajectory_artifact_hash"]
        ):
            raise ValueError("First Touch video trajectory binding changed")
        source_files[str(trajectory_path)] = hash_bytes(trajectory_path.read_bytes())
        trajectories[case_id] = _load_trajectory(trajectory_path)

    clips = _timeline(reports=reports, trajectories=trajectories, fps=fps)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for First Touch video")
    output.parent.mkdir(parents=True, exist_ok=True)
    goal = G1TrainingGoalSpec(plane_x_m=12.0, target_y_m=0.8, target_z_m=1.0)
    passer_origin = tuple(
        float(value) for value in trajectories["candidate"]["passer_pelvis_pose"][0, :3]
    )
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
            with tempfile.TemporaryDirectory(prefix="rosclaw-first-touch-growth-") as temp:
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
                    raise RuntimeError("First Touch video raw-video pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        renderer=renderer,
                        trajectories=trajectories,
                        reports=reports,
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
                    raise RuntimeError(f"First Touch video ffmpeg failed: {stderr[-3000:]}")
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
        raise RuntimeError("First Touch encoded video contract changed")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.first_touch_growth_video.v1",
        "claim": _CLAIM,
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_files": source_files,
        "source_exam_hash": exam["exam_hash"],
        "source_exam_passed": True,
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
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    validate_first_touch_growth_video_manifest(manifest_path)
    return manifest


def _contact_time(trajectory: dict[str, np.ndarray]) -> float:
    contacts = np.flatnonzero(trajectory["ball_contact_role"] == 2)
    if not contacts.size:
        raise ValueError("First Touch video source lacks a shooter contact")
    return float(trajectory["time"][int(contacts[0])])


def _segment(
    case_id: str, start: float, end: float, speed: float, view: str, fps: int
) -> tuple[_Frame, ...]:
    count = max(1, int(math.ceil((end - start) / speed * fps)))
    return tuple(
        _Frame(case_id, min(end, start + index / fps * speed), view) for index in range(count)
    )


def _timeline(
    *,
    reports: dict[str, dict[str, Any]],
    trajectories: dict[str, dict[str, np.ndarray]],
    fps: int,
) -> tuple[_Clip, ...]:
    baseline_contact = _contact_time(trajectories["baseline"])
    candidate_contact = _contact_time(trajectories["candidate"])
    before = reports["baseline"]["measurement"]
    after = reports["candidate"]["measurement"]
    return (
        _Clip(
            "FIRST TOUCH GROWTH · SAME BODY · SAME BALL · SAME PHYSICS",
            tuple(_Frame("baseline", baseline_contact - 1.0, "wide") for _ in range(fps)),
        ),
        _Clip(
            f"BEFORE · TOO HARD · {before['outgoing_speed_mps']:.2f} m/s · "
            f"ERROR {100.0 * before['target_error_m']:.1f} cm",
            _segment(
                "baseline", baseline_contact - 1.0, baseline_contact + 1.25, 0.65, "wide", fps
            ),
        ),
        _Clip(
            f"AFTER · CONTROLLED · {after['outgoing_speed_mps']:.2f} m/s · "
            f"ERROR {100.0 * after['target_error_m']:.1f} cm",
            _segment(
                "candidate", candidate_contact - 1.0, candidate_contact + 1.25, 0.65, "wide", fps
            ),
        ),
        _Clip(
            f"RIGHT-FOOT CONTROL · DIRECTION ERROR {after['direction_error_deg']:.2f}° · "
            f"NEXT ACTION {1000.0 * after['next_action_latency_sec']:.0f} ms",
            _segment(
                "candidate", candidate_contact - 0.45, candidate_contact + 0.85, 0.32, "hero", fps
            ),
        ),
        _Clip(
            "LOCAL ACQUISITION PASSED · DETERMINISTIC REPLAY · ZERO SAFETY VIOLATIONS",
            tuple(
                _Frame("candidate", candidate_contact + 1.25, "wide")
                for _ in range(round(1.4 * fps))
            ),
        ),
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
    result: dict[str, Any] = {"index": upper if ratio >= 0.5 else lower}
    for role in ("passer", "shooter"):
        result[f"{role}_pelvis_pose"] = _pose(
            trajectory[f"{role}_pelvis_pose"][lower],
            trajectory[f"{role}_pelvis_pose"][upper],
            ratio,
        )
        result[f"{role}_joint_position"] = _lerp(
            trajectory[f"{role}_joint_position"][lower],
            trajectory[f"{role}_joint_position"][upper],
            ratio,
        )
    result["ball_pose"] = _pose(
        trajectory["ball_pose"][lower], trajectory["ball_pose"][upper], ratio
    )
    return result


def _target_marker(trajectory: dict[str, np.ndarray], report: dict[str, Any]) -> np.ndarray:
    contacts = np.flatnonzero(trajectory["ball_contact_role"] == 2)
    index = int(contacts[0])
    angle = math.radians(float(report["scenario"]["target_direction_deg"]))
    distance = float(report["scenario"]["target_outgoing_speed_mps"]) * float(
        report["scenario"]["measurement_horizon_sec"]
    )
    result = np.asarray(trajectory["ball_pose"][index, :3], dtype=np.float64).copy()
    result[:2] += distance * np.asarray((math.cos(angle), math.sin(angle)))
    return result


def _write_frames(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    trajectories: dict[str, dict[str, np.ndarray]],
    reports: dict[str, dict[str, Any]],
    clips: tuple[_Clip, ...],
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
    targets = {
        case_id: _target_marker(trajectory, reports[case_id])
        for case_id, trajectory in trajectories.items()
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
            _set_camera(camera, frame.view, sampled)
            renderer.update_scene(data, camera=camera)
            _add_ball_trail(mujoco, renderer.scene, trajectory, int(sampled["index"]))
            append_sphere(
                mujoco,
                renderer.scene,
                targets[frame.case_id],
                0.07,
                (0.10, 0.95, 0.95, 0.50),
            )
            stream.write(np.ascontiguousarray(renderer.render().copy()).tobytes())


def _set_camera(camera: Any, view: str, sampled: dict[str, Any]) -> None:
    ball = np.asarray(sampled["ball_pose"])
    shooter = np.asarray(sampled["shooter_pelvis_pose"])
    camera.lookat[:] = 0.58 * shooter[:3] + 0.42 * ball[:3]
    camera.lookat[2] = 0.72 if view == "wide" else 0.64
    if view == "hero":
        camera.distance, camera.azimuth, camera.elevation = 3.7, 132.0, -7.0
    else:
        camera.distance, camera.azimuth, camera.elevation = 5.5, 116.0, -9.0


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
        f"drawtext={font_option}text='ROSClaw Soccer · FIRST TOUCH GROWTH':expansion=none:"
        f"x={left}:y={round(13 * scale)}:fontsize={round(32 * scale)}:fontcolor=white",
        f"drawtext={font_option}text='CPU MUJOCO TRUTH · CYAN = 0.2 s TARGET · "
        "SIM ONLY · PIXELS NEVER SCORE':expansion=none:"
        f"x={left}:y=h-{round(40 * scale)}:fontsize={round(18 * scale)}:fontcolor=0x8DD8FF",
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


def _main() -> None:
    # ``python -m`` imports the package before this module, and some optional
    # media dependencies may import MuJoCo during that package initialization.
    # Re-exec once with the backend already fixed instead of silently falling
    # back to GLFW in a headless evidence job.
    if "MUJOCO_GL" not in os.environ:
        environment = os.environ.copy()
        environment["MUJOCO_GL"] = "egl"
        environment["PYOPENGL_PLATFORM"] = "egl"
        module = (
            __spec__.name
            if __spec__ is not None
            else "rosclaw_soccer.media.first_touch_growth_video"
        )
        os.execvpe(
            sys.executable,
            (sys.executable, "-m", module, *sys.argv[1:]),
            environment,
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-exam", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    report = render_first_touch_growth_video(
        paired_exam_path=args.paired_exam,
        baseline_report_path=args.baseline_report,
        candidate_report_path=args.candidate_report,
        asset_root=args.asset_root,
        output_path=args.output,
        source_checkout=args.source_checkout,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    _main()


__all__ = [
    "render_first_touch_growth_video",
    "validate_first_touch_growth_video_manifest",
]
