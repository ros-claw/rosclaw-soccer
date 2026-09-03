"""Evidence-downstream video for the sealed prepared-finisher exam."""

from __future__ import annotations

import hashlib
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

from rosclaw_soccer.evidence.three_player import load_three_player_trajectory
from rosclaw_soccer.media.three_role_save_portfolio_video import (
    _add_ball_trail,
    _id,
    _joint_qpos,
    _lerp,
    _pose,
    _probe,
)
from rosclaw_soccer.media.trajectory_render import append_sphere, escape_filtergraph_option
from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.development_evidence import (
    three_role_development_kwargs,
    three_role_goal_spec,
)
from rosclaw_soccer.training.runtime_finish_plan_exam import (
    validate_runtime_finish_plan_exam,
)
from rosclaw_soccer.world.field import build_g1_three_player_stadium_model

_CLAIM = "SEALED_FRESH_PREPARED_FINISH_PLAN_GROWTH"


@dataclass(frozen=True)
class _Case:
    row_index: int
    case_id: str
    result: dict[str, Any]
    trajectory: dict[str, np.ndarray]
    trajectory_path: Path
    trajectory_hash: str
    trajectory_digest: str


@dataclass(frozen=True)
class _Frame:
    case_index: int
    simulation_time_sec: float
    view: str


@dataclass(frozen=True)
class _Clip:
    label: str
    frames: tuple[_Frame, ...]


def render_runtime_finish_plan_video(
    *,
    exam_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render only strict S167 candidates after validating all source bindings."""

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
        raise ValueError("runtime finish video output contract is invalid")
    report_path = exam_path.expanduser().resolve()
    report, request, cases, source_files = _load_sources(report_path)
    clips = _timeline(cases, fps)
    goal = three_role_goal_spec()
    kwargs = three_role_development_kwargs()
    keeper = kwargs["goalkeeper_config"]
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for runtime finish video")
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ["MUJOCO_GL"] = "egl"
    try:
        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        if request.get("body_hash") != qualification.body_hash:
            raise ValueError("runtime finish video body identity changed")
        import mujoco

        passer_origin = tuple(
            float(value) for value in cases[0].trajectory["passer_pelvis_pose"][0, :3]
        )
        model = build_g1_three_player_stadium_model(
            asset_root,
            passer_origin_m=cast(tuple[float, float, float], passer_origin),
            goalkeeper_origin_m=(goal.plane_x_m - keeper.depth_from_goal_line_m, 0.0, 0.0),
            spec=goal,
        )
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-runtime-finish-video-") as temp:
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
                    raise RuntimeError("runtime finish video raw pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        renderer=renderer,
                        cases=cases,
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
                    raise RuntimeError(f"runtime finish video ffmpeg failed: {stderr[-3000:]}")
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
        raise RuntimeError("runtime finish encoded video contract changed")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_finish_plan_video.v1",
        "claim": _CLAIM,
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_files": source_files,
        "source_exam_path": str(report_path),
        "source_exam_hash": report["report_hash"],
        "source_exam_status": report["status"],
        "source_candidate_strict_success_count": report["metrics"][
            "candidate_strict_success_count"
        ],
        "rendered_case_ids": [case.case_id for case in cases],
        "rendered_trajectory_digests": [case.trajectory_digest for case in cases],
        "renderer_hash": _renderer_hash(),
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
    validate_runtime_finish_plan_video_manifest(manifest_path)
    return manifest


def validate_runtime_finish_plan_video_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime finish video manifest must be an object")
    declared = payload.pop("manifest_hash", None)
    try:
        sources = payload.get("source_files")
        video_value = payload.get("video_path")
        exam_value = payload.get("source_exam_path")
        if (
            not isinstance(sources, dict)
            or not isinstance(video_value, str)
            or not isinstance(exam_value, str)
        ):
            raise ValueError("runtime finish video bindings are invalid")
        video = Path(video_value).expanduser().resolve()
        if not video.is_file() or hash_bytes(video.read_bytes()) != payload.get("video_hash"):
            raise ValueError("runtime finish video changed")
        for source_path, expected_hash in sources.items():
            source = Path(source_path).expanduser().resolve()
            if not source.is_file() or hash_bytes(source.read_bytes()) != expected_hash:
                raise ValueError("runtime finish video source changed")
        exam = validate_runtime_finish_plan_exam(Path(exam_value))
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            raise RuntimeError("ffprobe is required to validate runtime finish video")
        probe = _probe(ffprobe, video)
        if (
            payload.get("schema_version") != "rosclaw_soccer.runtime_finish_plan_video.v1"
            or payload.get("claim") != _CLAIM
            or payload.get("source_exam_hash") != exam.get("report_hash")
            or payload.get("source_exam_status") != "PASS_RUNTIME_FINISH_PLAN_FRESH_HOLDOUT"
            or payload.get("source_candidate_strict_success_count")
            != exam.get("metrics", {}).get("candidate_strict_success_count")
            or payload.get("rendered_case_ids")
            != [
                row["case_id"]
                for row in exam["rows"]
                if row["candidate"]["quality"]["strict_chain_passed"]
            ]
            or payload.get("renderer_hash") != _renderer_hash()
            or probe.get("width") != payload.get("width")
            or probe.get("height") != payload.get("height")
            or probe.get("fps") != payload.get("fps")
            or abs(int(probe.get("frame_count", -2)) - int(payload.get("frame_count", 0))) > 1
            or payload.get("visualization_only") is not True
            or payload.get("pixels_used_for_scoring") is not False
            or payload.get("promotion_eligible") is not False
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("commercial_use_allowed") is not False
            or declared != hash_json(payload)
        ):
            raise ValueError("runtime finish video authority contract is invalid")
    finally:
        payload["manifest_hash"] = declared
    return cast(dict[str, Any], payload)


def _load_sources(
    report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], tuple[_Case, ...], dict[str, str]]:
    report = validate_runtime_finish_plan_exam(report_path)
    request_path = report_path.parent / "request.json"
    if not request_path.is_file() or hash_bytes(request_path.read_bytes()) != report.get(
        "request_hash"
    ):
        raise ValueError("runtime finish video exam request changed")
    request = cast(dict[str, Any], json.loads(request_path.read_text(encoding="utf-8")))
    gates = report.get("gates")
    if (
        report.get("schema_version") != "rosclaw_soccer.runtime_finish_plan_exam.v1"
        or report.get("status") != "PASS_RUNTIME_FINISH_PLAN_FRESH_HOLDOUT"
        or report.get("sealed") is not True
        or report.get("promotion_eligible") is not True
        or report.get("partition") != "SEALED_FRESH_HOLDOUT"
        or not isinstance(gates, dict)
        or not gates
        or not all(value is True for value in gates.values())
        or report.get("implementation_hash") != request.get("implementation_hash")
    ):
        raise ValueError("runtime finish video requires a passing sealed exam")
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ValueError("runtime finish video exam rows are missing")
    source_files = {
        str(report_path): hash_bytes(report_path.read_bytes()),
        str(request_path): hash_bytes(request_path.read_bytes()),
    }
    cases: list[_Case] = []
    for index, row in enumerate(rows):
        if not row["candidate"]["quality"]["strict_chain_passed"]:
            continue
        if (
            row.get("exact_replay") is not True
            or row.get("neural_actor_active") is not True
            or row.get("teacher_active") is not False
            or row.get("scripted_contact_active") is not False
        ):
            raise ValueError("runtime finish video strict row authority changed")
        artifact = row["candidate_artifact"]
        trajectory_path = report_path.parent / f"case-{index:03d}" / artifact["file"]
        file_hash = hash_bytes(trajectory_path.read_bytes())
        if file_hash != artifact["file_hash"]:
            raise ValueError("runtime finish video trajectory file changed")
        trajectory = cast(dict[str, np.ndarray], load_three_player_trajectory(trajectory_path))
        digest = trajectory_digest(trajectory)
        if digest != artifact["trajectory_digest"]:
            raise ValueError("runtime finish video trajectory digest changed")
        source_files[str(trajectory_path)] = file_hash
        cases.append(
            _Case(
                row_index=index,
                case_id=str(row["case_id"]),
                result=cast(dict[str, Any], row["candidate"]["result"]),
                trajectory=trajectory,
                trajectory_path=trajectory_path,
                trajectory_hash=file_hash,
                trajectory_digest=digest,
            )
        )
    if len(cases) != 3 or not any(case.result["goalkeeper_save_observed"] for case in cases):
        raise ValueError("runtime finish video needs three strict goal/save cases")
    return report, request, tuple(cases), source_files


def _timeline(cases: tuple[_Case, ...], fps: int) -> tuple[_Clip, ...]:
    clips: list[_Clip] = [
        _Clip(
            "S167 SEALED FRESH HOLDOUT · 3 STRICT SUCCESSES · BASELINE 1",
            tuple(_Frame(0, float(cases[0].trajectory["time"][0]), "wide") for _ in range(fps)),
        )
    ]
    for index, case in enumerate(cases):
        result = case.result
        pass_time = float(result["pass_contact_time_sec"])
        shot_time = float(result["shot_contact_time_sec"])
        outcome = _outcome_time(case)
        outcome_label = (
            "MUJOCO CONTACT SAVE" if result["goalkeeper_save_observed"] else "RIGHT-FOOT GOAL"
        )
        clips.append(
            _Clip(
                f"{case.case_id} · PASS → FINISH · {outcome_label} · "
                f"{float(result['shot_peak_ball_speed_mps']):.2f} m/s",
                _segment(case, index, pass_time - 0.55, outcome + 0.85, 1.0, "wide", fps),
            )
        )
        clips.append(
            _Clip(
                f"{case.case_id} · STRICT CONTACT REPLAY · FROZEN NEURAL CEREBELLUM",
                _segment(case, index, shot_time - 0.45, outcome + 0.45, 0.42, "goal", fps),
            )
        )
    clips.append(
        _Clip(
            "SEALED PASS · +2 STRICT VS FROZEN BASELINE · 6/6 SAFE · 6/6 EXACT REPLAY",
            tuple(
                _Frame(len(cases) - 1, float(cases[-1].trajectory["time"][-1]), "wide")
                for _ in range(round(1.6 * fps))
            ),
        )
    )
    return tuple(clips)


def _outcome_time(case: _Case) -> float:
    if case.result["goalkeeper_save_observed"]:
        return float(case.result["goalkeeper_ball_contact_time_sec"])
    pose = case.trajectory["ball_pose"]
    after_shot = case.trajectory["time"] >= float(case.result["shot_contact_time_sec"])
    crossing = np.flatnonzero(after_shot & (pose[:, 0] >= three_role_goal_spec().plane_x_m))
    if not crossing.size:
        raise ValueError("runtime finish goal trajectory lacks a plane crossing")
    return float(case.trajectory["time"][int(crossing[0])])


def _segment(
    case: _Case,
    case_index: int,
    start: float,
    end: float,
    speed: float,
    view: str,
    fps: int,
) -> tuple[_Frame, ...]:
    lower = max(float(case.trajectory["time"][0]), start)
    upper = min(float(case.trajectory["time"][-1]), end)
    if upper <= lower or speed <= 0.0:
        raise ValueError("runtime finish video segment is invalid")
    count = max(1, int(math.ceil((upper - lower) / speed * fps)))
    return tuple(
        _Frame(case_index, min(upper, lower + index / fps * speed), view) for index in range(count)
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
    for role in ("passer", "shooter", "goalkeeper"):
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


def _write_frames(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    cases: tuple[_Case, ...],
    clips: tuple[_Clip, ...],
    stream: BinaryIO,
) -> None:
    ball_body = _id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_joint = int(model.body_jntadr[ball_body])
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    joints = {
        role: _joint_qpos(mujoco, model, prefix)
        for role, prefix in (("shooter", ""), ("passer", "passer_"), ("goalkeeper", "goalkeeper_"))
    }
    base_qpos = {"shooter": 0}
    for role, prefix in (("passer", "passer_"), ("goalkeeper", "goalkeeper_")):
        joint = _id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint")
        base_qpos[role] = int(model.jnt_qposadr[joint])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    target = np.asarray(
        (
            three_role_goal_spec().plane_x_m,
            three_role_goal_spec().target_y_m,
            three_role_goal_spec().target_z_m,
        ),
        dtype=np.float64,
    )
    for clip in clips:
        for frame in clip.frames:
            case = cases[frame.case_index]
            sampled = _sample(case.trajectory, frame.simulation_time_sec)
            data.qpos[:] = model.qpos0
            for role in ("shooter", "passer", "goalkeeper"):
                data.qpos[base_qpos[role] : base_qpos[role] + 7] = sampled[f"{role}_pelvis_pose"]
                data.qpos[joints[role]] = sampled[f"{role}_joint_position"]
            data.qpos[ball_qpos : ball_qpos + 7] = sampled["ball_pose"]
            mujoco.mj_forward(model, data)
            _set_camera(camera, frame.view, sampled)
            renderer.update_scene(data, camera=camera)
            _add_ball_trail(mujoco, renderer.scene, case.trajectory, int(sampled["index"]))
            append_sphere(mujoco, renderer.scene, target, 0.065, (0.10, 0.95, 0.95, 0.55))
            stream.write(np.ascontiguousarray(renderer.render().copy()).tobytes())


def _set_camera(camera: Any, view: str, sampled: dict[str, Any]) -> None:
    ball = np.asarray(sampled["ball_pose"], dtype=np.float64)
    shooter = np.asarray(sampled["shooter_pelvis_pose"], dtype=np.float64)
    if view == "goal":
        camera.lookat[:] = (three_role_goal_spec().plane_x_m - 0.7, 0.0, 0.82)
        camera.distance, camera.azimuth, camera.elevation = 7.0, 160.0, -7.0
    else:
        camera.lookat[:] = 0.58 * shooter[:3] + 0.42 * ball[:3]
        camera.lookat[2] = 0.72
        camera.distance, camera.azimuth, camera.elevation = 7.2, 112.0, -9.0


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
        f"drawtext={font_option}text='ROSClaw Soccer · PREPARED FINISH GROWTH':expansion=none:"
        f"x={left}:y={round(13 * scale)}:fontsize={round(32 * scale)}:fontcolor=white",
        f"drawtext={font_option}text='SEALED CPU MUJOCO EVIDENCE · SIM ONLY · PIXELS NEVER SCORE':"
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
        "-",
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


def _renderer_hash() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("trajectory_render.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


__all__ = [
    "render_runtime_finish_plan_video",
    "validate_runtime_finish_plan_video_manifest",
]
