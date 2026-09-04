"""Honest diagnostic reel for consumed continuous finisher-plan growth."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from rosclaw_soccer.evidence.three_player import load_three_player_trajectory
from rosclaw_soccer.media.runtime_finish_plan_video import (
    _Case,
    _Clip,
    _Frame,
    _timeline,
    _write_frames,
    _write_labels,
)
from rosclaw_soccer.media.three_role_save_portfolio_video import _probe
from rosclaw_soccer.media.trajectory_render import escape_filtergraph_option
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

_CLAIM = "CONSUMED_DEVELOPMENT_FINISH_PLAN_GROWTH"


def render_continuous_finish_plan_development_video(
    *,
    exam_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render strict development outcomes without making a sealed-evidence claim."""

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
        raise ValueError("continuous finish development video output contract is invalid")
    report_path = exam_path.expanduser().resolve()
    report, request, cases, source_files = _load_development_sources(report_path)
    clips = list(_timeline(cases, fps))
    clips[0] = replace(
        clips[0],
        label="S175 CONSUMED DEVELOPMENT · 4/6 STRICT · BASELINE 1/6",
    )
    clips[-1] = _Clip(
        "TRAINING RECOVERY ONLY · 6/6 SAFE · 6/6 EXACT REPLAY · FRESH EXAM PENDING",
        tuple(
            _Frame(len(cases) - 1, float(cases[-1].trajectory["time"][-1]), "wide")
            for _ in range(round(1.8 * fps))
        ),
    )
    active_clips = tuple(clips)
    goal = three_role_goal_spec()
    keeper = three_role_development_kwargs()["goalkeeper_config"]
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for continuous finish video")
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ["MUJOCO_GL"] = "egl"
    try:
        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        if request.get("body_hash") != qualification.body_hash:
            raise ValueError("continuous finish video body identity changed")
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
            with tempfile.TemporaryDirectory(prefix="rosclaw-continuous-finish-video-") as temp:
                labels = _write_labels(Path(temp), active_clips)
                process = subprocess.Popen(
                    _ffmpeg_command(
                        ffmpeg=ffmpeg,
                        output=output,
                        width=width,
                        height=height,
                        fps=fps,
                        clips=active_clips,
                        labels=labels,
                    ),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                if process.stdin is None:
                    raise RuntimeError("continuous finish video raw pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        renderer=renderer,
                        cases=cases,
                        clips=active_clips,
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
                    raise RuntimeError(
                        f"continuous finish development video ffmpeg failed: {stderr[-3000:]}"
                    )
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    probe = _probe(ffprobe, output)
    frame_count = sum(len(clip.frames) for clip in active_clips)
    if (
        probe["width"] != width
        or probe["height"] != height
        or probe["fps"] != fps
        or abs(probe["frame_count"] - frame_count) > 1
    ):
        raise RuntimeError("continuous finish development video encoding changed")
    precise_count = sum(
        bool(
            case.result["goal_crossed"]
            and isinstance(case.result.get("target_error_m"), int | float)
            and float(case.result["target_error_m"]) <= 0.10
        )
        for case in cases
    )
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.continuous_finish_plan_development_video.v1",
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
        "source_candidate_precise_goal_count": precise_count,
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
        "fresh_generalization_claimed": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
    }
    manifest["manifest_hash"] = hash_json(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    validate_continuous_finish_plan_development_video(manifest_path)
    return manifest


def validate_continuous_finish_plan_development_video(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("continuous finish video manifest must be an object")
    declared = payload.pop("manifest_hash", None)
    try:
        video = Path(str(payload.get("video_path"))).expanduser().resolve()
        exam_path = Path(str(payload.get("source_exam_path"))).expanduser().resolve()
        sources = payload.get("source_files")
        if (
            not video.is_file()
            or hash_bytes(video.read_bytes()) != payload.get("video_hash")
            or not isinstance(sources, dict)
            or any(
                not Path(source).is_file() or hash_bytes(Path(source).read_bytes()) != expected
                for source, expected in sources.items()
            )
        ):
            raise ValueError("continuous finish development video sources changed")
        report = validate_runtime_finish_plan_exam(exam_path)
        probe = _probe(shutil.which("ffprobe") or "", video)
        if (
            payload.get("schema_version")
            != "rosclaw_soccer.continuous_finish_plan_development_video.v1"
            or payload.get("claim") != _CLAIM
            or payload.get("source_exam_hash") != report.get("report_hash")
            or payload.get("source_exam_status") != "PASS_RUNTIME_FINISH_PLAN_DEVELOPMENT"
            or payload.get("source_candidate_strict_success_count") != 4
            or payload.get("source_candidate_precise_goal_count") != 1
            or probe.get("width") != payload.get("width")
            or probe.get("height") != payload.get("height")
            or probe.get("fps") != payload.get("fps")
            or abs(int(probe.get("frame_count", -2)) - int(payload.get("frame_count", 0))) > 1
            or payload.get("visualization_only") is not True
            or payload.get("pixels_used_for_scoring") is not False
            or payload.get("promotion_eligible") is not False
            or payload.get("fresh_generalization_claimed") is not False
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("commercial_use_allowed") is not False
            or payload.get("renderer_hash") != _renderer_hash()
            or declared != hash_json(payload)
        ):
            raise ValueError("continuous finish development video contract is invalid")
    finally:
        payload["manifest_hash"] = declared
    return cast(dict[str, Any], payload)


def _load_development_sources(
    report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], tuple[_Case, ...], dict[str, str]]:
    report = validate_runtime_finish_plan_exam(report_path)
    request_path = report_path.parent / "request.json"
    request = cast(dict[str, Any], json.loads(request_path.read_text(encoding="utf-8")))
    if (
        report.get("status") != "PASS_RUNTIME_FINISH_PLAN_DEVELOPMENT"
        or report.get("sealed") is not False
        or report.get("promotion_eligible") is not False
        or not all(value is True for value in cast(dict[str, bool], report["gates"]).values())
    ):
        raise ValueError("continuous finish video requires passing consumed development")
    source_files = {
        str(report_path): hash_bytes(report_path.read_bytes()),
        str(request_path): hash_bytes(request_path.read_bytes()),
    }
    cases: list[_Case] = []
    for index, row in enumerate(cast(list[dict[str, Any]], report["rows"])):
        if not row["candidate"]["quality"]["strict_chain_passed"]:
            continue
        artifact = cast(dict[str, Any], row["candidate_artifact"])
        trajectory_path = report_path.parent / f"case-{index:03d}" / str(artifact["file"])
        file_hash = hash_bytes(trajectory_path.read_bytes())
        trajectory = cast(dict[str, np.ndarray], load_three_player_trajectory(trajectory_path))
        digest = trajectory_digest(trajectory)
        if (
            row.get("exact_replay") is not True
            or row.get("neural_actor_active") is not True
            or row.get("teacher_active") is not False
            or row.get("scripted_contact_active") is not False
            or file_hash != artifact.get("file_hash")
            or digest != artifact.get("trajectory_digest")
        ):
            raise ValueError("continuous finish video trajectory binding changed")
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
    if len(cases) != 4 or not any(case.result["goalkeeper_save_observed"] for case in cases):
        raise ValueError("continuous finish development video needs four strict outcomes")
    return report, request, tuple(cases), source_files


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
        f"drawtext={font_option}text='ROSClaw Soccer · CONTINUAL FINISH GROWTH':"
        f"expansion=none:x={left}:y={round(13 * scale)}:fontsize={round(32 * scale)}:"
        "fontcolor=white",
        f"drawtext={font_option}text='CONSUMED CPU MUJOCO DEVELOPMENT · SIM ONLY · "
        f"PIXELS NEVER SCORE':expansion=none:x={left}:y=h-{round(40 * scale)}:"
        f"fontsize={round(18 * scale)}:fontcolor=0x8DD8FF",
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
    for path in (
        Path(__file__),
        Path(__file__).with_name("runtime_finish_plan_video.py"),
        Path(__file__).with_name("trajectory_render.py"),
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


__all__ = [
    "render_continuous_finish_plan_development_video",
    "validate_continuous_finish_plan_development_video",
]
