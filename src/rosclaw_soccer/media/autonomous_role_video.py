"""Evidence-downstream video for causal red/blue role autonomy."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.media.active_off_ball_video import _timeline
from rosclaw_soccer.media.three_role_save_portfolio_video import (
    _Clip,
    _probe,
    _write_frames,
    _write_labels,
)
from rosclaw_soccer.media.trajectory_render import escape_filtergraph_option
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets, trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.autonomous_role_growth import validate_autonomous_role_growth
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_three_player_stadium_model

_CLAIM = "AUTONOMOUS_RED_BLUE_ROLES_THREE_G1_STRICT_REPLAY"


def render_autonomous_role_video(
    *,
    evidence_dir: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    root = evidence_dir.expanduser().resolve()
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
        raise ValueError("autonomous role video output contract is invalid")
    report_path = root / "retention/retention-exam.json"
    report = validate_autonomous_role_growth(report_path)
    if report.get("status") != "PASS_AUTONOMOUS_ROLE_GROWTH":
        raise ValueError("autonomous role evidence did not pass")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("autonomous role retention rows are incomplete")
    trajectories: dict[str, dict[str, NDArray[Any]]] = {}
    labels: dict[str, str] = {}
    source_files = {str(report_path): hash_bytes(report_path.read_bytes())}
    required = {
        "time",
        "ball_pose",
        "passer_pelvis_pose",
        "passer_joint_position",
        "shooter_pelvis_pose",
        "shooter_joint_position",
        "goalkeeper_pelvis_pose",
        "goalkeeper_joint_position",
        "passer_role_intent_code",
        "goalkeeper_role_intent_code",
    }
    lane_ids = ("pass-a", "pass-b", "shoot-a", "shoot-b")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not (
            row.get("qualified") is True
            and row.get("safe") is True
            and row.get("exact_replay") is True
        ):
            raise ValueError("autonomous role video selected an unqualified case")
        artifact = row.get("primary_artifact")
        if not isinstance(artifact, dict):
            raise ValueError("autonomous role trajectory artifact is absent")
        path = root / "retention" / f"case-{index:03d}" / str(artifact.get("file"))
        if hash_bytes(path.read_bytes()) != artifact.get("file_hash"):
            raise ValueError("autonomous role trajectory file changed")
        with np.load(path, allow_pickle=False) as archive:
            trajectory = {name: np.asarray(archive[name]) for name in archive.files}
        if required - trajectory.keys() or trajectory_digest(trajectory) != artifact.get(
            "trajectory_digest"
        ):
            raise ValueError("autonomous role trajectory binding changed")
        lane_id = lane_ids[index]
        trajectories[lane_id] = trajectory
        source_files[str(path)] = hash_bytes(path.read_bytes())
        result = row["result"]
        labels[lane_id] = (
            f"{str(row['action']).upper()} · RED SUPPORT "
            f"{100.0 * float(result['teammate_motion']['active_fraction']):.0f}% ACTIVE / "
            f"{int(result['teammate_intent']['switch_count'])} INTENT SWITCHES · "
            f"BLUE PRESS {100.0 * float(result['defender_motion']['active_fraction']):.0f}% ACTIVE"
        )
    clips = _timeline(trajectories, labels, fps)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for autonomous role video")
    output.parent.mkdir(parents=True, exist_ok=True)
    goal = G1TrainingGoalSpec(
        plane_x_m=7.50,
        width_m=3.0,
        height_m=2.0,
        depth_m=1.2,
        target_y_m=-0.80,
        target_z_m=0.35,
        precision_radius_m=0.30,
    )
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        import mujoco

        model = build_g1_three_player_stadium_model(
            asset_root.expanduser().resolve(),
            passer_origin_m=(5.50, -0.40, 0.0),
            goalkeeper_origin_m=(2.00, 0.40, 0.0),
            spec=goal,
        )
        _color_team(model, root_body_name="pelvis", rgba=(0.82, 0.10, 0.08, 1.0))
        _color_team(model, root_body_name="passer_pelvis", rgba=(0.82, 0.10, 0.08, 1.0))
        _color_team(
            model,
            root_body_name="goalkeeper_pelvis",
            rgba=(0.08, 0.28, 0.86, 1.0),
        )
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-autonomous-role-") as temp:
                label_paths = _write_labels(Path(temp), clips)
                process = subprocess.Popen(
                    _ffmpeg_command(
                        ffmpeg=ffmpeg,
                        output=output,
                        width=width,
                        height=height,
                        fps=fps,
                        clips=clips,
                        labels=label_paths,
                    ),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                if process.stdin is None:
                    raise RuntimeError("autonomous role raw-video pipe is unavailable")
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
                    raise RuntimeError(f"autonomous role ffmpeg failed: {stderr[-3000:]}")
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
        raise RuntimeError("autonomous role encoded video contract changed")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.autonomous_role_video.v1",
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_files": source_files,
        "source_report_hash": report["report_hash"],
        "claim": _CLAIM,
        "strict_replay": True,
        "cases_shown": list(trajectories),
        "whole_body_g1_count": 3,
        "red_agent_count": 2,
        "blue_agent_count": 1,
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
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_autonomous_role_video_manifest(manifest_path)
    return manifest


def _color_team(
    model: Any,
    *,
    root_body_name: str,
    rgba: tuple[float, float, float, float],
) -> None:
    """Apply evidence-downstream team colours without changing physics."""

    root = int(model.body(root_body_name).id)
    for geom_id in range(int(model.ngeom)):
        body = int(model.geom_bodyid[geom_id])
        while body > 0 and body != root:
            body = int(model.body_parentid[body])
        if body == root:
            model.geom_matid[geom_id] = -1
            model.geom_rgba[geom_id] = rgba


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
        f"drawtext={font_option}text='ROSClaw Soccer · AUTONOMOUS RED / BLUE ROLES':"
        f"expansion=none:x={left}:y={round(13 * scale)}:fontsize={round(32 * scale)}:"
        "fontcolor=white",
        f"drawtext={font_option}text='10 HZ ROLE INTENT · 50 HZ NEURAL LOCOMOTION · CPU MUJOCO':"
        f"expansion=none:x={left}:y=h-{round(40 * scale)}:fontsize={round(18 * scale)}:"
        "fontcolor=0x8DD8FF",
    ]
    offset = 0.0
    for label, clip in zip(labels, clips, strict=True):
        end = offset + len(clip.frames) / fps
        filters.append(
            f"drawtext={font_option}textfile={escape_filtergraph_option(str(label))}:"
            f"expansion=none:x={left}:y={round(61 * scale)}:fontsize={round(19 * scale)}:"
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


def validate_autonomous_role_video_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("autonomous role video manifest must be an object")
    expected = payload.pop("manifest_hash", None)
    if expected != hash_json(payload):
        raise ValueError("autonomous role video manifest integrity mismatch")
    video = Path(str(payload.get("video_path"))).expanduser().resolve()
    sources = payload.get("source_files")
    numbers = tuple(
        payload.get(name) for name in ("fps", "width", "height", "frame_count", "duration_sec")
    )
    if (
        not video.is_file()
        or hash_bytes(video.read_bytes()) != payload.get("video_hash")
        or not isinstance(sources, dict)
        or payload.get("schema_version") != "rosclaw_soccer.autonomous_role_video.v1"
        or payload.get("claim") != _CLAIM
        or payload.get("whole_body_g1_count") != 3
        or payload.get("red_agent_count") != 2
        or payload.get("blue_agent_count") != 1
        or payload.get("strict_replay") is not True
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
        raise ValueError("autonomous role video authority contract is invalid")
    for source_value, source_hash in sources.items():
        source = Path(source_value).expanduser().resolve()
        if not source.is_file() or hash_bytes(source.read_bytes()) != source_hash:
            raise ValueError("autonomous role video source binding changed")
    payload["manifest_hash"] = expected
    return cast(dict[str, Any], payload)


__all__ = ["render_autonomous_role_video", "validate_autonomous_role_video_manifest"]
