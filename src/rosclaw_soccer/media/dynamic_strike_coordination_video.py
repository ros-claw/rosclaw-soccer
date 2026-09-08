"""Render the evidence-bound S209 baseline/learned quick-strike comparison."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from rosclaw_soccer.media.continuous_competitive_match_video import (
    _Clip,
    _Frame,
    _write_frames,
    _write_labels,
)
from rosclaw_soccer.media.independent_team_video import _color_player, _probe
from rosclaw_soccer.media.trajectory_render import escape_filtergraph_option
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.continuous_competitive_match_growth import (
    build_continuous_competitive_fixture,
)
from rosclaw_soccer.training.dynamic_strike_coordination_exam import (
    validate_dynamic_strike_coordination_exam,
)
from rosclaw_soccer.training.dynamic_strike_coordination_probe import (
    validate_dynamic_strike_coordination_probe,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec
from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model

_CLAIM = "DATA_BOUND_QUICK_STRIKE_WITH_PHYSICAL_OPPONENT_SHIN_BLOCK_REPLAY"


def render_dynamic_strike_coordination_video(
    *,
    exam_path: Path,
    asset_root: Path,
    output_path: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render verified trajectories; the pixels never participate in scoring."""

    output = output_path.expanduser().resolve()
    manifest_path = output.with_suffix(".json")
    if (
        output.exists()
        or manifest_path.exists()
        or output.suffix.lower() != ".mp4"
        or not 20 <= fps <= 60
        or not 1280 <= width <= 3840
        or not 720 <= height <= 2160
    ):
        raise ValueError("dynamic strike video output contract is invalid")
    report_path = exam_path.expanduser().resolve()
    report = validate_dynamic_strike_coordination_exam(report_path)
    if report.get("status") != "PASS_FIXED_CONTEXT_LEARNED_QUICK_STRIKE_BLOCK":
        raise ValueError("dynamic strike video requires the passed fixed-context exam")
    paths = cast(dict[str, str], report["source_paths"])
    parent_path = Path(paths["parent_exam"])
    learned_probe_path = Path(paths["learned_primary_probe"])
    learned_probe = validate_dynamic_strike_coordination_probe(learned_probe_path)
    parent = cast(dict[str, Any], report["baseline"])
    learned = cast(dict[str, Any], report["learned"])
    parent_report = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_trajectory_path = parent_path.parent / str(parent_report["primary_artifact"]["file"])
    learned_trajectory_path = learned_probe_path.parent / str(
        learned_probe["trajectory_artifact"]["file"]
    )
    baseline_trajectory = _load_trajectory(parent_trajectory_path)
    learned_trajectory = _load_trajectory(learned_trajectory_path)
    if (
        trajectory_digest(baseline_trajectory) != parent["trajectory_digest"]
        or trajectory_digest(learned_trajectory) != learned["primary_trajectory_digest"]
    ):
        raise ValueError("dynamic strike video trajectory identity changed")
    baseline_assessment = cast(dict[str, Any], parent["assessment"])
    learned_assessment = cast(dict[str, Any], learned["assessment"])
    contact = cast(dict[str, Any], learned["first_contested_contact"])
    baseline_clips = _baseline_timeline(
        baseline_trajectory,
        assessment=baseline_assessment,
        fps=fps,
    )
    learned_clips = _learned_timeline(
        learned_trajectory,
        assessment=learned_assessment,
        block_time=float(contact["time_sec"]),
        fps=fps,
    )
    clips = baseline_clips + learned_clips
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for S209 video")
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ["MUJOCO_GL"] = "egl"
    try:
        fixture = build_continuous_competitive_fixture(asset_root)
        goal = G1TrainingGoalSpec(**cast(dict[str, Any], learned_probe["goal_spec"]))
        import mujoco

        model = build_g1_multi_player_stadium_model(asset_root, players=fixture.players, spec=goal)
        for player in fixture.players:
            _color_player(
                model,
                root_body_name=player.body_prefix + "pelvis",
                rgba=(0.88, 0.05, 0.04, 1.0)
                if player.agent_id.startswith("red.")
                else (0.02, 0.20, 0.92, 1.0),
            )
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-s209-video-") as temp:
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
                    raise RuntimeError("S209 raw video pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        renderer=renderer,
                        players=fixture.players,
                        trajectory=baseline_trajectory,
                        clips=baseline_clips,
                        stream=cast(BinaryIO, process.stdin),
                    )
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        renderer=renderer,
                        players=fixture.players,
                        trajectory=learned_trajectory,
                        clips=learned_clips,
                        stream=cast(BinaryIO, process.stdin),
                    )
                except BrokenPipeError as error:
                    process.stdin.close()
                    process.wait()
                    stderr = (
                        process.stderr.read().decode(errors="replace") if process.stderr else ""
                    )
                    raise RuntimeError(f"S209 ffmpeg closed its input: {stderr[-3000:]}") from error
                except BaseException:
                    process.stdin.close()
                    process.kill()
                    process.wait()
                    raise
                process.stdin.close()
                stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
                if process.wait():
                    raise RuntimeError(f"S209 ffmpeg failed: {stderr[-3000:]}")
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl
    frame_count = sum(len(clip.frames) for clip in clips)
    media_probe = _probe(ffprobe, output)
    if (
        media_probe["width"] != width
        or media_probe["height"] != height
        or media_probe["fps"] != fps
        or abs(media_probe["frame_count"] - frame_count) > 1
    ):
        raise RuntimeError("S209 encoded video contract changed")
    source_files = {
        str(report_path): hash_bytes(report_path.read_bytes()),
        str(parent_trajectory_path): hash_bytes(parent_trajectory_path.read_bytes()),
        str(learned_trajectory_path): hash_bytes(learned_trajectory_path.read_bytes()),
    }
    improvement = cast(dict[str, float], report["improvement"])
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.dynamic_strike_coordination_video.v1",
        "claim": _CLAIM,
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_exam_path": str(report_path),
        "source_exam_hash": report["report_hash"],
        "source_files": source_files,
        "baseline_trajectory_digest": parent["trajectory_digest"],
        "learned_trajectory_digest": learned["primary_trajectory_digest"],
        "receive_to_strike_reduction_sec": improvement["receive_to_strike_reduction_sec"],
        "peak_shot_speed_gain_fraction": improvement["peak_shot_speed_gain_fraction"],
        "ballistic_target_error_reduction_fraction": improvement[
            "ballistic_target_error_reduction_fraction"
        ],
        "defender_geom_name": contact["defender_geom_name"],
        "defender_contact_force_n": contact["contact_force_n"],
        "intentional_block_claimed": False,
        "goal_claimed": False,
        "goalkeeper_save_claimed": False,
        "cross_target_generalization_claimed": False,
        "fixed_context_exact_replay": learned["exact_replay"],
        "whole_body_g1_count": 6,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps,
        "camera_views": sorted({frame.camera for clip in clips for frame in clip.frames}),
        "ball_trail_is_visualization": True,
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "renderer_hash": hash_bytes(Path(__file__).read_bytes()),
    }
    manifest["manifest_hash"] = hash_json(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validate_dynamic_strike_coordination_video_manifest(manifest_path)


def validate_dynamic_strike_coordination_video_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S209 video manifest must be an object")
    declared = payload.pop("manifest_hash", None)
    try:
        video = Path(str(payload.get("video_path"))).expanduser().resolve()
        exam_path = Path(str(payload.get("source_exam_path"))).expanduser().resolve()
        report = validate_dynamic_strike_coordination_exam(exam_path)
        sources = payload.get("source_files")
        if (
            declared != hash_json(payload)
            or payload.get("schema_version")
            != "rosclaw_soccer.dynamic_strike_coordination_video.v1"
            or payload.get("claim") != _CLAIM
            or not video.is_file()
            or hash_bytes(video.read_bytes()) != payload.get("video_hash")
            or payload.get("source_exam_hash") != report.get("report_hash")
            or not isinstance(sources, dict)
            or any(
                not Path(source).is_file() or hash_bytes(Path(source).read_bytes()) != expected
                for source, expected in sources.items()
            )
            or payload.get("fixed_context_exact_replay") is not True
            or payload.get("whole_body_g1_count") != 6
            or payload.get("ball_trail_is_visualization") is not True
            or payload.get("visualization_only") is not True
            or payload.get("pixels_used_for_scoring") is not False
            or payload.get("intentional_block_claimed") is not False
            or payload.get("goal_claimed") is not False
            or payload.get("goalkeeper_save_claimed") is not False
            or payload.get("cross_target_generalization_claimed") is not False
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("commercial_use_allowed") is not False
            or payload.get("renderer_hash") != hash_bytes(Path(__file__).read_bytes())
        ):
            raise ValueError("S209 video integrity or authority contract changed")
    finally:
        payload["manifest_hash"] = declared
    return cast(dict[str, Any], payload)


def _baseline_timeline(
    trajectory: dict[str, np.ndarray], *, assessment: dict[str, Any], fps: int
) -> tuple[_Clip, ...]:
    start = float(trajectory["time"][0])
    end = float(trajectory["time"][-1])
    strike = float(assessment["events"]["strike_time_sec"])
    save = float(assessment["events"]["save_time_sec"])
    return (
        _Clip("S208 PARENT · 4.70 s RECEIVE → STRIKE", _hold(start, 0.9, fps, "wide")),
        _Clip("PHYSICAL PASS + RUN-ONTO RECEIVE", _play(start, 3.30, 1.0, fps, "broadcast")),
        _Clip(
            "BASELINE ORIENT + PLANT",
            _play(2.85, min(end, strike + 0.10), 1.0, fps, "touchline"),
        ),
        _Clip(
            "7.14 m/s SHOT · GOALKEEPER CONTACT",
            _play(strike - 0.18, min(end, save + 0.30), 0.48, fps, "counter"),
        ),
    )


def _learned_timeline(
    trajectory: dict[str, np.ndarray],
    *,
    assessment: dict[str, Any],
    block_time: float,
    fps: int,
) -> tuple[_Clip, ...]:
    start = float(trajectory["time"][0])
    end = float(trajectory["time"][-1])
    strike = float(assessment["events"]["strike_time_sec"])
    return (
        _Clip("S209 LEARNED · DATASET + ACTOR HASH BOUND", _hold(start, 0.9, fps, "wide")),
        _Clip(
            "SAME PASS · LEARNED CONTINUOUS COORDINATION", _play(start, 3.30, 1.0, fps, "broadcast")
        ),
        _Clip(
            "4.38 s RECEIVE → STRIKE · 0.32 s FASTER",
            _play(2.85, min(end, strike + 0.10), 1.0, fps, "touchline"),
        ),
        _Clip(
            "10.88 m/s ON-TARGET INTENT → OPPONENT LEFT-SHIN BLOCK",
            _play(strike - 0.16, min(end, block_time + 0.28), 0.32, fps, "counter"),
        ),
        _Clip(
            "PHYSICAL BLOCK + STABLE RECOVERY · NO GOAL CLAIM",
            _play(max(start, block_time - 0.05), end, 0.72, fps, "broadcast"),
        ),
        _Clip("FIXED-CONTEXT EXACT REPLAY · SIM_ONLY", _hold(end, 1.4, fps, "wide")),
    )


def _hold(timestamp: float, duration_sec: float, fps: int, camera: str) -> tuple[_Frame, ...]:
    return tuple(_Frame(timestamp, camera) for _ in range(round(duration_sec * fps)))


def _play(first: float, last: float, speed: float, fps: int, camera: str) -> tuple[_Frame, ...]:
    duration = max(0.0, last - first) / speed
    return tuple(
        _Frame(min(last, first + index / fps * speed), camera)
        for index in range(max(1, math.ceil(duration * fps)))
    )


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
    scale = height / 720.0
    left = round(28 * scale)
    filters = [
        f"drawbox=x=0:y=0:w=iw:h={round(124 * scale)}:color=0x020714@0.82:t=fill",
        f"drawbox=x=0:y=h-{round(72 * scale)}:w=iw:h={round(72 * scale)}:"
        "color=0x020714@0.84:t=fill",
        f"drawtext=font='DejaVu Sans':text='ROSClaw Soccer · S209 LEARNED QUICK STRIKE':"
        f"expansion=none:x={left}:y={round(12 * scale)}:fontsize={round(31 * scale)}:"
        "fontcolor=white",
        f"drawtext=font='DejaVu Sans':text='CPU MUJOCO · 6 G1 · DATA-BOUND ACTOR · EXACT REPLAY':"
        f"expansion=none:x={left}:y={round(50 * scale)}:fontsize={round(18 * scale)}:"
        "fontcolor=0xFFC857",
        "drawtext=font='DejaVu Sans':text='PASS  |  RECEIVE  |  QUICK STRIKE  |  "
        "PHYSICAL SHIN BLOCK  |  RECOVERY':"
        f"expansion=none:x={left}:y=h-{round(47 * scale)}:fontsize={round(18 * scale)}:"
        "fontcolor=0x8ED9FF",
    ]
    offset = 0.0
    for clip, label in zip(clips, labels, strict=True):
        end = offset + len(clip.frames) / fps
        filters.append(
            f"drawtext=font='DejaVu Sans':textfile={escape_filtergraph_option(str(label))}:"
            f"expansion=none:x={left}:y={round(84 * scale)}:fontsize={round(19 * scale)}:"
            f"fontcolor=0x68F596:enable='between(t,{offset:.6f},{end:.6f})'"
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


def _load_trajectory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exam", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    manifest = render_dynamic_strike_coordination_video(
        exam_path=arguments.exam,
        asset_root=arguments.asset_root,
        output_path=arguments.output,
        fps=arguments.fps,
        width=arguments.width,
        height=arguments.height,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "render_dynamic_strike_coordination_video",
    "validate_dynamic_strike_coordination_video_manifest",
]
