"""Evidence-downstream reel for the S112 Core-closed full-chain champion."""

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

from rosclaw_soccer.media.continuous_second_striker_save_video import (
    _ffmpeg_command,
    _write_frames,
)
from rosclaw_soccer.media.three_role_save_portfolio_video import (
    _Clip,
    _Frame,
    _probe,
    _segment,
    _write_labels,
)
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.current_runtime_prefix_requalification import (
    validate_current_runtime_prefix_requalification,
)
from rosclaw_soccer.world.field import (
    G1TrainingGoalSpec,
    build_g1_four_player_two_ball_stadium_model,
)

_CLAIM = "CROSS_PROCESS_CONTINUOUS_FOUR_G1_CORE_CLOSURE_CHAMPION_VIDEO"


def _implementation_hash() -> str:
    source = Path(__file__)
    dependency = source.with_name("continuous_second_striker_save_video.py")
    return str(
        hash_json(
            {
                source.name: hash_bytes(source.read_bytes()),
                dependency.name: hash_bytes(dependency.read_bytes()),
            }
        )
    )


def validate_current_runtime_requalification_video(path: Path) -> dict[str, Any]:
    """Validate video bytes, physics sources and non-hardware authority."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("current-runtime video manifest must be an object")
    expected = payload.pop("manifest_hash", None)
    try:
        video_value = payload.get("video_path")
        sources = payload.get("source_files")
        if not isinstance(video_value, str) or not isinstance(sources, dict):
            raise ValueError("current-runtime video bindings are invalid")
        video = Path(video_value).expanduser().resolve()
        if not video.is_file() or payload.get("video_hash") != hash_bytes(video.read_bytes()):
            raise ValueError("current-runtime video bytes changed")
        for source_value, source_hash in sources.items():
            source = Path(source_value).expanduser().resolve()
            if not source.is_file() or hash_bytes(source.read_bytes()) != source_hash:
                raise ValueError("current-runtime video source binding changed")
        numeric = tuple(
            payload.get(name) for name in ("fps", "width", "height", "frame_count", "duration_sec")
        )
        replay_count = payload.get("cross_process_replay_count")
        if (
            payload.get("schema_version")
            != "rosclaw_soccer.current_runtime_requalification_video.v2"
            or payload.get("claim") != _CLAIM
            or payload.get("evidence_passed") is not True
            or payload.get("strict_cross_process_replay") is not True
            or not isinstance(replay_count, int)
            or replay_count < 2
            or payload.get("four_g1_visible") is not True
            or payload.get("two_physical_balls_visible") is not True
            or payload.get("two_physical_saves") is not True
            or payload.get("visualization_only") is not True
            or payload.get("pixels_used_for_scoring") is not False
            or payload.get("promotion_eligible") is not False
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("commercial_use_allowed") is not False
            or payload.get("implementation_hash") != _implementation_hash()
            or expected != hash_json(payload)
            or any(
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in numeric
            )
        ):
            raise ValueError("current-runtime video authority or integrity contract is invalid")
    finally:
        if expected is not None:
            payload["manifest_hash"] = expected
    return cast(dict[str, Any], payload)


def render_current_runtime_requalification_video(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render one frozen worker trajectory; pixels never enter qualification."""

    evidence_file = evidence_path.expanduser().resolve()
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
        raise ValueError("current-runtime video output contract is invalid")
    evidence = validate_current_runtime_prefix_requalification(evidence_file)
    if evidence.get("passed") is not True or evidence.get("strict_replay") is not True:
        raise ValueError("current-runtime evidence is not render eligible")
    request_path = evidence_file.parent / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if evidence.get("request_hash") != hash_bytes(request_path.read_bytes()):
        raise ValueError("current-runtime video request binding changed")
    trajectory_path = evidence_file.parent / str(evidence["trajectory_file"])
    if evidence.get("trajectory_hash") != hash_bytes(trajectory_path.read_bytes()):
        raise ValueError("current-runtime video trajectory binding changed")
    goal = G1TrainingGoalSpec(**cast(dict[str, Any], request["goal_spec"]))
    if abs(goal.width_m - 7.32) > 1.0e-9 or abs(goal.height_m - 2.44) > 1.0e-9:
        raise ValueError("current-runtime video requires a regulation goal")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        raw = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "time",
        "ball_pose",
        "second_ball_pose",
        "shooter_pelvis_pose",
        "shooter_joint_position",
        "passer_pelvis_pose",
        "passer_joint_position",
        "goalkeeper_pelvis_pose",
        "goalkeeper_joint_position",
        "second_striker_pelvis_pose",
        "second_striker_joint_position",
    }
    if not required <= set(raw):
        raise ValueError("current-runtime video trajectory is incomplete")
    trajectory = dict(raw)
    trajectory["first_ball_pose"] = raw["ball_pose"]
    trajectory["source_shooter_pelvis_pose"] = raw["shooter_pelvis_pose"]
    trajectory["source_shooter_joint_position"] = raw["shooter_joint_position"]
    continuous = cast(dict[str, Any], evidence["continuous"])
    result = cast(dict[str, Any], continuous["result"])
    clips = _timeline(trajectory, result, evidence, fps)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for the S112 video")
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        if qualification.body_hash != request.get("body_hash"):
            raise ValueError("current-runtime video Body hash changed")
        import mujoco

        striker = cast(dict[str, Any], request["case"])["striker"]
        model = build_g1_four_player_two_ball_stadium_model(
            asset_root,
            passer_origin_m=(5.10, -0.864060065039216, 0.0),
            goalkeeper_origin_m=(goal.plane_x_m - 0.30, 0.0, 0.0),
            second_striker_origin_m=tuple(striker["origin_m"]),
            first_ball_origin_m=(1.0, 0.0, goal.ball_radius_m),
            second_ball_origin_m=tuple(striker["ball_origin_m"]),
            spec=goal,
        )
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-s112-video-") as temp:
                labels = _write_labels(Path(temp), clips)
                process = subprocess.Popen(
                    _s111_ffmpeg_command(
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
                    raise RuntimeError("S112 raw-video pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        renderer=renderer,
                        trajectory=trajectory,
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
                    raise RuntimeError(f"S112 ffmpeg failed: {stderr[-3000:]}")
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
        raise RuntimeError("current-runtime encoded video contract changed")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.current_runtime_requalification_video.v2",
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_files": {
            str(evidence_file): hash_bytes(evidence_file.read_bytes()),
            str(request_path): hash_bytes(request_path.read_bytes()),
            str(trajectory_path): hash_bytes(trajectory_path.read_bytes()),
        },
        "claim": _CLAIM,
        "evidence_report_hash": evidence["report_hash"],
        "evidence_passed": True,
        "strict_cross_process_replay": True,
        "cross_process_replay_count": evidence["cross_process_replay_count"],
        "trajectory_digest": evidence["worker_reports"][0]["trajectory_digest"],
        "four_g1_visible": True,
        "two_physical_balls_visible": True,
        "two_physical_saves": True,
        "first_glove_contact_height_m": result["goalkeeper_glove_contact_height_m"],
        "second_glove_contact_height_m": result["goalkeeper_second_glove_contact_height_m"],
        "second_striker_contact_force_peak_n": result["second_striker_contact_force_peak_n"],
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
        "implementation_hash": _implementation_hash(),
    }
    manifest["manifest_hash"] = hash_json(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return validate_current_runtime_requalification_video(manifest_path)


def _timeline(
    trajectory: dict[str, np.ndarray],
    result: dict[str, Any],
    evidence: dict[str, Any],
    fps: int,
) -> tuple[_Clip, ...]:
    start = float(trajectory["time"][0])
    end = float(trajectory["time"][-1])
    first = float(result["goalkeeper_glove_contact_time_sec"])
    rearm = float(result["second_threat_rearm_time_sec"])
    strike = float(result["second_striker_contact_time_sec"])
    second = float(result["goalkeeper_second_glove_contact_time_sec"])
    force = float(result["second_striker_contact_force_peak_n"])
    first_height = float(result["goalkeeper_glove_contact_height_m"])
    second_height = float(result["goalkeeper_second_glove_contact_height_m"])
    replays = int(evidence["cross_process_replay_count"])
    title = tuple(_Frame("left-inner", start, "four") for _ in range(round(1.8 * fps)))
    final = tuple(_Frame("left-inner", end, "goal") for _ in range(round(2.0 * fps)))
    return (
        _Clip(f"S112 · CORE CLOSURE · {replays} FRESH BYTE-IDENTICAL PROCESSES", title),
        _Clip(
            "PASS → FIRST G1 HIGH SHOT → AIRBORNE SAVE",
            _segment("left-inner", 4.4, first + 0.45, 1.0, "four", fps),
        ),
        _Clip(
            f"FIRST TRUE GLOVE SAVE · {first_height:.3f} m · CONTACT-GROUNDED",
            _segment("left-inner", first - 0.50, first + 0.55, 0.34, "goal", fps),
        ),
        _Clip(
            "MEASURED RECOVERY → READY → PHYSICAL HANDOFF",
            _segment("left-inner", first + 0.55, rearm + 0.10, 2.35, "goal", fps),
        ),
        _Clip(
            "FOURTH G1 APPROACH · LEARNED ACTOR + MUSCLE MEMORY",
            _segment("left-inner", 12.0, strike - 0.28, 1.35, "striker", fps),
        ),
        _Clip(
            f"ANATOMICAL RIGHT-FOOT STRIKE · {force:.0f} N · NO CANNON",
            _segment("left-inner", strike - 0.42, second + 0.28, 0.42, "contact", fps),
        ),
        _Clip(
            f"SECOND PHYSICAL GLOVE SAVE · {second_height:.3f} m · BALL CLEARED",
            _segment("left-inner", second - 0.42, second + 0.78, 0.31, "goal", fps),
        ),
        _Clip(
            "SECOND SAVE → DOUBLE SUPPORT → FINAL READY",
            _segment("left-inner", second + 0.55, 23.5, 1.45, "goal", fps),
        ),
        _Clip(
            "ONE CLOCK · FOUR G1 · TWO BALLS · ZERO RESET OR TELEPORT",
            _segment("left-inner", start, end, 2.1, "four", fps),
        ),
        _Clip("CORE-CLOSED CHAMPION · SIM ONLY · CROSS-PROCESS VERIFIED", final),
    )


def _s111_ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    width: int,
    height: int,
    fps: int,
    clips: tuple[_Clip, ...],
    labels: tuple[Path, ...],
) -> list[str]:
    command = _ffmpeg_command(
        ffmpeg=ffmpeg,
        output=output,
        width=width,
        height=height,
        fps=fps,
        clips=clips,
        labels=labels,
    )
    filter_index = command.index("-vf") + 1
    command[filter_index] = command[filter_index].replace(
        "ROSClaw Soccer · S109 PHYSICAL SECOND STRIKER",
        "ROSClaw Soccer · S112 CORE-CLOSED CHAMPION",
    )
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = render_current_runtime_requalification_video(
        evidence_path=args.evidence,
        asset_root=args.asset_root,
        output_path=args.output,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = [
    "render_current_runtime_requalification_video",
    "validate_current_runtime_requalification_video",
]
