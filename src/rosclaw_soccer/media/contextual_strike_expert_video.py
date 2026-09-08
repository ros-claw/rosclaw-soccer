"""Render the honest S210 contextual-strike growth and holdout comparison."""

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
from rosclaw_soccer.training.contextual_strike_expert_exam import (
    validate_contextual_strike_expert_exam,
)
from rosclaw_soccer.training.contextual_strike_expert_probe import (
    validate_contextual_strike_expert_probe,
)
from rosclaw_soccer.training.continuous_competitive_match_growth import (
    build_continuous_competitive_fixture,
)
from rosclaw_soccer.training.dynamic_strike_coordination_probe import (
    validate_dynamic_strike_coordination_probe,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec
from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model

_CLAIM = "VISUALIZED_DAGGER_ANCHOR_REPAIR_WITH_HELDOUT_FAILURE_DISCLOSURE"


def render_contextual_strike_expert_video(
    *,
    exam_path: Path,
    asset_root: Path,
    output_path: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render selected verified trajectories; pixels never participate in scoring."""

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
        raise ValueError("contextual strike video output contract is invalid")
    report_path = exam_path.expanduser().resolve()
    report = validate_contextual_strike_expert_exam(report_path)
    if (
        report.get("status") != "PASS_DAGGER_MEMORY_REJECT_HELDOUT_GENERALIZATION"
        or report.get("evidence_passed") is not True
        or report.get("promotion_eligible") is not False
    ):
        raise ValueError("contextual strike video requires the honest passed evidence exam")
    paths = cast(dict[str, str], report["source_paths"])
    baseline_root = Path(paths["baseline_probe_dir"])
    primary_root = Path(paths["primary_probe_dir"])
    selected = (
        ("baseline0700", baseline_root / "gy070" / "probe.json", False),
        ("training0700", primary_root / "train0700" / "probe.json", True),
        ("training0800", primary_root / "train0800" / "probe.json", True),
        ("holdout07125", primary_root / "hold07125" / "probe.json", True),
        ("holdout07375", primary_root / "hold07375" / "probe.json", False),
    )
    reports: dict[str, dict[str, Any]] = {}
    trajectories: dict[str, dict[str, np.ndarray]] = {}
    source_files = {str(report_path): hash_bytes(report_path.read_bytes())}
    for name, probe_path, expected_success in selected:
        probe = (
            validate_dynamic_strike_coordination_probe(probe_path)
            if name == "baseline0700"
            else validate_contextual_strike_expert_probe(probe_path)
        )
        success = bool(
            _baseline_success(probe) if name == "baseline0700" else probe["candidate_success"]
        )
        if success is not expected_success:
            raise ValueError(f"contextual strike video result changed: {name}")
        artifact = cast(dict[str, Any], probe["trajectory_artifact"])
        trajectory_path = probe_path.parent / str(artifact["file"])
        trajectory = _load_trajectory(trajectory_path)
        if trajectory_digest(trajectory) != probe["trajectory_digest"]:
            raise ValueError(f"contextual strike video trajectory changed: {name}")
        reports[name] = probe
        trajectories[name] = trajectory
        source_files[str(probe_path)] = hash_bytes(probe_path.read_bytes())
        source_files[str(trajectory_path)] = hash_bytes(trajectory_path.read_bytes())
    segments = _segments(trajectories=trajectories, reports=reports, fps=fps)
    clips = tuple(clip for _, clip_group in segments for clip in clip_group)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for S210 video")
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ["MUJOCO_GL"] = "egl"
    try:
        fixture = build_continuous_competitive_fixture(asset_root)
        goal = G1TrainingGoalSpec(**cast(dict[str, Any], reports["training0800"]["goal_spec"]))
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
            with tempfile.TemporaryDirectory(prefix="rosclaw-s210-video-") as temp:
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
                    raise RuntimeError("S210 raw video pipe is unavailable")
                try:
                    for name, clip_group in segments:
                        _write_frames(
                            mujoco=mujoco,
                            model=model,
                            data=data,
                            renderer=renderer,
                            players=fixture.players,
                            trajectory=trajectories[name],
                            clips=clip_group,
                            stream=cast(BinaryIO, process.stdin),
                        )
                except BrokenPipeError as error:
                    process.stdin.close()
                    process.wait()
                    stderr = (
                        process.stderr.read().decode(errors="replace") if process.stderr else ""
                    )
                    raise RuntimeError(f"S210 ffmpeg closed its input: {stderr[-3000:]}") from error
                except BaseException:
                    process.stdin.close()
                    process.kill()
                    process.wait()
                    raise
                process.stdin.close()
                stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
                if process.wait():
                    raise RuntimeError(f"S210 ffmpeg failed: {stderr[-3000:]}")
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
        raise RuntimeError("S210 encoded video contract changed")
    summary = cast(dict[str, Any], report["summary"])
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.contextual_strike_expert_video.v1",
        "claim": _CLAIM,
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_exam_path": str(report_path),
        "source_exam_hash": report["report_hash"],
        "source_files": source_files,
        "trajectory_digests": {name: probe["trajectory_digest"] for name, probe in reports.items()},
        "training_success_rate": summary["training_success_rate"],
        "holdout_success_rate": summary["holdout_success_rate"],
        "exact_replays": summary["exact_replays"],
        "exact_replay_cases": summary["exact_replay_cases"],
        "anchor_failure_repair_visualized": True,
        "holdout_failure_disclosed": True,
        "continuous_target_generalization_claimed": False,
        "promotion_eligible": False,
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
    _atomic_json(manifest_path, manifest)
    return validate_contextual_strike_expert_video_manifest(manifest_path)


def validate_contextual_strike_expert_video_manifest(path: Path) -> dict[str, Any]:
    """Validate media bytes, source evidence, renderer, and claim boundary."""

    payload = _read_json(path.expanduser().resolve())
    declared = payload.pop("manifest_hash", None)
    try:
        video = Path(str(payload.get("video_path"))).expanduser().resolve()
        exam_path = Path(str(payload.get("source_exam_path"))).expanduser().resolve()
        report = validate_contextual_strike_expert_exam(exam_path)
        sources = payload.get("source_files")
        summary = cast(dict[str, Any], report["summary"])
        if (
            declared != hash_json(payload)
            or payload.get("schema_version") != "rosclaw_soccer.contextual_strike_expert_video.v1"
            or payload.get("claim") != _CLAIM
            or not video.is_file()
            or hash_bytes(video.read_bytes()) != payload.get("video_hash")
            or payload.get("source_exam_hash") != report.get("report_hash")
            or not isinstance(sources, dict)
            or any(
                not Path(source).is_file() or hash_bytes(Path(source).read_bytes()) != expected
                for source, expected in sources.items()
            )
            or payload.get("training_success_rate") != summary.get("training_success_rate")
            or payload.get("holdout_success_rate") != summary.get("holdout_success_rate")
            or payload.get("exact_replays") != summary.get("exact_replays")
            or payload.get("anchor_failure_repair_visualized") is not True
            or payload.get("holdout_failure_disclosed") is not True
            or payload.get("continuous_target_generalization_claimed") is not False
            or payload.get("promotion_eligible") is not False
            or payload.get("whole_body_g1_count") != 6
            or payload.get("ball_trail_is_visualization") is not True
            or payload.get("visualization_only") is not True
            or payload.get("pixels_used_for_scoring") is not False
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("commercial_use_allowed") is not False
            or payload.get("renderer_hash") != hash_bytes(Path(__file__).read_bytes())
        ):
            raise ValueError("S210 video integrity or authority contract changed")
    finally:
        payload["manifest_hash"] = declared
    return payload


def _segments(
    *,
    trajectories: dict[str, dict[str, np.ndarray]],
    reports: dict[str, dict[str, Any]],
    fps: int,
) -> tuple[tuple[str, tuple[_Clip, ...]], ...]:
    baseline = trajectories["baseline0700"]
    repaired = trajectories["training0700"]
    high = trajectories["training0800"]
    holdout_pass = trajectories["holdout07125"]
    holdout_fail = trajectories["holdout07375"]
    baseline_strike = _strike_time(reports["baseline0700"])
    repaired_strike = _strike_time(reports["training0700"])
    high_strike = _strike_time(reports["training0800"])
    pass_strike = _strike_time(reports["holdout07125"])
    fail_strike = _strike_time(reports["holdout07375"])
    return (
        (
            "baseline0700",
            (
                _Clip("S210 · FAILURE-DRIVEN CONTEXT MEMORY", _hold(0.0, 1.1, fps, "wide")),
                _Clip(
                    "GLOBAL S209 ACTOR · y=0.700 m · UNSAFE + OFF TARGET",
                    _play(0.0, float(baseline["time"][-1]), 1.25, fps, "broadcast"),
                ),
                _Clip(
                    "FAILURE RECORDED · NO SILENT PROMOTION",
                    _play(baseline_strike - 0.22, baseline_strike + 0.65, 0.38, fps, "counter"),
                ),
            ),
        ),
        (
            "training0700",
            (
                _Clip(
                    "DAGGER REPAIR · LOCAL VERIFIED EXPERT y=0.700 m",
                    _play(0.0, float(repaired["time"][-1]), 1.15, fps, "broadcast"),
                ),
                _Clip(
                    "PHYSICAL PASS → RECEIVE → STRIKE → STABLE RECOVERY",
                    _play(repaired_strike - 0.32, repaired_strike + 1.00, 0.42, fps, "touchline"),
                ),
            ),
        ),
        (
            "training0800",
            (
                _Clip(
                    "RETAINED HIGH TARGET · y=0.800 m · 10.88 m/s",
                    _play(2.70, float(high["time"][-1]), 0.92, fps, "counter"),
                ),
                _Clip(
                    "LOCAL EXPERT LATCHED ONCE · NO PER-FRAME SWITCHING",
                    _play(high_strike - 0.24, high_strike + 0.85, 0.38, fps, "broadcast"),
                ),
            ),
        ),
        (
            "holdout07125",
            (
                _Clip(
                    "UNSEEN MIDPOINT y=0.7125 m · PASS",
                    _play(2.70, float(holdout_pass["time"][-1]), 0.92, fps, "touchline"),
                ),
                _Clip(
                    "ONE OF 3/8 HELD-OUT SUCCESSES",
                    _play(pass_strike - 0.24, pass_strike + 0.85, 0.38, fps, "counter"),
                ),
            ),
        ),
        (
            "holdout07375",
            (
                _Clip(
                    "UNSEEN MIDPOINT y=0.7375 m · FAILURE DISCLOSED",
                    _play(2.70, float(holdout_fail["time"][-1]), 0.92, fps, "broadcast"),
                ),
                _Clip(
                    "HELD-OUT GENERALIZATION 3/8 · PROMOTION REJECTED",
                    _play(fail_strike - 0.24, fail_strike + 0.85, 0.38, fps, "counter"),
                ),
                _Clip(
                    "18/18 EXACT REPLAYS · CPU MUJOCO · SIM_ONLY",
                    _hold(float(holdout_fail["time"][-1]), 1.5, fps, "wide"),
                ),
            ),
        ),
    )


def _baseline_success(report: dict[str, Any]) -> bool:
    assessment = cast(dict[str, Any], report["assessment"])
    gates = cast(dict[str, bool], assessment["gates"])
    projection = cast(dict[str, Any], report["shot_projection"])
    return bool(
        report["world_safe"] is True
        and assessment["phase_sequence"] == [1, 2, 3, 4, 5, 6]
        and projection["whole_ball_inside_goal"] is True
        and gates.get("physical_teammate_pass_received") is True
        and gates.get("physical_foot_strike_in_strike_phase") is True
        and gates.get("stable_recovery_completed") is True
    )


def _strike_time(report: dict[str, Any]) -> float:
    assessment = cast(dict[str, Any], report["assessment"])
    events = cast(dict[str, Any], assessment["events"])
    return float(events["strike_time_sec"])


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
        f"drawtext=font='DejaVu Sans':text='ROSClaw Soccer · S210 CONTEXTUAL GROWTH':"
        f"expansion=none:x={left}:y={round(12 * scale)}:fontsize={round(31 * scale)}:"
        "fontcolor=white",
        f"drawtext=font='DejaVu Sans':text='CPU MUJOCO · 6 G1 · FAILURE MEMORY · EXACT REPLAY':"
        f"expansion=none:x={left}:y={round(50 * scale)}:fontsize={round(18 * scale)}:"
        "fontcolor=0xFFC857",
        "drawtext=font='DejaVu Sans':text='ANCHORS 9/9  |  HOLDOUT 3/8  |  PROMOTION REJECTED':"
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("S210 video manifest must be an object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".json", mode="w", encoding="utf-8", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


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
    manifest = render_contextual_strike_expert_video(
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
    "render_contextual_strike_expert_video",
    "validate_contextual_strike_expert_video_manifest",
]
