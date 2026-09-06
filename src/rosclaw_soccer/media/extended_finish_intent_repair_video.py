"""Render the S205 failed finish beside its learned stable repair."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, cast

import numpy as np

from rosclaw_soccer.media.three_player_video import (
    ThreePlayerVideoClip,
    _configure_offscreen_framebuffer,
    _ffmpeg_command,
    _probe_video,
    _segment,
    _timelines,
    _write_frames,
)
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.development_evidence import (
    three_role_development_kwargs,
    three_role_goal_spec,
)
from rosclaw_soccer.training.extended_finish_intent_repair import (
    validate_extended_finish_intent_repair,
)
from rosclaw_soccer.world.field import build_g1_three_player_stadium_model

_CLAIM = "S205_FAILURE_FEEDBACK_STABLE_FINISH_REPAIR_VISUALIZATION"


def render_extended_finish_intent_repair_video(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    fps: int = 30,
) -> dict[str, Any]:
    """Render state-only evidence after S205 passes independent validation."""

    report_path = evidence_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    manifest_path = output.with_suffix(".json")
    if output.suffix.lower() != ".mp4" or output.exists() or manifest_path.exists():
        raise ValueError("S205 video output must be a new MP4 path")
    if not 10 <= fps <= 60:
        raise ValueError("S205 video fps must be in [10, 60]")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for S205 video")
    report = validate_extended_finish_intent_repair(report_path)
    if report.get("status") != "PASS_EXTENDED_FINISH_INTENT_REPAIR":
        raise ValueError("S205 video requires passing repair evidence")
    request_path = report_path.parent / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    source_path = _source_s204_path(request, report_path)
    source_request_path = source_path.parent / "request.json"
    source_request = json.loads(source_request_path.read_text(encoding="utf-8"))
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if qualification.body_hash != source_request["body_hash"]:
        raise ValueError("S205 video body does not match physical evidence")

    baseline_record = cast(dict[str, Any], report["baseline"]["trajectory"])
    selected_record = cast(dict[str, Any], report["selected_replay"]["trajectory"])
    baseline_trajectory = _load_trajectory(report_path.parent, baseline_record)
    selected_trajectory = _load_trajectory(report_path.parent, selected_record)
    goal = three_role_goal_spec()
    development = three_role_development_kwargs()
    keeper = development["goalkeeper_config"]
    context = cast(dict[str, Any], request["context"])
    model = build_g1_three_player_stadium_model(
        asset_root.expanduser().resolve(),
        passer_origin_m=tuple(development["passer_origin"]),
        passer_yaw_rad=float(context["passer_yaw_rad"]),
        goalkeeper_origin_m=(
            goal.plane_x_m - keeper.depth_from_goal_line_m,
            keeper.initial_lateral_position_m,
            0.0,
        ),
        spec=goal,
    )
    width, height = 1920, 1080
    _configure_offscreen_framebuffer(model, width=width, height=height)
    baseline_bundle = _bundle(
        physical_target=source_request["physical_target_m"],
        goal=goal,
        result=report["baseline"]["result"],
        trajectory=baseline_trajectory,
        passed=False,
    )
    selected_bundle = _bundle(
        physical_target=source_request["physical_target_m"],
        goal=goal,
        result=report["selected_replay"]["result"],
        trajectory=selected_trajectory,
        passed=True,
    )
    baseline_timelines, _ = _timelines(baseline_bundle, fps)
    selected_timelines, _ = _timelines(selected_bundle, fps)
    selected_result = cast(dict[str, Any], report["selected_replay"]["result"])
    recovery_timeline = _segment(
        float(selected_result["shot_contact_time_sec"]) + 0.75,
        float(selected_trajectory["time"][-1]),
        0.65,
        "recovery_shooter",
        fps,
    )
    timelines = (
        baseline_timelines[0],
        baseline_timelines[1],
        baseline_timelines[3],
        selected_timelines[1],
        selected_timelines[3],
        recovery_timeline,
        selected_timelines[6],
    )
    titles = (
        "S205 FAILURE-FEEDBACK FINISH REPAIR",
        "BEFORE · REJECTED CONTEXT",
        "BEFORE · 28.97 cm TARGET ERROR",
        "AFTER · TWO-STAGE ACTIVE SEARCH",
        "AFTER · 5.96 cm TARGET ERROR",
        "AFTER · STABLE POST-CONTACT RECOVERY",
        "CONTENT-BOUND LEARNED MEMORY",
    )
    kinds = (
        "VERIFIED_POSE_HOLD",
        "STRICT_FAILURE_REPLAY",
        "INTERPOLATED_FAILURE_REVIEW",
        "STRICT_LEARNED_REPLAY",
        "INTERPOLATED_PRECISION_REVIEW",
        "INTERPOLATED_STABILITY_REVIEW",
        "VERIFIED_FINAL_POSE_HOLD",
    )
    clips = tuple(
        ThreePlayerVideoClip(
            clip_id=f"{index:02d}-s205",
            title=title,
            frame_count=len(timeline),
            duration_sec=len(timeline) / fps,
            playback_kind=kind,
        )
        for index, (title, kind, timeline) in enumerate(
            zip(titles, kinds, timelines, strict=True), start=1
        )
    )
    metrics = cast(dict[str, Any], report["metrics"])
    labels = (
        "S205 · FAILURE → SEARCH → STABLE MUSCLE MEMORY",
        (
            "BEFORE · CONTEXTUAL EXPERT MISSED · "
            f"ERROR {float(metrics['source_failed_target_error_m']) * 100:.2f} cm"
        ),
        "FAILURE KEPT AS TRAINING SIGNAL · BACKEND REMAINED REJECTED",
        "66 PHYSICS CANDIDATES · COARSE SEARCH → STABILITY-CENTERED REFINEMENT",
        (
            f"AFTER · ERROR {float(metrics['selected_target_error_m']) * 100:.2f} cm / 10 cm · "
            f"IMPROVED {float(metrics['error_improvement_m']) * 100:.2f} cm"
        ),
        (
            f"NO FALL · MIN PELVIS {float(metrics['selected_min_pelvis_height_m']):.3f} m · "
            f"SUPPORT SLIP {float(metrics['selected_support_foot_slip_m']):.3f} m"
        ),
        "STRICT REPLAY PASS · ONE-CONTEXT REPAIR · NOT GENERALIZED · SIM ONLY",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco

        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-s205-video-") as temporary:
                label_paths = tuple(
                    Path(temporary) / f"label-{index}.txt" for index in range(len(labels))
                )
                for label_path, label in zip(label_paths, labels, strict=True):
                    label_path.write_text(label, encoding="utf-8")
                process = subprocess.Popen(
                    _ffmpeg_command(
                        ffmpeg=ffmpeg,
                        output=output,
                        fps=fps,
                        width=width,
                        height=height,
                        labels=label_paths,
                        clips=clips,
                        source_evidence_passed=True,
                        candidate_promoted=False,
                    ),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                if process.stdin is None:
                    raise RuntimeError("S205 video raw frame pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        renderer=renderer,
                        bundle=baseline_bundle,
                        timelines=timelines[:3],
                        stream=cast(BinaryIO, process.stdin),
                    )
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        renderer=renderer,
                        bundle=selected_bundle,
                        timelines=timelines[3:],
                        stream=cast(BinaryIO, process.stdin),
                    )
                except BaseException:
                    process.stdin.close()
                    process.kill()
                    process.wait()
                    raise
                process.stdin.close()
                stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
                code = process.wait()
                if code:
                    raise RuntimeError(f"S205 video encoding failed ({code}): {stderr[-3000:]}")
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    probe = _probe_video(ffprobe, output)
    expected_frames = sum(clip.frame_count for clip in clips)
    if (
        probe["width"] != width
        or probe["height"] != height
        or probe["fps"] != fps
        or probe["frame_count"] != expected_frames
    ):
        raise RuntimeError("S205 encoded video does not match its render contract")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.extended_finish_intent_repair_video.v1",
        "claim": _CLAIM,
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_report_path": str(report_path),
        "source_report_hash": report["report_hash"],
        "source_request_hash": hash_bytes(request_path.read_bytes()),
        "baseline_trajectory_hash": baseline_record["file_hash"],
        "selected_trajectory_hash": selected_record["file_hash"],
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": expected_frames,
        "duration_sec": probe["duration_sec"],
        "codec_name": probe["codec_name"],
        "clips": [asdict(clip) for clip in clips],
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "renderer_hash": _renderer_hash(),
    }
    manifest["manifest_hash"] = hash_json(manifest)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return validate_extended_finish_intent_repair_video_manifest(manifest_path)


def validate_extended_finish_intent_repair_video_manifest(path: Path) -> dict[str, Any]:
    """Validate media integrity while keeping pixels outside scoring authority."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S205 video manifest must be an object")
    expected = payload.pop("manifest_hash", None)
    try:
        video = Path(str(payload.get("video_path"))).expanduser().resolve()
        source = Path(str(payload.get("source_report_path"))).expanduser().resolve()
        report = validate_extended_finish_intent_repair(source)
        if (
            expected != hash_json(payload)
            or not video.is_file()
            or hash_bytes(video.read_bytes()) != payload.get("video_hash")
            or report.get("report_hash") != payload.get("source_report_hash")
            or payload.get("claim") != _CLAIM
            or payload.get("renderer_hash") != _renderer_hash()
            or payload.get("width") != 1920
            or payload.get("height") != 1080
            or payload.get("visualization_only") is not True
            or payload.get("pixels_used_for_scoring") is not False
            or payload.get("promotion_eligible") is not False
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("commercial_use_allowed") is not False
        ):
            raise ValueError("S205 video integrity or authority contract is invalid")
    finally:
        if expected is not None:
            payload["manifest_hash"] = expected
    return payload


def _bundle(
    *,
    physical_target: list[float],
    goal: Any,
    result: dict[str, Any],
    trajectory: dict[str, np.ndarray],
    passed: bool,
) -> Any:
    return SimpleNamespace(
        request={
            "physical_scoring_target_m": physical_target,
            "goal_spec": asdict(goal),
        },
        report={"result": result, "passed": passed},
        trajectory=trajectory,
    )


def _load_trajectory(root: Path, record: dict[str, Any]) -> dict[str, np.ndarray]:
    path = (root / str(record["file"])).resolve()
    if hash_bytes(path.read_bytes()) != record.get("file_hash"):
        raise ValueError("S205 video trajectory content changed")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _source_s204_path(request: dict[str, Any], report_path: Path) -> Path:
    expected = request.get("source_s204_file_hash")
    candidates = report_path.parents[1].glob(
        "s204-contextual-finish-portfolio-*/contextual-finish-portfolio.json"
    )
    for candidate in candidates:
        if hash_bytes(candidate.read_bytes()) == expected:
            return candidate
    raise ValueError("S205 video source S204 evidence is unavailable")


def _renderer_hash() -> str:
    return cast(
        str,
        hash_json(
            {
                "module": hash_bytes(Path(__file__).read_bytes()),
                "shared_renderer": hash_bytes(
                    (Path(__file__).parent / "three_player_video.py").read_bytes()
                ),
            }
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    result = render_extended_finish_intent_repair_video(
        evidence_path=args.evidence,
        asset_root=args.asset_root,
        output_path=args.output,
        fps=args.fps,
    )
    print(json.dumps({"video_hash": result["video_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "render_extended_finish_intent_repair_video",
    "validate_extended_finish_intent_repair_video_manifest",
]
