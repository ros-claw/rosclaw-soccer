"""Render S206 multi-context finish growth from strict physics trajectories."""

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
    _timelines,
    _write_frames,
)
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.development_evidence import (
    three_role_development_kwargs,
    three_role_goal_spec,
)
from rosclaw_soccer.training.contextual_finish_intent_portfolio import (
    validate_contextual_finish_intent_portfolio,
)
from rosclaw_soccer.world.field import build_g1_three_player_stadium_model

_CLAIM = "S206_MULTI_CONTEXT_4D_FINISH_INTENT_VISUALIZATION"


def render_contextual_finish_intent_video(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    fps: int = 30,
) -> dict[str, Any]:
    """Render four bound trajectories only after S206 fully validates."""

    report_path = evidence_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    manifest_path = output.with_suffix(".json")
    if output.suffix.lower() != ".mp4" or output.exists() or manifest_path.exists():
        raise ValueError("S206 video output must be a new MP4 path")
    if not 10 <= fps <= 60:
        raise ValueError("S206 video fps must be in [10, 60]")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for S206 video")
    report = validate_contextual_finish_intent_portfolio(report_path)
    if report.get("status") != "PASS_CONTEXTUAL_FINISH_INTENT_PORTFOLIO":
        raise ValueError("S206 video requires passing portfolio evidence")
    request_path = report_path.parent / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if qualification.body_hash != request["body_hash"]:
        raise ValueError("S206 video body does not match physical evidence")
    bundles = _bundles(report_path.parent, report, request)
    goal = three_role_goal_spec()
    development = three_role_development_kwargs()
    keeper = development["goalkeeper_config"]
    repair_context = cast(dict[str, Any], request["repair_source_context"])
    model = build_g1_three_player_stadium_model(
        asset_root.expanduser().resolve(),
        passer_origin_m=tuple(development["passer_origin"]),
        passer_yaw_rad=float(repair_context["passer_yaw_rad"]),
        goalkeeper_origin_m=(
            goal.plane_x_m - keeper.depth_from_goal_line_m,
            keeper.initial_lateral_position_m,
            0.0,
        ),
        spec=goal,
    )
    width, height = 1920, 1080
    _configure_offscreen_framebuffer(model, width=width, height=height)
    source_timelines, _ = _timelines(bundles[0], fps)
    repaired_timelines, _ = _timelines(bundles[1], fps)
    holdout_a_timelines, _ = _timelines(bundles[2], fps)
    holdout_b_timelines, _ = _timelines(bundles[3], fps)
    grouped_timelines = (
        source_timelines[:3],
        (repaired_timelines[1], repaired_timelines[3], repaired_timelines[4]),
        (holdout_a_timelines[1], holdout_a_timelines[3]),
        (
            holdout_b_timelines[1],
            holdout_b_timelines[3],
            holdout_b_timelines[4],
            holdout_b_timelines[6],
        ),
    )
    timelines = tuple(timeline for group in grouped_timelines for timeline in group)
    metrics = cast(dict[str, Any], report["metrics"])
    holdouts = cast(dict[str, Any], report["holdouts"])
    titles = (
        "S206 · MULTI-CONTEXT FINISH GROWTH",
        "BEFORE · B-BASIN TARGET ERROR",
        "BEFORE · STRICT FAILURE REVIEW",
        "AFTER · 4-D ACTIVE REPAIR",
        "AFTER · CENTIMETRE-LEVEL FINISH",
        "AFTER · STABLE RECOVERY",
        "FRESH A-BASIN HOLDOUT",
        "A-BASIN · STRICT TARGET REVIEW",
        "FRESH B-BASIN HOLDOUT",
        "B-BASIN · STRICT TARGET REVIEW",
        "B-BASIN · STABLE RECOVERY",
        "8 EXPERTS · OOD FAIL-CLOSED",
    )
    kinds = (
        "VERIFIED_POSE_HOLD",
        "STRICT_SOURCE_REPLAY",
        "INTERPOLATED_FAILURE_REVIEW",
        "STRICT_REPAIR_REPLAY",
        "INTERPOLATED_PRECISION_REVIEW",
        "INTERPOLATED_STABILITY_REVIEW",
        "STRICT_FRESH_HOLDOUT_REPLAY",
        "INTERPOLATED_PRECISION_REVIEW",
        "STRICT_FRESH_HOLDOUT_REPLAY",
        "INTERPOLATED_PRECISION_REVIEW",
        "INTERPOLATED_STABILITY_REVIEW",
        "VERIFIED_FINAL_POSE_HOLD",
    )
    clips = tuple(
        ThreePlayerVideoClip(
            clip_id=f"{index:02d}-s206",
            title=title,
            frame_count=len(timeline),
            duration_sec=len(timeline) / fps,
            playback_kind=kind,
        )
        for index, (title, kind, timeline) in enumerate(
            zip(titles, kinds, timelines, strict=True), start=1
        )
    )
    labels = (
        "S206 · FAILURE MEMORY → TWO LOCAL EXPERT BASINS → FRESH HOLDOUTS",
        (f"BEFORE · B-BASIN ERROR {float(metrics['repair_source_target_error_m']) * 100:.2f} cm"),
        "FAILURE RETAINED · NO GATE RELAXATION · CPU MUJOCO TRUTH",
        (
            f"98 PHYSICS CANDIDATES · {int(metrics['repair_eligible_candidate_count'])} "
            "PRECISE + STABLE"
        ),
        (
            "AFTER · B-BASIN ERROR "
            f"{float(metrics['repair_selected_target_error_m']) * 100:.2f} cm · "
            f"GAIN {float(metrics['repair_error_improvement_m']) * 100:.2f} cm"
        ),
        "NO FALL · PASS DELIVERY RETAINED · STRICT REPLAY",
        (
            "FRESH A HOLDOUT · ERROR "
            f"{float(holdouts['s206-a-holdout']['result']['target_error_m']) * 100:.2f} cm"
        ),
        "NEAREST VERIFIED 4-D EXPERT · NO INTERPOLATION ACROSS CONTACT CLIFFS",
        (
            "FRESH B HOLDOUT · ERROR "
            f"{float(holdouts['s206-b-holdout']['result']['target_error_m']) * 100:.2f} cm"
        ),
        "SECOND PHYSICAL BASIN · STRICT INDEPENDENT REPLAY",
        "POST-CONTACT SUPPORT RETAINED · HIGH-LEVEL INTENT ONLY",
        "2/2 FRESH SUCCESS · 2/2 OOD REJECT · SIM-ONLY BACKEND ACCEPTED",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco

        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-s206-video-") as temporary:
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
                    raise RuntimeError("S206 video raw frame pipe is unavailable")
                try:
                    for bundle, group in zip(bundles, grouped_timelines, strict=True):
                        _write_frames(
                            mujoco=mujoco,
                            model=model,
                            data=data,
                            renderer=renderer,
                            bundle=bundle,
                            timelines=group,
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
                    raise RuntimeError(f"S206 video encoding failed ({code}): {stderr[-3000:]}")
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
        raise RuntimeError("S206 encoded video does not match its render contract")
    trajectory_records = _trajectory_records(report)
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.contextual_finish_intent_video.v1",
        "claim": _CLAIM,
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_report_path": str(report_path),
        "source_report_hash": report["report_hash"],
        "source_request_hash": hash_bytes(request_path.read_bytes()),
        "trajectory_file_hashes": [record["file_hash"] for record in trajectory_records],
        "trajectory_digests": [record["trajectory_digest"] for record in trajectory_records],
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
    return validate_contextual_finish_intent_video_manifest(manifest_path)


def validate_contextual_finish_intent_video_manifest(path: Path) -> dict[str, Any]:
    """Validate source and media bytes without granting pixels scoring authority."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S206 video manifest must be an object")
    expected = payload.pop("manifest_hash", None)
    try:
        video = Path(str(payload.get("video_path"))).expanduser().resolve()
        source = Path(str(payload.get("source_report_path"))).expanduser().resolve()
        report = validate_contextual_finish_intent_portfolio(source)
        records = _trajectory_records(report)
        if (
            expected != hash_json(payload)
            or not video.is_file()
            or hash_bytes(video.read_bytes()) != payload.get("video_hash")
            or report.get("report_hash") != payload.get("source_report_hash")
            or payload.get("claim") != _CLAIM
            or payload.get("renderer_hash") != _renderer_hash()
            or payload.get("trajectory_file_hashes") != [record["file_hash"] for record in records]
            or payload.get("trajectory_digests")
            != [record["trajectory_digest"] for record in records]
            or payload.get("width") != 1920
            or payload.get("height") != 1080
            or payload.get("visualization_only") is not True
            or payload.get("pixels_used_for_scoring") is not False
            or payload.get("promotion_eligible") is not False
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("commercial_use_allowed") is not False
        ):
            raise ValueError("S206 video integrity or authority contract is invalid")
    finally:
        if expected is not None:
            payload["manifest_hash"] = expected
    return payload


def _bundles(root: Path, report: dict[str, Any], request: dict[str, Any]) -> tuple[Any, ...]:
    records = _trajectory_records(report)
    results = (
        report["repair_baseline"]["result"],
        report["repair"]["selected_replay"]["result"],
        report["holdouts"]["s206-a-holdout"]["result"],
        report["holdouts"]["s206-b-holdout"]["result"],
    )
    goal = three_role_goal_spec()
    return tuple(
        SimpleNamespace(
            request={
                "physical_scoring_target_m": request["physical_target_m"],
                "goal_spec": asdict(goal),
            },
            report={"result": result, "passed": index > 0},
            trajectory=_load_trajectory(root, record),
        )
        for index, (record, result) in enumerate(zip(records, results, strict=True))
    )


def _trajectory_records(report: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return (
        cast(dict[str, Any], report["repair_baseline"]["trajectory"]),
        cast(dict[str, Any], report["repair"]["selected_replay"]["trajectory"]),
        cast(dict[str, Any], report["holdouts"]["s206-a-holdout"]["trajectory"]),
        cast(dict[str, Any], report["holdouts"]["s206-b-holdout"]["trajectory"]),
    )


def _load_trajectory(root: Path, record: dict[str, Any]) -> dict[str, np.ndarray]:
    path = (root / str(record["file"])).resolve()
    if hash_bytes(path.read_bytes()) != record.get("file_hash"):
        raise ValueError("S206 video trajectory content changed")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _renderer_hash() -> str:
    return str(
        hash_json(
            {
                "module": hash_bytes(Path(__file__).read_bytes()),
                "shared_renderer": hash_bytes(
                    (Path(__file__).parent / "three_player_video.py").read_bytes()
                ),
            }
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    result = render_contextual_finish_intent_video(
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
    "render_contextual_finish_intent_video",
    "validate_contextual_finish_intent_video_manifest",
]
