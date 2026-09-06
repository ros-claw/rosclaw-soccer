"""Render the S203 precise, safe finisher seed from its strict replay."""

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
from rosclaw_soccer.training.contextual_finish_target_growth import (
    validate_contextual_finish_target_growth,
)
from rosclaw_soccer.world.field import build_g1_three_player_stadium_model

_CLAIM = "S203_PRECISE_SAFE_CONTEXTUAL_FINISH_VISUALIZATION"


def render_contextual_finish_target_video(
    *, evidence_path: Path, asset_root: Path, output_path: Path, fps: int = 30
) -> dict[str, Any]:
    """Render only after the machine-readable S203 gate validates."""

    report_path = evidence_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    manifest_path = output.with_suffix(".json")
    if output.suffix.lower() != ".mp4" or output.exists() or manifest_path.exists():
        raise ValueError("S203 video output must be a new MP4 path")
    if not 10 <= fps <= 60:
        raise ValueError("S203 video fps must be in [10, 60]")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for S203 video")
    report = validate_contextual_finish_target_growth(report_path)
    if report.get("status") != "PASS_SINGLE_CONTEXT_FINISH_TARGET_SEED":
        raise ValueError("S203 video requires passing development evidence")
    replay = cast(dict[str, Any], report["selected_replay"])
    trajectory_record = cast(dict[str, Any], replay["trajectory"])
    trajectory_path = (report_path.parent / str(trajectory_record["file"])).resolve()
    with np.load(trajectory_path, allow_pickle=False) as archive:
        trajectory = {name: archive[name] for name in archive.files}
    request = json.loads((report_path.parent / "request.json").read_text(encoding="utf-8"))
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if qualification.body_hash != request["body_hash"]:
        raise ValueError("S203 video body does not match physical evidence")

    goal = three_role_goal_spec()
    development = three_role_development_kwargs()
    keeper = development["goalkeeper_config"]
    model = build_g1_three_player_stadium_model(
        asset_root.expanduser().resolve(),
        passer_origin_m=tuple(development["passer_origin"]),
        passer_yaw_rad=float(request["passer_yaw_rad"]),
        goalkeeper_origin_m=(
            goal.plane_x_m - keeper.depth_from_goal_line_m,
            keeper.initial_lateral_position_m,
            0.0,
        ),
        spec=goal,
    )
    width, height = 1920, 1080
    _configure_offscreen_framebuffer(model, width=width, height=height)
    compatibility_bundle: Any = SimpleNamespace(
        request={
            "physical_scoring_target_m": request["physical_target_m"],
            "goal_spec": asdict(goal),
        },
        report={"result": replay["result"], "passed": True},
        trajectory=trajectory,
    )
    timelines, clips = _timelines(compatibility_bundle, fps)
    result = cast(dict[str, Any], replay["result"])
    metrics = cast(dict[str, Any], report["metrics"])
    labels = (
        "S203 FINISHER GROWTH · 32-CANDIDATE CLOSED LOOP",
        "ONE BALL · LEARNED PASS → CONTINUOUS RECEIVE → PRECISE SHOT",
        f"ROLLING PASS ERROR {float(result['pass_delivery_error_m']) * 100:.2f} cm",
        (
            f"SHOT {float(result['shot_peak_ball_speed_mps']):.2f} m/s · "
            f"TARGET ERROR {float(result['target_error_m']) * 100:.2f} cm / 10 cm"
        ),
        (
            f"NO FALL · MIN PELVIS {float(result['shooter_min_pelvis_height_m']):.3f} m · "
            f"POST-CONTACT SLIP {float(result['shooter_post_contact_support_foot_slip_m']):.3f} m"
        ),
        (
            "PREDICTIVE JOINT ENVELOPE · ZERO LIMIT BREACH · "
            f"ACTIVE {float(metrics['selected_joint_guard_fraction']) * 100:.1f}%"
        ),
        "STRICT REPLAY · SINGLE-CONTEXT SEED · NOT PROMOTED · SIM ONLY",
    )
    if len(labels) != len(clips):
        raise RuntimeError("S203 video labels no longer match the review timeline")
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco

        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-s203-video-") as temporary:
                label_paths = tuple(Path(temporary) / f"label-{index}.txt" for index in range(7))
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
                    raise RuntimeError("S203 video raw frame pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        renderer=renderer,
                        bundle=compatibility_bundle,
                        timelines=timelines,
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
                    raise RuntimeError(f"S203 video encoding failed ({code}): {stderr[-3000:]}")
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
        raise RuntimeError("S203 encoded video does not match its render contract")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.contextual_finish_target_video.v1",
        "claim": _CLAIM,
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_report_hash": report["report_hash"],
        "source_trajectory_hash": trajectory_record["file_hash"],
        "source_trajectory_digest": trajectory_record["trajectory_digest"],
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
        "renderer_hash": hash_json(
            {
                "module": hash_bytes(Path(__file__).read_bytes()),
                "shared_renderer": hash_bytes(
                    (Path(__file__).parent / "three_player_video.py").read_bytes()
                ),
            }
        ),
    }
    manifest["manifest_hash"] = hash_json(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return validate_contextual_finish_target_video_manifest(manifest_path)


def validate_contextual_finish_target_video_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S203 video manifest must be an object")
    expected = payload.pop("manifest_hash", None)
    try:
        video = Path(str(payload.get("video_path"))).expanduser().resolve()
        if (
            expected != hash_json(payload)
            or not video.is_file()
            or hash_bytes(video.read_bytes()) != payload.get("video_hash")
            or payload.get("claim") != _CLAIM
            or payload.get("width") != 1920
            or payload.get("height") != 1080
            or payload.get("visualization_only") is not True
            or payload.get("pixels_used_for_scoring") is not False
            or payload.get("promotion_eligible") is not False
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("commercial_use_allowed") is not False
        ):
            raise ValueError("S203 video integrity or authority contract is invalid")
    finally:
        if expected is not None:
            payload["manifest_hash"] = expected
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    result = render_contextual_finish_target_video(
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
    "render_contextual_finish_target_video",
    "validate_contextual_finish_target_video_manifest",
]
