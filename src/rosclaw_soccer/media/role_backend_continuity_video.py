"""Render the S202 learned-pass continuity and causal handoff evidence."""

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
from rosclaw_soccer.training.role_backend_continuity_evidence import (
    validate_role_backend_continuity_evidence,
)
from rosclaw_soccer.world.field import build_g1_three_player_stadium_model

_CLAIM = "S202_ROLE_QUALIFIED_PASS_AND_CAUSAL_HANDOFF_VISUALIZATION"


def render_role_backend_continuity_video(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    fps: int = 30,
) -> dict[str, Any]:
    """Render an evidence-downstream 1080p review; pixels never score policy."""

    report_path = evidence_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    manifest_path = output.with_suffix(".json")
    if output.suffix.lower() != ".mp4" or output.exists() or manifest_path.exists():
        raise ValueError("S202 video output must be a new MP4 path")
    if not 10 <= fps <= 60:
        raise ValueError("S202 video fps must be in [10, 60]")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for S202 video")
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    report = validate_role_backend_continuity_evidence(report_path)
    trajectory_record = cast(dict[str, Any], report["trajectory"])
    trajectory_path = (report_path.parent / str(trajectory_record["file"])).resolve()
    with np.load(trajectory_path, allow_pickle=False) as archive:
        trajectory = {name: archive[name] for name in archive.files}
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if qualification.body_hash != report["request"]["body_hash"]:
        raise ValueError("S202 video body does not match its physical evidence")

    goal = three_role_goal_spec()
    development = three_role_development_kwargs()
    keeper = development["goalkeeper_config"]
    model = build_g1_three_player_stadium_model(
        asset_root.expanduser().resolve(),
        passer_origin_m=tuple(development["passer_origin"]),
        passer_yaw_rad=float(report["request"]["executed_passer_yaw_rad"]),
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
            "physical_scoring_target_m": report["chain_request"]["goal_target_m"],
            "goal_spec": asdict(goal),
        },
        report={"result": report["result"], "passed": True},
        trajectory=trajectory,
    )
    timelines, clips = _timelines(compatibility_bundle, fps)
    labels = (
        "LEARNED PLAYMAKER BACKEND · ROLE-QUALIFIED ROUTE",
        "ONE BALL · PASS → ONE-TOUCH RECEIVE → SHOT",
        (
            f"PASS ERROR {float(report['result']['pass_delivery_error_m']) * 100:.2f} cm · "
            f"SPEED {float(report['result']['pass_peak_ball_speed_mps']):.2f} m/s"
        ),
        (
            f"NEXT FAILURE: SHOT ERROR {float(report['result']['target_error_m']):.3f} m · "
            "GOALKEEPER CREDIT BLOCKED"
        ),
        "RED.FINISHER → PLASTIC · DOWNSTREAM LEARNING WAITS",
        "RED.PLAYMAKER + OTHER 4 CELLS → FROZEN",
        "STRICT REPLAY · SAFE TEAM · SIM ONLY · VIDEO DOES NOT SCORE",
    )
    if len(labels) != len(clips):
        raise RuntimeError("S202 video labels no longer match the review timeline")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import mujoco

        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-s202-video-") as temporary:
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
                    raise RuntimeError("S202 video raw frame pipe is unavailable")
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
                    raise RuntimeError(f"S202 video encoding failed ({code}): {stderr[-3000:]}")
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
        raise RuntimeError("S202 encoded video does not match its render contract")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.role_backend_continuity_video.v1",
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
    return validate_role_backend_continuity_video_manifest(manifest_path)


def validate_role_backend_continuity_video_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S202 video manifest must be an object")
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
            raise ValueError("S202 video integrity or authority contract is invalid")
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
    result = render_role_backend_continuity_video(
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
    "render_role_backend_continuity_video",
    "validate_role_backend_continuity_video_manifest",
]
