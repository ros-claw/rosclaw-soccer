"""Evidence-bound parent/candidate reel for the S105 recovery athlete."""

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

from rosclaw_soccer.media.three_role_save_portfolio_video import (
    _Clip,
    _ffmpeg_command,
    _Frame,
    _probe,
    _segment,
    _write_frames,
    _write_labels,
)
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.recovery_athlete_integration_exam import (
    validate_recovery_athlete_integration_exam,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_three_player_stadium_model

_CLAIM = "CONTEXT_GATED_NEURAL_RECOVERY_WITH_PER_LANE_PEAK_NONREGRESSION"


def _implementation_hash() -> str:
    return str(hash_bytes(Path(__file__).read_bytes()))


def validate_recovery_athlete_video_manifest(path: Path) -> dict[str, Any]:
    """Validate video bytes, source trajectories and visualization authority."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery athlete video manifest must be an object")
    expected = payload.pop("manifest_hash", None)
    if expected != hash_json(payload):
        raise ValueError("recovery athlete video manifest integrity mismatch")
    video_value = payload.get("video_path")
    source_files = payload.get("source_files")
    if not isinstance(video_value, str) or not isinstance(source_files, dict):
        raise ValueError("recovery athlete video bindings are invalid")
    video = Path(video_value).expanduser().resolve()
    if not video.is_file() or hash_bytes(video.read_bytes()) != payload.get("video_hash"):
        raise ValueError("recovery athlete video bytes changed")
    for source_value, source_hash in source_files.items():
        source = Path(source_value).expanduser().resolve()
        if not source.is_file() or hash_bytes(source.read_bytes()) != source_hash:
            raise ValueError("recovery athlete video source binding changed")
    numbers = tuple(
        payload.get(name) for name in ("fps", "width", "height", "frame_count", "duration_sec")
    )
    if (
        payload.get("schema_version") != "rosclaw_soccer.recovery_athlete_video.v2"
        or payload.get("claim") != _CLAIM
        or payload.get("evidence_passed") is not True
        or payload.get("strict_replay") is not True
        or payload.get("case_count") != 4
        or payload.get("visualization_only") is not True
        or payload.get("pixels_used_for_scoring") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
        or payload.get("commercial_use_allowed") is not False
        or payload.get("implementation_hash") != _implementation_hash()
        or any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in numbers
        )
    ):
        raise ValueError("recovery athlete video authority contract is invalid")
    payload["manifest_hash"] = expected
    return cast(dict[str, Any], payload)


def render_recovery_athlete_video(
    *,
    evidence_path: Path,
    goal_contract_path: Path,
    asset_root: Path,
    output_path: Path,
    fps: int = 60,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render matched S104 parent and S105 neural recovery trajectories."""

    evidence_file = evidence_path.expanduser().resolve()
    goal_contract_file = goal_contract_path.expanduser().resolve()
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
        raise ValueError("recovery athlete video output contract is invalid")
    evidence = validate_recovery_athlete_integration_exam(evidence_file)
    request_path = evidence_file.parent / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    goal_contract = json.loads(goal_contract_file.read_text(encoding="utf-8"))
    goal_specs = goal_contract.get("lane_goal_specs")
    cases = cast(dict[str, dict[str, Any]], evidence["cases"])
    if (
        not isinstance(goal_specs, dict)
        or set(goal_specs) != set(cases)
        or goal_contract.get("body_hash") != request.get("body_hash")
    ):
        raise ValueError("recovery athlete video goal contract changed")
    trajectories: dict[str, dict[str, np.ndarray]] = {}
    goals: dict[str, G1TrainingGoalSpec] = {}
    source_files = {
        str(evidence_file): hash_bytes(evidence_file.read_bytes()),
        str(request_path): hash_bytes(request_path.read_bytes()),
        str(goal_contract_file): hash_bytes(goal_contract_file.read_bytes()),
    }
    required = {
        "time",
        "ball_pose",
        "passer_pelvis_pose",
        "passer_joint_position",
        "shooter_pelvis_pose",
        "shooter_joint_position",
        "goalkeeper_pelvis_pose",
        "goalkeeper_joint_position",
        "goalkeeper_foot_contact",
    }
    for lane_id, case in cases.items():
        goal = G1TrainingGoalSpec(**goal_specs[lane_id])
        if abs(goal.width_m - 7.32) > 1.0e-9 or abs(goal.height_m - 2.44) > 1.0e-9:
            raise ValueError("recovery athlete video requires a regulation goal")
        goals[lane_id] = goal
        for route in ("parent", "candidate"):
            name = case.get(f"{route}_trajectory_file")
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("recovery athlete video trajectory name is invalid")
            trajectory_path = evidence_file.parent / name
            with np.load(trajectory_path, allow_pickle=False) as archive:
                trajectory = {key: np.asarray(archive[key]) for key in archive.files}
            if not required <= set(trajectory):
                raise ValueError("recovery athlete video trajectory is incomplete")
            trajectories[f"{lane_id}:{route}"] = trajectory
            source_files[str(trajectory_path)] = hash_bytes(trajectory_path.read_bytes())
    portfolio_metrics = cast(dict[str, Any], evidence["portfolio_metrics"])
    clips = _timeline(cases, trajectories, portfolio_metrics, fps)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for recovery athlete video")
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        if qualification.body_hash != request.get("body_hash"):
            raise ValueError("recovery athlete video Body hash changed")
        import mujoco

        first_lane = next(iter(cases))
        goal = goals[first_lane]
        model = build_g1_three_player_stadium_model(
            asset_root.expanduser().resolve(),
            passer_origin_m=(5.10, -0.16406006503921598, 0.0),
            goalkeeper_origin_m=(goal.plane_x_m - 0.48, 0.0, 0.0),
            spec=goal,
        )
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-recovery-athlete-") as temp:
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
                    raise RuntimeError("recovery athlete raw-video pipe is unavailable")
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
                    raise RuntimeError(f"recovery athlete ffmpeg failed: {stderr[-3000:]}")
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl
    video_probe = _probe(ffprobe, output)
    frame_count = sum(len(clip.frames) for clip in clips)
    if (
        video_probe["width"] != width
        or video_probe["height"] != height
        or video_probe["fps"] != fps
        or abs(video_probe["frame_count"] - frame_count) > 1
    ):
        raise RuntimeError("recovery athlete encoded video contract changed")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_athlete_video.v2",
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_files": source_files,
        "claim": _CLAIM,
        "evidence_report_hash": evidence["report_hash"],
        "evidence_passed": True,
        "strict_replay": all(row["strict_replay"] for row in cases.values()),
        "case_count": len(cases),
        "portfolio_metrics": portfolio_metrics,
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
    validate_recovery_athlete_video_manifest(manifest_path)
    return manifest


def _timeline(
    cases: dict[str, dict[str, Any]],
    trajectories: dict[str, dict[str, np.ndarray]],
    portfolio_metrics: dict[str, Any],
    fps: int,
) -> tuple[_Clip, ...]:
    clips: list[_Clip] = []
    first = f"{next(iter(cases))}:candidate"
    start = float(trajectories[first]["time"][0])
    clips.append(
        _Clip(
            "S106 CONTEXT-GATED RECOVERY · NEURAL ACTOR + MONOTONE AUTHORITY ENVELOPE",
            tuple(_Frame(first, start, "wide_goal") for _ in range(round(1.8 * fps))),
        )
    )
    for index, (lane_id, case) in enumerate(cases.items(), start=1):
        candidate = cast(dict[str, Any], case["candidate"])
        parent = cast(dict[str, Any], case["parent"])
        result = cast(dict[str, Any], candidate["result"])
        successor = cast(dict[str, Any], candidate["successor"])
        probe = cast(dict[str, Any], successor["probe"])["metrics"]
        parent_metrics = cast(dict[str, Any], parent["command_metrics"])
        candidate_metrics = cast(dict[str, Any], candidate["command_metrics"])
        shot = float(result["shot_contact_time_sec"])
        save = float(result["goalkeeper_glove_contact_time_sec"])
        probe_start = float(probe["start_sec"])
        probe_stop = float(probe["stop_sec"])
        end = float(trajectories[f"{lane_id}:candidate"]["time"][-1])
        parent_tv = float(parent_metrics["lateral_command_total_variation_mps"])
        candidate_tv = float(candidate_metrics["lateral_command_total_variation_mps"])
        parent_peak = float(parent_metrics["lateral_command_peak_step_mps"])
        candidate_peak = float(candidate_metrics["lateral_command_peak_step_mps"])
        actor_fraction = float(candidate_metrics["recovery_actor_active_fraction"])
        suppressed_fraction = float(
            cast(dict[str, Any], candidate["result"])[
                "goalkeeper_recovery_athlete_suppressed_fraction"
            ]
        )
        latency = float(candidate["ready_latency_sec"])
        clips.extend(
            (
                _Clip(
                    f"LANE {index}/4 · PASS → LIVE HIGH SHOT → AIRBORNE GLOVE SAVE",
                    _segment(f"{lane_id}:candidate", shot - 0.45, save + 0.32, 0.82, "chain", fps),
                ),
                _Clip(
                    f"S104 PARENT RECOVERY · COMMAND TV {parent_tv:.3f} m/s",
                    _segment(
                        f"{lane_id}:parent",
                        save + 0.35,
                        probe_start - 0.20,
                        2.25,
                        "keeper_front",
                        fps,
                    ),
                ),
                _Clip(
                    f"S106 GATED NEURAL RECOVERY · TV {candidate_tv:.3f} · "
                    f"PEAK {parent_peak:.3f}→{candidate_peak:.3f} m/s",
                    _segment(
                        f"{lane_id}:candidate",
                        save + 0.35,
                        probe_start - 0.20,
                        2.25,
                        "keeper_front",
                        fps,
                    ),
                ),
                _Clip(
                    f"ENVELOPE ADJUSTED {suppressed_fraction:.1%} FRAMES · "
                    f"ACTOR {actor_fraction:.1%} · SUCCESSOR "
                    f"{float(probe['signed_displacement_m']):+.3f} m",
                    _segment(
                        f"{lane_id}:candidate",
                        probe_start - 0.25,
                        probe_stop + 0.55,
                        0.72,
                        "keeper_hero",
                        fps,
                    ),
                ),
                _Clip(
                    f"READY AGAIN IN {latency:.2f}s · DOUBLE SUPPORT · NO RESET / NO TELEPORT",
                    _segment(f"{lane_id}:candidate", end - 1.0, end, 1.0, "keeper_front", fps),
                ),
            )
        )
    last = f"{next(reversed(cases))}:candidate"
    end = float(trajectories[last]["time"][-1])
    reduction = 100.0 * (1.0 - float(portfolio_metrics["candidate_to_parent_variation_ratio"]))
    clips.append(
        _Clip(
            f"4/4 · STRICT REPLAY · {reduction:.1f}% LESS COMMAND VARIATION · "
            "EVERY LANE PEAK-NONINFERIOR",
            tuple(_Frame(last, end, "wide_goal") for _ in range(round(2.2 * fps))),
        )
    )
    return tuple(clips)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--goal-contract", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = render_recovery_athlete_video(
        evidence_path=args.evidence,
        goal_contract_path=args.goal_contract,
        asset_root=args.asset_root,
        output_path=args.output,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["render_recovery_athlete_video", "validate_recovery_athlete_video_manifest"]
