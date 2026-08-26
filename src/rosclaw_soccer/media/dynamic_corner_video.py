"""Evidence-bound four-lane reel for the strict dynamic corner portfolio."""

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
from rosclaw_soccer.training.dynamic_corner_save import (
    validate_dynamic_corner_evidence,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_three_player_stadium_model

_CLAIM = "STRICT_MULTI_CORNER_AIRBORNE_SAVE_PORTFOLIO"


def validate_dynamic_corner_video_manifest(path: Path) -> dict[str, Any]:
    """Verify the video, every trajectory binding and the authority ceiling."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dynamic corner video manifest must be an object")
    expected = payload.pop("manifest_hash", None)
    if expected != hash_json(payload):
        raise ValueError("dynamic corner video manifest integrity mismatch")
    video_value = payload.get("video_path")
    source_files = payload.get("source_files")
    if not isinstance(video_value, str) or not isinstance(source_files, dict):
        raise ValueError("dynamic corner video bindings are invalid")
    video = Path(video_value).expanduser().resolve()
    if not video.is_file() or hash_bytes(video.read_bytes()) != payload.get("video_hash"):
        raise ValueError("dynamic corner video hash changed")
    for source_value, source_hash in source_files.items():
        source = Path(source_value).expanduser().resolve()
        if not source.is_file() or hash_bytes(source.read_bytes()) != source_hash:
            raise ValueError("dynamic corner video source binding changed")
    numbers = tuple(
        payload.get(name) for name in ("fps", "width", "height", "frame_count", "duration_sec")
    )
    if (
        payload.get("schema_version") != "rosclaw_soccer.dynamic_corner_video.v1"
        or payload.get("claim") != _CLAIM
        or payload.get("case_count") != 4
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
        raise ValueError("dynamic corner video authority contract is invalid")
    payload["manifest_hash"] = expected
    return cast(dict[str, Any], payload)


def render_dynamic_corner_video(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 60,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render all four frozen lanes; no image observation enters a score."""

    evidence_file = evidence_path.expanduser().resolve()
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
        raise ValueError("dynamic corner video output contract is invalid")
    evidence = validate_dynamic_corner_evidence(evidence_file)
    cases_value = cast(dict[str, dict[str, Any]], evidence["cases"])
    request_path = evidence_file.parent / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    goal_specs = request.get("lane_goal_specs")
    config_value = request.get("config")
    if not isinstance(goal_specs, dict) or set(goal_specs) != set(cases_value):
        raise ValueError("dynamic corner goal contracts are incomplete")
    trajectories: dict[str, dict[str, np.ndarray]] = {}
    goals: dict[str, G1TrainingGoalSpec] = {}
    source_files = {
        str(evidence_file): hash_bytes(evidence_file.read_bytes()),
        str(request_path): hash_bytes(request_path.read_bytes()),
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
    for lane_id, case in cases_value.items():
        name = case.get("trajectory_file")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("dynamic corner trajectory name is invalid")
        trajectory_path = evidence_file.parent / name
        with np.load(trajectory_path, allow_pickle=False) as archive:
            trajectory = {key: np.asarray(archive[key]) for key in archive.files}
        if not required <= set(trajectory):
            raise ValueError("dynamic corner trajectory is incomplete")
        goal = G1TrainingGoalSpec(**goal_specs[lane_id])
        if abs(goal.width_m - 7.32) > 1.0e-9 or abs(goal.height_m - 2.44) > 1.0e-9:
            raise ValueError("dynamic corner video requires a regulation goal")
        trajectories[lane_id] = trajectory
        goals[lane_id] = goal
        source_files[str(trajectory_path)] = hash_bytes(trajectory_path.read_bytes())
    clips = _timeline(cases_value, trajectories, evidence, fps)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for dynamic corner video")
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        if qualification.body_hash != request.get("body_hash"):
            raise ValueError("dynamic corner Body hash changed")
        lanes = config_value.get("lanes") if isinstance(config_value, dict) else None
        if not isinstance(lanes, list) or not lanes:
            raise ValueError("dynamic corner video config is incomplete")
        first_takeoff = lanes[0].get("takeoff_config")
        first_lunge = first_takeoff.get("lunge_config") if isinstance(first_takeoff, dict) else None
        first_aerial = first_lunge.get("aerial_config") if isinstance(first_lunge, dict) else None
        if not isinstance(first_aerial, dict):
            raise ValueError("dynamic corner aerial config is incomplete")
        import mujoco

        first_lane = next(iter(cases_value))
        goal = goals[first_lane]
        depth = float(first_aerial["goalkeeper_depth_from_goal_line_m"])
        model = build_g1_three_player_stadium_model(
            asset_root.expanduser().resolve(),
            passer_origin_m=(5.10, -0.16406006503921598, 0.0),
            goalkeeper_origin_m=(goal.plane_x_m - depth, 0.0, 0.0),
            spec=goal,
        )
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-dynamic-corner-") as temp:
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
                    raise RuntimeError("dynamic corner raw-video pipe is unavailable")
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
                    raise RuntimeError(f"dynamic corner ffmpeg failed: {stderr[-3000:]}")
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
        raise RuntimeError("dynamic corner encoded video contract changed")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.dynamic_corner_video.v1",
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_files": source_files,
        "claim": _CLAIM,
        "strict_replay": True,
        "case_count": len(cases_value),
        "contact_span_m": evidence["contact_span_m"],
        "contact_positions_y_m": evidence["contact_positions_y_m"],
        "dive_athlete": evidence.get("dive_athlete"),
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
    }
    manifest["manifest_hash"] = hash_json(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    validate_dynamic_corner_video_manifest(manifest_path)
    return manifest


def _timeline(
    cases: dict[str, dict[str, Any]],
    trajectories: dict[str, dict[str, np.ndarray]],
    evidence: dict[str, Any],
    fps: int,
) -> tuple[_Clip, ...]:
    clips: list[_Clip] = []
    first_lane = next(iter(cases))
    start = float(trajectories[first_lane]["time"][0])
    athlete = evidence.get("dive_athlete")
    authority_by_lane = athlete.get("authority_by_lane") if isinstance(athlete, dict) else None
    title = (
        "S103 TARGET-CONDITIONED NEURAL DIVE · FOUR STRICT HIGH-CORNER SAVES"
        if isinstance(authority_by_lane, dict)
        else "FOUR HIGH CORNERS · TWO GLOVES · FOUR STRICT AIRBORNE SAVES"
    )
    clips.append(
        _Clip(
            title,
            tuple(_Frame(first_lane, start, "wide") for _ in range(round(1.5 * fps))),
        )
    )
    for index, (lane_id, case) in enumerate(cases.items(), start=1):
        replay = cast(dict[str, Any], case["replay"])
        metrics = cast(dict[str, Any], replay["metrics"])
        base = cast(dict[str, Any], replay["base"])
        result = cast(dict[str, Any], case["result"])
        lane = cast(dict[str, Any], case["lane"])
        pass_time = float(result["pass_contact_time_sec"])
        shot_time = float(result["shot_contact_time_sec"])
        save_time = float(result["goalkeeper_glove_contact_time_sec"])
        landing_time = float(metrics["landing_time_sec"])
        contact = cast(list[float], result["goalkeeper_glove_contact_position_m"])
        flight_ms = 1_000.0 * float(metrics["airborne_duration_sec"])
        rise_mm = 1_000.0 * float(metrics["flight_pelvis_rise_m"])
        incoming = float(base["incoming_speed_mps"])
        authority_label = ""
        if isinstance(authority_by_lane, dict):
            authority = authority_by_lane.get(lane_id)
            if isinstance(authority, int | float) and not isinstance(authority, bool):
                authority_label = f" · NEURAL RESIDUAL {100.0 * float(authority):.0f}%"
        clips.append(
            _Clip(
                f"SAVE {index}/4 · {lane['label']}{authority_label} · "
                "CONTINUOUS PASS → STRIKE → SAVE",
                (
                    *_segment(lane_id, pass_time - 0.55, shot_time + 0.15, 1.0, "chain", fps),
                    *_segment(lane_id, shot_time + 0.15, save_time + 0.55, 0.80, "hero", fps),
                ),
            )
        )
        clips.append(
            _Clip(
                f"{result['goalkeeper_glove_contact_side'].upper()} GLOVE · "
                f"y={contact[1]:+.3f} m · {incoming:.2f} m/s · NO GOAL",
                _segment(
                    lane_id,
                    float(metrics["airborne_start_sec"]) - 0.12,
                    landing_time + 0.25,
                    0.24,
                    "goal_front",
                    fps,
                ),
            )
        )
        clips.append(
            _Clip(
                f"TRUE FLIGHT {flight_ms:.0f} ms · {rise_mm:.1f} mm RISE · FOOT-CONTACT LANDING",
                _segment(lane_id, save_time - 0.10, landing_time + 0.35, 0.35, "hero", fps),
            )
        )
    last_lane = next(reversed(cases))
    end = float(trajectories[last_lane]["time"][-1])
    span = float(evidence["contact_span_m"])
    clips.append(
        _Clip(
            f"4/4 STRICT REPLAYS · {span:.3f} m CONTACT SPAN · STABLE RECOVERY",
            tuple(_Frame(last_lane, end, "wide_goal") for _ in range(round(2.0 * fps))),
        )
    )
    return tuple(clips)


__all__ = ["render_dynamic_corner_video", "validate_dynamic_corner_video_manifest"]
