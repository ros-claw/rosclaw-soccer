"""Evidence-bound reel for contact-grounded airborne-save successors."""

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

from rosclaw_soccer.media.three_role_aerial_save_video import (
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
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_three_player_stadium_model

_CLAIM = "TRUE_AIRBORNE_SAVE_WITH_FOOT_CONTACT_GROUNDED_LANDING"


def validate_dynamic_takeoff_video_manifest(path: Path) -> dict[str, Any]:
    """Verify every source binding and the visualization authority ceiling."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dynamic takeoff video manifest must be an object")
    expected = payload.pop("manifest_hash", None)
    if expected != hash_json(payload):
        raise ValueError("dynamic takeoff video manifest integrity mismatch")
    video_value = payload.get("video_path")
    source_files = payload.get("source_files")
    if not isinstance(video_value, str) or not isinstance(source_files, dict):
        raise ValueError("dynamic takeoff video bindings are invalid")
    video = Path(video_value).expanduser().resolve()
    if not video.is_file() or hash_bytes(video.read_bytes()) != payload.get("video_hash"):
        raise ValueError("dynamic takeoff video hash changed")
    for source_value, source_hash in source_files.items():
        source = Path(source_value).expanduser().resolve()
        if not source.is_file() or hash_bytes(source.read_bytes()) != source_hash:
            raise ValueError("dynamic takeoff video source binding changed")
    numbers = tuple(
        payload.get(name) for name in ("fps", "width", "height", "frame_count", "duration_sec")
    )
    if (
        payload.get("schema_version") != "rosclaw_soccer.dynamic_takeoff_video.v1"
        or payload.get("claim") != _CLAIM
        or payload.get("strict_replay") is not True
        or payload.get("true_airborne") is not True
        or payload.get("foot_contact_grounded") is not True
        or payload.get("true_glove_contact") is not True
        or payload.get("bounded_landing") is not True
        or payload.get("post_save_recovered") is not True
        or payload.get("commercial_use_allowed") is not False
        or payload.get("visualization_only") is not True
        or payload.get("pixels_used_for_scoring") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
        or any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in numbers
        )
    ):
        raise ValueError("dynamic takeoff video authority contract is invalid")
    payload["manifest_hash"] = expected
    return cast(dict[str, Any], payload)


def render_dynamic_takeoff_video(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 60,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render the frozen replay; no image observation enters any score."""

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
        raise ValueError("dynamic takeoff video output contract is invalid")
    evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
    replay = evidence.get("replay") if isinstance(evidence, dict) else None
    gates = replay.get("gates") if isinstance(replay, dict) else None
    metrics = replay.get("metrics") if isinstance(replay, dict) else None
    base = replay.get("base") if isinstance(replay, dict) else None
    base_gates = base.get("gates") if isinstance(base, dict) else None
    result = base.get("result") if isinstance(base, dict) else None
    if not (
        isinstance(evidence, dict)
        and evidence.get("passed") is True
        and evidence.get("strict_replay") is True
        and evidence.get("promotion_status") == "FROZEN_RESEARCH_DEMO"
        and evidence.get("physics_authority") == "CPU_MUJOCO"
        and evidence.get("claim") == _CLAIM
        and evidence.get("commercial_use_allowed") is False
        and evidence.get("activation_ceiling") == "SIM_ONLY"
        and evidence.get("hardware_command_sent") is False
        and evidence.get("pixels_used_for_scoring") is False
        and isinstance(replay, dict)
        and replay.get("passed") is True
        and isinstance(gates, dict)
        and all(gates.values())
        and isinstance(metrics, dict)
        and isinstance(base, dict)
        and base.get("passed") is True
        and isinstance(base_gates, dict)
        and all(base_gates.values())
        and isinstance(result, dict)
        and result.get("goalkeeper_save_observed") is True
        and result.get("goal_crossed") is False
    ):
        raise ValueError("dynamic takeoff evidence is not render eligible")
    request_path = evidence_file.parent / "request.json"
    trajectory_name = evidence.get("trajectory_file")
    if not isinstance(trajectory_name, str) or Path(trajectory_name).name != trajectory_name:
        raise ValueError("dynamic takeoff trajectory name is invalid")
    trajectory_path = evidence_file.parent / trajectory_name
    if (
        hash_bytes(request_path.read_bytes()) != evidence.get("request_hash")
        or hash_bytes(trajectory_path.read_bytes()) != evidence.get("trajectory_hash")
    ):
        raise ValueError("dynamic takeoff evidence bindings changed")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    goal_value = request.get("goal_spec")
    config_value = request.get("config")
    if not isinstance(goal_value, dict) or not isinstance(config_value, dict):
        raise ValueError("dynamic takeoff request is incomplete")
    goal = G1TrainingGoalSpec(**goal_value)
    if abs(goal.width_m - 7.32) > 1e-9 or abs(goal.height_m - 2.44) > 1e-9:
        raise ValueError("dynamic takeoff video requires a regulation goal")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        trajectory = {name: np.asarray(archive[name]) for name in archive.files}
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
    if not required <= set(trajectory):
        raise ValueError("dynamic takeoff trajectory is incomplete")
    clips = _timeline(result, metrics, trajectory, fps)
    landing_capture_active = np.asarray(
        trajectory.get("goalkeeper_landing_capture_active", np.zeros(0, dtype=bool)),
        dtype=bool,
    )
    landing_capture_observed = bool(np.any(landing_capture_active))
    landing_capture_duration_sec = _observed_mask_duration_sec(
        np.asarray(trajectory["time"], dtype=float),
        landing_capture_active,
    )
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for dynamic takeoff video")
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        if qualification.body_hash != request.get("body_hash"):
            raise ValueError("dynamic takeoff Body hash changed")
        lunge = config_value.get("lunge_config")
        aerial = lunge.get("aerial_config") if isinstance(lunge, dict) else None
        lane = lunge.get("lane") if isinstance(lunge, dict) else None
        if not isinstance(lane, dict) or not isinstance(aerial, dict):
            raise ValueError("dynamic takeoff lane contract is incomplete")
        attacker_offset = float(lane["attacker_lateral_offset_m"])
        goalkeeper_lateral = float(lane["goalkeeper_initial_lateral_m"])
        goalkeeper_depth = float(aerial["goalkeeper_depth_from_goal_line_m"])
        import mujoco

        model = build_g1_three_player_stadium_model(
            asset_root.expanduser().resolve(),
            passer_origin_m=(5.10, -0.16406006503921598 + attacker_offset, 0.0),
            goalkeeper_origin_m=(
                goal.plane_x_m - goalkeeper_depth,
                goalkeeper_lateral,
                0.0,
            ),
            spec=goal,
        )
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-dynamic-takeoff-") as temp_text:
                labels = _write_labels(Path(temp_text), clips)
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
                    raise RuntimeError("dynamic takeoff raw-video pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        renderer=renderer,
                        trajectory=trajectory,
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
                    raise RuntimeError(f"dynamic takeoff ffmpeg failed: {stderr[-3000:]}")
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
        raise RuntimeError("dynamic takeoff encoded video contract changed")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.dynamic_takeoff_video.v1",
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_files": {
            str(evidence_file): hash_bytes(evidence_file.read_bytes()),
            str(request_path): hash_bytes(request_path.read_bytes()),
            str(trajectory_path): hash_bytes(trajectory_path.read_bytes()),
        },
        "claim": _CLAIM,
        "strict_replay": True,
        "true_airborne": True,
        "airborne_duration_sec": metrics["airborne_duration_sec"],
        "takeoff_peak_vertical_speed_mps": metrics["takeoff_peak_vertical_speed_mps"],
        "flight_pelvis_rise_m": metrics["flight_pelvis_rise_m"],
        "foot_contact_grounded": True,
        "true_glove_contact": True,
        "glove_contact_side": result["goalkeeper_glove_contact_side"],
        "glove_surface_distance_m": result["goalkeeper_glove_contact_surface_distance_m"],
        "bounded_landing": True,
        "landing_vertical_speed_mps": metrics["landing_vertical_speed_mps"],
        "landing_angular_speed_rad_s": metrics["landing_angular_speed_rad_s"],
        "landing_capture_observed": landing_capture_observed,
        "landing_capture_duration_sec": landing_capture_duration_sec,
        "post_save_recovered": True,
        "commercial_use_allowed": False,
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
    }
    manifest["manifest_hash"] = hash_json(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_dynamic_takeoff_video_manifest(manifest_path)
    return manifest


def _timeline(
    result: dict[str, Any],
    metrics: dict[str, Any],
    trajectory: dict[str, np.ndarray],
    fps: int,
) -> tuple[_Clip, ...]:
    start = float(trajectory["time"][0])
    end = float(trajectory["time"][-1])
    pass_time = float(result["pass_contact_time_sec"])
    shot_time = float(result["shot_contact_time_sec"])
    save_time = float(result["goalkeeper_glove_contact_time_sec"])
    flight_start = float(metrics["airborne_start_sec"])
    flight_stop = float(metrics["airborne_stop_sec"])
    landing_time = float(metrics["landing_time_sec"])
    intro = tuple(_Frame(max(start, pass_time - 0.8), "wide") for _ in range(round(1.5 * fps)))
    continuous = (
        *_segment(max(start, pass_time - 0.8), shot_time - 0.30, 1.0, "pass", fps),
        *_segment(shot_time - 0.30, save_time + 0.70, 1.0, "hero", fps),
    )
    takeoff = _segment(shot_time - 0.22, flight_stop + 0.22, 0.20, "hero", fps)
    contact = _segment(flight_start - 0.14, landing_time + 0.16, 0.10, "goal_front", fps)
    landing = _segment(flight_start - 0.10, landing_time + 0.55, 0.22, "hero", fps)
    recovery = _segment(landing_time + 0.25, end, 1.0, "wide_goal", fps)
    finale = tuple(_Frame(end, "wide_goal") for _ in range(round(1.6 * fps)))
    surface_mm = 1_000.0 * float(result["goalkeeper_glove_contact_surface_distance_m"])
    duration_ms = 1_000.0 * float(metrics["airborne_duration_sec"])
    rise_mm = 1_000.0 * float(metrics["flight_pelvis_rise_m"])
    capture_mask = np.asarray(
        trajectory.get("goalkeeper_landing_capture_active", np.zeros(0, dtype=bool)),
        dtype=bool,
    )
    capture_duration = _observed_mask_duration_sec(
        np.asarray(trajectory["time"], dtype=float), capture_mask
    )
    if capture_duration > 0.0:
        landing_label = (
            "FOOT-CONTACT LANDING · "
            f"{metrics['landing_vertical_speed_mps']:.3f} m/s · "
            f"{1_000.0 * capture_duration:.0f} ms PROPRIOCEPTIVE CAPTURE"
        )
        recovery_label = "CAPTURE → FEEDBACK FOUNDATION · STABLE RECOVERY · NO FALL"
    else:
        landing_label = (
            f"FOOT-CONTACT LANDING · {metrics['landing_vertical_speed_mps']:.3f} m/s VERTICAL"
        )
        recovery_label = "BOUNDED LANDING → STABLE RECOVERY · NO FALL"
    return (
        _Clip("THREE G1 · ONE BALL · ONE PHYSICAL WORLD", intro),
        _Clip("CONTINUOUS PASS → HIGH STRIKE → AIRBORNE SAVE", continuous),
        _Clip(
            f"TRUE TAKEOFF · BOTH FEET CLEAR {duration_ms:.0f} ms · {rise_mm:.1f} mm RISE",
            takeoff,
        ),
        _Clip(
            f"RIGHT-GLOVE CONTACT IN FLIGHT · {surface_mm:+.2f} mm · NO GOAL",
            contact,
        ),
        _Clip(landing_label, landing),
        _Clip(recovery_label, recovery),
        _Clip("STRICT CPU MUJOCO REPLAY PASS · SIM ONLY · RESEARCH ONLY", finale),
    )


def _observed_mask_duration_sec(time: np.ndarray, mask: np.ndarray) -> float:
    """Return first-to-last observed duration for a trajectory-owned event mask."""

    if time.ndim != 1 or mask.ndim != 1 or time.shape != mask.shape or not np.any(mask):
        return 0.0
    selected = time[mask]
    if selected.size == 1:
        if time.size < 2:
            return 0.0
        return float(np.median(np.diff(time)))
    return float(selected[-1] - selected[0])


__all__ = ["render_dynamic_takeoff_video", "validate_dynamic_takeoff_video_manifest"]
