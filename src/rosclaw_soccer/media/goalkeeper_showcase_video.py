"""Evidence-bound goalkeeper-only multi-angle showcase.

The main reel replays only strict S85 CPU-MuJoCo trajectories.  A short
double-save bonus may be cut from an individually successful S37 rollout, but
its rejected policy-level exam remains permanently declared in the manifest
and in the pixels.  Pixels never feed back into scoring.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.media.trajectory_render import append_sphere, escape_filtergraph_option
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_three_player_stadium_model


@dataclass(frozen=True)
class _Frame:
    lane_id: str
    simulation_time_sec: float
    view: str


@dataclass(frozen=True)
class _Clip:
    label: str
    frames: tuple[_Frame, ...]


def validate_goalkeeper_showcase_manifest(path: Path) -> dict[str, Any]:
    """Fail closed if media, sources, or authority declarations drift."""

    manifest_path = path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("goalkeeper showcase manifest must be an object")
    expected = payload.pop("manifest_hash", None)
    if expected != hash_json(payload):
        raise ValueError("goalkeeper showcase manifest integrity mismatch")
    video_value = payload.get("video_path")
    source_values = payload.get("source_files")
    if not isinstance(video_value, str) or not isinstance(source_values, dict):
        raise ValueError("goalkeeper showcase paths are invalid")
    video_path = Path(video_value).expanduser().resolve()
    for source_path, source_hash in source_values.items():
        path_value = Path(source_path).expanduser().resolve()
        if not path_value.is_file() or hash_bytes(path_value.read_bytes()) != source_hash:
            raise ValueError("goalkeeper showcase source binding changed")
    metrics = payload.get("double_save_rollout")
    numbers = tuple(
        payload.get(name) for name in ("fps", "width", "height", "frame_count", "duration_sec")
    )
    if (
        payload.get("schema_version") != "rosclaw_soccer.goalkeeper_showcase_video.v1"
        or not video_path.is_file()
        or hash_bytes(video_path.read_bytes()) != payload.get("video_hash")
        or any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in numbers
        )
        or payload.get("strict_primary_save_count") != 4
        or payload.get("controlled_dive_source_status") != "REJECTED_NO_SAFE_CANDIDATE"
        or payload.get("controlled_dive_clip_claim")
        != "INDIVIDUAL_STABLE_QUALIFIED_SAVE_NOT_POLICY_PROMOTION"
        or not isinstance(payload.get("controlled_dive_rollout"), dict)
        or not all(
            payload["controlled_dive_rollout"].get(name) is True
            for name in (
                "first_save",
                "first_hand_save",
                "qualified_save",
                "recovered",
                "stable_save",
            )
        )
        or payload.get("double_save_source_status") != "REJECTED_BY_CPU_MUJOCO_EXAM"
        or payload.get("double_save_clip_claim")
        != "INDIVIDUAL_PASSED_ROLLOUT_NOT_POLICY_PROMOTION"
        or not isinstance(metrics, dict)
        or not all(metrics.get(name) is True for name in ("first_save", "recovered", "second_save"))
        or not (metrics.get("first_hand_save") is True or metrics.get("second_hand_save") is True)
        or payload.get("fall_then_second_save_included") is not False
        or payload.get("visualization_only") is not True
        or payload.get("pixels_used_for_scoring") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("goalkeeper showcase authority contract is invalid")
    payload["manifest_hash"] = expected
    return cast(dict[str, Any], payload)


def render_goalkeeper_showcase_video(
    *,
    portfolio_evidence_path: Path,
    controlled_dive_manifest_path: Path,
    controlled_dive_report_path: Path,
    double_save_manifest_path: Path,
    double_save_exam_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render strict saves plus honest controlled-dive and double-save bonuses."""

    evidence_file = portfolio_evidence_path.expanduser().resolve()
    dive_manifest_file = controlled_dive_manifest_path.expanduser().resolve()
    dive_report_file = controlled_dive_report_path.expanduser().resolve()
    double_manifest_file = double_save_manifest_path.expanduser().resolve()
    double_exam_file = double_save_exam_path.expanduser().resolve()
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
        raise ValueError("goalkeeper showcase output contract is invalid")
    evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict) or not (
        evidence.get("passed") is True
        and evidence.get("promotion_status") == "FROZEN_SIM_DEMO"
        and evidence.get("physics_authority") == "CPU_MUJOCO"
        and evidence.get("activation_ceiling") == "SIM_ONLY"
        and evidence.get("hardware_command_sent") is False
        and evidence.get("pixels_used_for_scoring") is False
        and isinstance(evidence.get("portfolio_gates"), dict)
        and all(evidence["portfolio_gates"].values())
    ):
        raise ValueError("goalkeeper showcase requires passed S85 evidence")
    request_file = evidence_file.parent / "request.json"
    if hash_bytes(request_file.read_bytes()) != evidence.get("request_hash"):
        raise ValueError("goalkeeper showcase request binding changed")
    request = json.loads(request_file.read_text(encoding="utf-8"))
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if qualification.body_hash != request.get("body_hash"):
        raise ValueError("goalkeeper showcase Body hash changed")

    cases_value = evidence.get("cases")
    goal_specs = request.get("lane_goal_specs")
    if not (
        isinstance(cases_value, dict)
        and len(cases_value) == 4
        and isinstance(goal_specs, dict)
    ):
        raise ValueError("goalkeeper showcase requires exactly four frozen lanes")
    trajectories: dict[str, dict[str, np.ndarray]] = {}
    cases: dict[str, dict[str, Any]] = {}
    goals: dict[str, G1TrainingGoalSpec] = {}
    trajectory_hashes: dict[str, str] = {}
    required = {
        "time",
        "ball_pose",
        "goalkeeper_pelvis_pose",
        "goalkeeper_joint_position",
    }
    for lane_id, raw_case in cases_value.items():
        if not isinstance(lane_id, str) or not isinstance(raw_case, dict) or not (
            raw_case.get("passed") is True and raw_case.get("strict_replay") is True
        ):
            raise ValueError("goalkeeper showcase lane did not strictly pass")
        replay = raw_case.get("replay")
        if not isinstance(replay, dict) or replay.get("passed") is not True:
            raise ValueError("goalkeeper showcase replay is invalid")
        gates = replay.get("gates")
        if not isinstance(gates, dict) or not all(gates.values()):
            raise ValueError("goalkeeper showcase replay gates are incomplete")
        name = raw_case.get("trajectory_file")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("goalkeeper showcase trajectory name is invalid")
        trajectory_path = evidence_file.parent / name
        trajectory_hash = hash_bytes(trajectory_path.read_bytes())
        if trajectory_hash != raw_case.get("trajectory_hash"):
            raise ValueError("goalkeeper showcase trajectory binding changed")
        with np.load(trajectory_path, allow_pickle=False) as archive:
            trajectory = {key: np.asarray(archive[key]) for key in archive.files}
        if not required <= set(trajectory):
            raise ValueError("goalkeeper showcase trajectory is incomplete")
        goal = G1TrainingGoalSpec(**goal_specs[lane_id])
        if abs(goal.width_m - 7.32) > 1e-9 or abs(goal.height_m - 2.44) > 1e-9:
            raise ValueError("goalkeeper showcase requires a regulation goal")
        cases[lane_id] = raw_case
        trajectories[lane_id] = trajectory
        goals[lane_id] = goal
        trajectory_hashes[lane_id] = trajectory_hash

    double_manifest = json.loads(double_manifest_file.read_text(encoding="utf-8"))
    expected_double_hash = double_manifest.pop("manifest_hash", None)
    if expected_double_hash != hash_json(double_manifest):
        raise ValueError("goalkeeper showcase double-save manifest drifted")
    double_video = Path(str(double_manifest.get("output_path", ""))).expanduser().resolve()
    double_exam = json.loads(double_exam_file.read_text(encoding="utf-8"))
    if not (
        double_manifest.get("cpu_exam_file_hash") == hash_bytes(double_exam_file.read_bytes())
        and double_manifest.get("video_hash") == hash_bytes(double_video.read_bytes())
        and double_exam.get("promotion_status") == "REJECTED_BY_CPU_MUJOCO_EXAM"
    ):
        raise ValueError("goalkeeper showcase double-save source contract changed")
    selected = double_manifest.get("selected_rollout_metrics")
    if not isinstance(selected, list):
        raise ValueError("goalkeeper showcase double-save metrics are unavailable")
    double_rollout = next(
        (
            row
            for row in selected
            if isinstance(row, dict)
            and row.get("first_save") is True
            and row.get("recovered") is True
            and row.get("second_save") is True
            and (row.get("first_hand_save") is True or row.get("second_hand_save") is True)
            and float(row.get("minimum_pelvis_height_m", 0.0)) >= 0.70
            and float(row.get("maximum_root_angular_speed_rad_s", math.inf)) <= 2.0
        ),
        None,
    )
    if double_rollout is None:
        raise ValueError("goalkeeper showcase has no individually safe double-save rollout")
    selected_seeds = double_manifest.get("selected_seeds")
    if not isinstance(selected_seeds, list) or double_rollout["seed"] not in selected_seeds:
        raise ValueError("goalkeeper showcase double-save seed binding is invalid")
    double_index = selected_seeds.index(double_rollout["seed"])
    segment_duration = float(double_manifest["duration_sec"]) / len(selected_seeds)
    double_start = double_index * segment_duration

    dive_manifest = json.loads(dive_manifest_file.read_text(encoding="utf-8"))
    dive_report = json.loads(dive_report_file.read_text(encoding="utf-8"))
    dive_video = Path(str(dive_manifest.get("video_path", ""))).expanduser().resolve()
    if not (
        dive_manifest.get("source_replay_file_hash") == hash_bytes(dive_report_file.read_bytes())
        and dive_manifest.get("video_hash") == hash_bytes(dive_video.read_bytes())
        and dive_manifest.get("promotion_status") == "REJECTED_NO_SAFE_CANDIDATE"
        and dive_report.get("promotion_status") == "REJECTED_NO_SAFE_CANDIDATE"
    ):
        raise ValueError("goalkeeper showcase controlled-dive source contract changed")
    dive_clips = dive_report.get("clips")
    if not isinstance(dive_clips, list):
        raise ValueError("goalkeeper showcase controlled-dive metrics are unavailable")
    dive_rollout = next(
        (
            row
            for row in dive_clips
            if isinstance(row, dict)
            and all(
                row.get(name) is True
                for name in (
                    "first_save",
                    "first_hand_save",
                    "qualified_save",
                    "recovered",
                    "stable_save",
                )
            )
            and row.get("failed") is False
            and float(row.get("maximum_root_angular_speed_rad_s", math.inf)) <= 3.0
        ),
        None,
    )
    if dive_rollout is None:
        raise ValueError("goalkeeper showcase has no individually stable controlled dive")
    dive_index = dive_clips.index(dive_rollout)
    dive_segment_duration = float(dive_manifest["duration_sec"]) / len(dive_clips)
    dive_start = dive_index * dive_segment_duration

    clips = _timeline(cases, trajectories, fps)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for goalkeeper showcase")
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import mujoco

        first_lane = next(iter(cases))
        goal = goals[first_lane]
        depth = float(request["config"]["aerial_config"]["goalkeeper_depth_from_goal_line_m"])
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
            with tempfile.TemporaryDirectory(prefix="rosclaw-goalkeeper-showcase-") as temp_text:
                temp = Path(temp_text)
                primary = temp / "keeper-primary.mp4"
                labels = _write_labels(temp, clips)
                process = subprocess.Popen(
                    _primary_ffmpeg_command(
                        ffmpeg=ffmpeg,
                        output=primary,
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
                    raise RuntimeError("goalkeeper showcase raw-video pipe is unavailable")
                try:
                    _write_primary_frames(
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
                    raise RuntimeError(
                        f"goalkeeper showcase primary encode failed: {stderr[-3000:]}"
                    )
                _compose_final(
                    ffmpeg=ffmpeg,
                    primary=primary,
                    dive_video=dive_video,
                    double_video=double_video,
                    output=output,
                    dive_start=dive_start,
                    dive_duration=dive_segment_duration,
                    double_start=double_start,
                    double_duration=segment_duration,
                )
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    probe = _probe(ffprobe, output)
    primary_frames = sum(len(clip.frames) for clip in clips)
    expected_frames = (
        primary_frames
        + round(dive_segment_duration * fps)
        + round(segment_duration * fps)
    )
    if (
        probe["width"] != width
        or probe["height"] != height
        or probe["fps"] != fps
        or abs(probe["frame_count"] - expected_frames) > 2
    ):
        raise RuntimeError("goalkeeper showcase encoded video contract changed")
    source_files = {
        str(evidence_file): hash_bytes(evidence_file.read_bytes()),
        str(request_file): hash_bytes(request_file.read_bytes()),
        str(dive_manifest_file): hash_bytes(dive_manifest_file.read_bytes()),
        str(dive_report_file): hash_bytes(dive_report_file.read_bytes()),
        str(dive_video): hash_bytes(dive_video.read_bytes()),
        str(double_manifest_file): hash_bytes(double_manifest_file.read_bytes()),
        str(double_exam_file): hash_bytes(double_exam_file.read_bytes()),
        str(double_video): hash_bytes(double_video.read_bytes()),
    }
    source_files.update(
        {
            str(evidence_file.parent / cases[lane_id]["trajectory_file"]): trajectory_hash
            for lane_id, trajectory_hash in trajectory_hashes.items()
        }
    )
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.goalkeeper_showcase_video.v1",
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_files": source_files,
        "strict_primary_save_count": len(cases),
        "strict_primary_contact_span_m": evidence["contact_span_m"],
        "strict_primary_trajectory_hashes": trajectory_hashes,
        "controlled_dive_source_status": dive_report["promotion_status"],
        "controlled_dive_clip_claim": (
            "INDIVIDUAL_STABLE_QUALIFIED_SAVE_NOT_POLICY_PROMOTION"
        ),
        "controlled_dive_rollout": dive_rollout,
        "double_save_source_status": double_exam["promotion_status"],
        "double_save_clip_claim": "INDIVIDUAL_PASSED_ROLLOUT_NOT_POLICY_PROMOTION",
        "double_save_rollout": double_rollout,
        "fall_then_second_save_included": False,
        "fall_then_second_save_reason": "NO_CONTINUOUS_STRICT_PHYSICS_EPISODE_AVAILABLE",
        "clips": [{"label": clip.label, "frame_count": len(clip.frames)} for clip in clips]
        + [
            {
                "label": "BONUS CONTROLLED LOW DIVE + RECOVERY · DEVELOPMENT ROLLOUT",
                "frame_count": round(dive_segment_duration * fps),
            }
        ]
        + [
            {
                "label": "BONUS CONTINUOUS DOUBLE SAVE · DEVELOPMENT ROLLOUT",
                "frame_count": round(segment_duration * fps),
            }
        ],
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": probe["frame_count"],
        "duration_sec": probe["frame_count"] / fps,
        "goal_contract": "REGULATION_7.32X2.44M_GOAL_FOR_STRICT_PRIMARY_CLIPS",
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "implementation_hash": hash_bytes(Path(__file__).read_bytes()),
    }
    manifest["manifest_hash"] = hash_json(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return validate_goalkeeper_showcase_manifest(manifest_path)


def validate_collision_faithful_goalkeeper_manifest(path: Path) -> dict[str, Any]:
    """Validate the strict-only showcase and its exact-contact source bindings."""

    manifest_path = path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("collision-faithful showcase manifest must be an object")
    expected = payload.pop("manifest_hash", None)
    if expected != hash_json(payload):
        raise ValueError("collision-faithful showcase manifest integrity mismatch")
    video = Path(str(payload.get("video_path", ""))).expanduser().resolve()
    sources = payload.get("source_files")
    contacts = payload.get("exact_glove_contacts")
    if not isinstance(sources, dict) or not isinstance(contacts, list):
        raise ValueError("collision-faithful showcase bindings are invalid")
    sources_valid = all(
        Path(source).expanduser().resolve().is_file()
        and hash_bytes(Path(source).expanduser().resolve().read_bytes()) == source_hash
        for source, source_hash in sources.items()
    )
    contacts_valid = len(contacts) == 4 and all(
        isinstance(contact, dict)
        and isinstance(contact.get("signed_surface_distance_m"), int | float)
        and -0.018
        <= float(contact["signed_surface_distance_m"])
        <= 0.001
        and contact.get("glove_side") in {"left", "right", "both"}
        and isinstance(contact.get("time_sec"), int | float)
        for contact in contacts
    )
    if not (
        payload.get("schema_version")
        == "rosclaw_soccer.collision_faithful_goalkeeper_video.v1"
        and video.is_file()
        and hash_bytes(video.read_bytes()) == payload.get("video_hash")
        and sources_valid
        and contacts_valid
        and payload.get("strict_save_count") == 4
        and payload.get("exact_contact_trace_hz") == 500
        and payload.get("positive_surface_separation_gate_m") == 0.001
        and payload.get("pixels_used_for_scoring") is False
        and payload.get("activation_ceiling") == "SIM_ONLY"
        and payload.get("hardware_command_sent") is False
    ):
        raise ValueError("collision-faithful showcase authority contract is invalid")
    payload["manifest_hash"] = expected
    return cast(dict[str, Any], payload)


def render_collision_faithful_goalkeeper_video(
    *,
    portfolio_evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 60,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render only strict 500 Hz glove-contact replays from passed evidence."""

    evidence_file = portfolio_evidence_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    manifest_path = output.with_suffix(".json")
    if (
        output.exists()
        or manifest_path.exists()
        or output.suffix.lower() != ".mp4"
        or output == checkout
        or checkout in output.parents
        or not 30 <= fps <= 60
        or not 1280 <= width <= 3840
        or not 720 <= height <= 2160
    ):
        raise ValueError("collision-faithful showcase output contract is invalid")
    evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict) or not (
        evidence.get("passed") is True
        and evidence.get("promotion_status") == "FROZEN_SIM_DEMO"
        and evidence.get("physics_authority") == "CPU_MUJOCO"
        and evidence.get("activation_ceiling") == "SIM_ONLY"
        and evidence.get("hardware_command_sent") is False
        and evidence.get("pixels_used_for_scoring") is False
        and isinstance(evidence.get("portfolio_gates"), dict)
        and all(evidence["portfolio_gates"].values())
    ):
        raise ValueError("collision-faithful showcase requires passed portfolio evidence")
    request_file = evidence_file.parent / "request.json"
    if hash_bytes(request_file.read_bytes()) != evidence.get("request_hash"):
        raise ValueError("collision-faithful showcase request binding changed")
    request = json.loads(request_file.read_text(encoding="utf-8"))
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if qualification.body_hash != request.get("body_hash"):
        raise ValueError("collision-faithful showcase Body hash changed")

    raw_cases = evidence.get("cases")
    raw_goals = request.get("lane_goal_specs")
    if not isinstance(raw_cases, dict) or len(raw_cases) != 4 or not isinstance(raw_goals, dict):
        raise ValueError("collision-faithful showcase requires four lanes")
    cases: dict[str, dict[str, Any]] = {}
    trajectories: dict[str, dict[str, np.ndarray]] = {}
    goals: dict[str, G1TrainingGoalSpec] = {}
    trajectory_hashes: dict[str, str] = {}
    exact_contacts: list[dict[str, Any]] = []
    required = {
        "time",
        "ball_pose",
        "goalkeeper_pelvis_pose",
        "goalkeeper_joint_position",
        "goalkeeper_contact_window_time",
        "goalkeeper_contact_window_ball_pose",
        "goalkeeper_contact_window_pelvis_pose",
        "goalkeeper_contact_window_joint_position",
    }
    for lane_id, raw_case in raw_cases.items():
        if not isinstance(lane_id, str) or not isinstance(raw_case, dict) or not (
            raw_case.get("passed") is True and raw_case.get("strict_replay") is True
        ):
            raise ValueError("collision-faithful showcase lane did not strictly pass")
        replay = raw_case.get("replay")
        gates = replay.get("gates") if isinstance(replay, dict) else None
        if not isinstance(replay, dict) or not isinstance(gates, dict) or not (
            replay.get("passed") is True
            and gates.get("collision_faithful_glove_contact") is True
            and all(gates.values())
        ):
            raise ValueError("collision-faithful showcase exact-contact gate failed")
        name = raw_case.get("trajectory_file")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("collision-faithful showcase trajectory name is invalid")
        trajectory_path = evidence_file.parent / name
        trajectory_hash = hash_bytes(trajectory_path.read_bytes())
        if trajectory_hash != raw_case.get("trajectory_hash"):
            raise ValueError("collision-faithful showcase trajectory binding changed")
        with np.load(trajectory_path, allow_pickle=False) as archive:
            trajectory = {key: np.asarray(archive[key]) for key in archive.files}
        if not required <= set(trajectory) or len(trajectory["goalkeeper_contact_window_time"]) < 2:
            raise ValueError("collision-faithful showcase lacks a 500 Hz contact window")
        result = replay.get("result")
        if not isinstance(result, dict):
            raise ValueError("collision-faithful showcase result is unavailable")
        signed_distance = replay.get("glove_contact_surface_distance_m")
        if not isinstance(signed_distance, int | float) or not (
            -0.018 <= float(signed_distance) <= 0.001
        ):
            raise ValueError("collision-faithful showcase surface distance is invalid")
        contact_time = replay.get("glove_contact_time_sec")
        side = replay.get("glove_contact_side")
        if not isinstance(contact_time, int | float) or side not in {"left", "right", "both"}:
            raise ValueError("collision-faithful showcase exact event is invalid")
        goal = G1TrainingGoalSpec(**raw_goals[lane_id])
        if abs(goal.width_m - 7.32) > 1e-9 or abs(goal.height_m - 2.44) > 1e-9:
            raise ValueError("collision-faithful showcase requires a regulation goal")
        cases[lane_id] = raw_case
        trajectories[lane_id] = trajectory
        goals[lane_id] = goal
        trajectory_hashes[lane_id] = trajectory_hash
        exact_contacts.append(
            {
                "lane_id": lane_id,
                "time_sec": float(contact_time),
                "position_m": replay["glove_contact_position_m"],
                "signed_surface_distance_m": float(signed_distance),
                "glove_side": side,
            }
        )

    clips = _timeline(cases, trajectories, fps)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for goalkeeper showcase")
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import mujoco

        first_lane = next(iter(cases))
        goal = goals[first_lane]
        depth = float(request["config"]["aerial_config"]["goalkeeper_depth_from_goal_line_m"])
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
            with tempfile.TemporaryDirectory(prefix="rosclaw-true-contact-") as temp_text:
                labels = _write_labels(Path(temp_text), clips)
                process = subprocess.Popen(
                    _primary_ffmpeg_command(
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
                    raise RuntimeError("collision-faithful raw-video pipe is unavailable")
                try:
                    _write_primary_frames(
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
                    raise RuntimeError(
                        f"collision-faithful showcase encode failed: {stderr[-3000:]}"
                    )
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    probe = _probe(ffprobe, output)
    expected_frames = sum(len(clip.frames) for clip in clips)
    if (
        probe["width"] != width
        or probe["height"] != height
        or probe["fps"] != fps
        or abs(probe["frame_count"] - expected_frames) > 2
    ):
        raise RuntimeError("collision-faithful encoded video contract changed")
    source_files = {
        str(evidence_file): hash_bytes(evidence_file.read_bytes()),
        str(request_file): hash_bytes(request_file.read_bytes()),
    }
    source_files.update(
        {
            str(evidence_file.parent / cases[lane_id]["trajectory_file"]): trajectory_hash
            for lane_id, trajectory_hash in trajectory_hashes.items()
        }
    )
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.collision_faithful_goalkeeper_video.v1",
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_files": source_files,
        "strict_save_count": len(cases),
        "strict_contact_span_m": evidence["contact_span_m"],
        "exact_glove_contacts": exact_contacts,
        "exact_contact_trace_hz": 500,
        "positive_surface_separation_gate_m": 0.001,
        "maximum_surface_penetration_m": 0.018,
        "glove_full_dimensions_m": (0.19, 0.10, 0.065),
        "clips": [
            {"label": clip.label, "frame_count": len(clip.frames)} for clip in clips
        ],
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": probe["frame_count"],
        "duration_sec": probe["frame_count"] / fps,
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "implementation_hash": hash_bytes(Path(__file__).read_bytes()),
    }
    manifest["manifest_hash"] = hash_json(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return validate_collision_faithful_goalkeeper_manifest(manifest_path)


def _timeline(
    cases: dict[str, dict[str, Any]], trajectories: dict[str, dict[str, np.ndarray]], fps: int
) -> tuple[_Clip, ...]:
    first_lane = next(iter(cases))
    first_time = float(trajectories[first_lane]["time"][0])
    clips: list[_Clip] = [
        _Clip(
            "G1 GOALKEEPER · FOUR ANGLES · FOUR STRICT SAVES",
            tuple(_Frame(first_lane, first_time, "hero") for _ in range(round(1.5 * fps))),
        )
    ]
    views = (
        ("hero", "glove_close"),
        ("goal_line", "glove_close"),
        ("hero", "glove_close"),
        ("keeper_side", "glove_close"),
    )
    for index, ((lane_id, case), (real_view, slow_view)) in enumerate(
        zip(cases.items(), views, strict=True), start=1
    ):
        replay = cast(dict[str, Any], case["replay"])
        result = cast(dict[str, Any], replay["result"])
        save_time = float(result["goalkeeper_glove_contact_time_sec"])
        signed_distance_mm = 1000.0 * float(replay["glove_contact_surface_distance_m"])
        lane_label = cast(dict[str, Any], case["lane"])["label"]
        clips.append(
            _Clip(
                f"SAVE {index}/4 · {lane_label} · REAL-TIME REACTION",
                _segment(lane_id, save_time - 0.70, save_time + 0.65, 1.0, real_view, fps),
            )
        )
        clips.append(
            _Clip(
                f"TRUE 2 ms CONTACT · SIGNED DIST {signed_distance_mm:+.1f} mm · NO GOAL",
                _segment(lane_id, save_time - 0.12, save_time, 0.20, slow_view, fps)
                + tuple(_Frame(lane_id, save_time, slow_view) for _ in range(round(0.12 * fps)))
                + _segment(lane_id, save_time, save_time + 0.28, 0.20, slow_view, fps),
            )
        )
    last_lane = next(reversed(cases))
    end = float(trajectories[last_lane]["time"][-1])
    contact_y = [
        float(cast(dict[str, Any], case["replay"])["glove_contact_position_m"][1])
        for case in cases.values()
    ]
    contact_span_m = max(contact_y) - min(contact_y)
    clips.append(
        _Clip(
            f"4/4 STRICT SAVES · {contact_span_m:.3f} m CONTACT SPAN · REGULATION GOAL",
            tuple(_Frame(last_lane, end, "hero") for _ in range(round(1.8 * fps))),
        )
    )
    return tuple(clips)


def _segment(
    lane_id: str, start: float, end: float, speed: float, view: str, fps: int
) -> tuple[_Frame, ...]:
    if end <= start or speed <= 0.0:
        raise ValueError("goalkeeper showcase segment is invalid")
    count = max(1, int(math.ceil((end - start) / speed * fps)))
    return tuple(
        _Frame(lane_id, min(end, start + index / fps * speed), view)
        for index in range(count)
    )


def _write_primary_frames(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    trajectories: dict[str, dict[str, np.ndarray]],
    clips: tuple[_Clip, ...],
    goal: G1TrainingGoalSpec,
    stream: BinaryIO,
) -> None:
    ball_body = _id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_joint = int(model.body_jntadr[ball_body])
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    goalkeeper_joint = _id(
        mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, "goalkeeper_floating_base_joint"
    )
    goalkeeper_free = int(model.jnt_qposadr[goalkeeper_joint])
    goalkeeper_joints = _joint_qpos(mujoco, model, "goalkeeper_")
    passer_joint = _id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, "passer_floating_base_joint")
    passer_free = int(model.jnt_qposadr[passer_joint])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    for clip in clips:
        for frame in clip.frames:
            trajectory = trajectories[frame.lane_id]
            sample = _sample(trajectory, frame.simulation_time_sec)
            data.qpos[:] = model.qpos0
            # Keep the reel goalkeeper-only without changing the scored states:
            # the attacker bodies are moved out of the render scene, while the
            # keeper and ball are replayed byte-for-byte from frozen evidence.
            data.qpos[:3] = (-30.0, 0.0, 0.80)
            data.qpos[passer_free : passer_free + 3] = (-32.0, 0.0, 0.80)
            data.qpos[goalkeeper_free : goalkeeper_free + 7] = sample["goalkeeper_pelvis_pose"]
            data.qpos[goalkeeper_joints] = sample["goalkeeper_joint_position"]
            data.qpos[ball_qpos : ball_qpos + 7] = sample["ball_pose"]
            mujoco.mj_forward(model, data)
            _set_camera(camera, frame.view, sample, goal)
            renderer.update_scene(data, camera=camera)
            stream.write(np.ascontiguousarray(renderer.render().copy()).tobytes())


def _sample(
    trajectory: dict[str, np.ndarray], simulation_time_sec: float
) -> dict[str, NDArray[np.float64] | int]:
    control_time = np.asarray(trajectory["time"], dtype=np.float64)
    control_index = int(
        np.clip(np.searchsorted(control_time, simulation_time_sec), 0, len(control_time) - 1)
    )
    window_time = np.asarray(
        trajectory.get("goalkeeper_contact_window_time", np.asarray([])),
        dtype=np.float64,
    )
    if (
        len(window_time) >= 2
        and window_time[0] <= simulation_time_sec <= window_time[-1]
    ):
        time = window_time
        key_prefix = "goalkeeper_contact_window_"
        pelvis = trajectory[f"{key_prefix}pelvis_pose"]
        joints = trajectory[f"{key_prefix}joint_position"]
        ball = trajectory[f"{key_prefix}ball_pose"]
    else:
        time = control_time
        pelvis = trajectory["goalkeeper_pelvis_pose"]
        joints = trajectory["goalkeeper_joint_position"]
        ball = trajectory["ball_pose"]
    upper = int(np.searchsorted(time, simulation_time_sec, side="right"))
    if upper <= 0:
        lower = upper = 0
        ratio = 0.0
    elif upper >= len(time):
        lower = upper = len(time) - 1
        ratio = 0.0
    else:
        lower = upper - 1
        ratio = float((simulation_time_sec - time[lower]) / (time[upper] - time[lower]))
    return {
        "index": control_index,
        "goalkeeper_pelvis_pose": _pose(
            pelvis[lower],
            pelvis[upper],
            ratio,
        ),
        "goalkeeper_joint_position": _lerp(
            joints[lower],
            joints[upper],
            ratio,
        ),
        "ball_pose": _pose(ball[lower], ball[upper], ratio),
    }


def _set_camera(
    camera: Any,
    view: str,
    sample: dict[str, NDArray[np.float64] | int],
    goal: G1TrainingGoalSpec,
) -> None:
    ball = cast(NDArray[np.float64], sample["ball_pose"])
    keeper = cast(NDArray[np.float64], sample["goalkeeper_pelvis_pose"])
    focus_y = 0.62 * float(ball[1]) + 0.38 * float(keeper[1])
    if view == "glove_close":
        camera.lookat[:] = (
            0.60 * float(ball[0]) + 0.40 * float(keeper[0]),
            focus_y,
            0.58 * float(ball[2]) + 0.42 * float(keeper[2]),
        )
        camera.distance = 3.05
        camera.azimuth = 76.0 if float(ball[1]) >= float(keeper[1]) else 284.0
        camera.elevation = -4.0
    elif view == "goal_line":
        camera.lookat[:] = (goal.plane_x_m - 0.42, focus_y, 1.14)
        camera.distance, camera.azimuth, camera.elevation = 4.15, 18.0, -3.0
    elif view == "keeper_side":
        camera.lookat[:] = (goal.plane_x_m - 0.55, focus_y, 1.13)
        camera.distance = 4.05
        camera.azimuth = 86.0 if float(ball[1]) >= float(keeper[1]) else 274.0
        camera.elevation = -2.0
    else:
        camera.lookat[:] = (goal.plane_x_m - 0.72, focus_y, 1.08)
        camera.distance, camera.azimuth, camera.elevation = 4.60, 154.0, -4.0


def _pose(left: np.ndarray, right: np.ndarray, ratio: float) -> NDArray[np.float64]:
    result: NDArray[np.float64] = np.empty(7, dtype=np.float64)
    result[:3] = _lerp(left[:3], right[:3], ratio)
    result[3:] = _slerp(left[3:], right[3:], ratio)
    return result


def _lerp(left: np.ndarray, right: np.ndarray, ratio: float) -> NDArray[np.float64]:
    start = np.asarray(left, dtype=np.float64)
    return np.asarray(start + ratio * (np.asarray(right, dtype=np.float64) - start))


def _slerp(left: np.ndarray, right: np.ndarray, ratio: float) -> NDArray[np.float64]:
    start = np.asarray(left, dtype=np.float64).copy()
    end = np.asarray(right, dtype=np.float64).copy()
    start /= np.linalg.norm(start)
    end /= np.linalg.norm(end)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        value = start + ratio * (end - start)
        return np.asarray(value / np.linalg.norm(value), dtype=np.float64)
    angle = float(np.arccos(dot))
    scale = float(np.sin(angle))
    return np.asarray(
        np.sin((1.0 - ratio) * angle) / scale * start
        + np.sin(ratio * angle) / scale * end,
        dtype=np.float64,
    )


def _add_ball_trail(mujoco: Any, scene: Any, trajectory: dict[str, np.ndarray], index: int) -> None:
    start = max(0, index - 24)
    indices = np.linspace(start, index, min(12, index - start + 1), dtype=int)
    for trail_index, alpha in zip(indices, np.linspace(0.03, 0.36, len(indices)), strict=True):
        append_sphere(
            mujoco,
            scene,
            np.asarray(trajectory["ball_pose"][trail_index, :3]),
            0.020,
            (0.20, 0.82, 1.0, float(alpha)),
        )


def _joint_qpos(mujoco: Any, model: Any, prefix: str) -> NDArray[np.int64]:
    return np.asarray(
        [
            model.jnt_qposadr[_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)]
            for name in G1_DDS_JOINT_NAMES
        ],
        dtype=np.int64,
    )


def _id(mujoco: Any, model: Any, object_type: Any, name: str) -> int:
    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise ValueError(f"goalkeeper showcase model is missing {name}")
    return value


def _write_labels(root: Path, clips: tuple[_Clip, ...]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for index, clip in enumerate(clips):
        path = root / f"label-{index}.txt"
        path.write_text(clip.label, encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def _primary_ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    width: int,
    height: int,
    fps: int,
    clips: tuple[_Clip, ...],
    labels: tuple[Path, ...],
) -> list[str]:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font_option = f"fontfile={escape_filtergraph_option(str(font))}:" if font.is_file() else ""
    scale = height / 720.0
    left = round(30 * scale)
    filters = [
        f"drawbox=x=0:y=0:w=iw:h={round(112 * scale)}:color=0x030711@0.82:t=fill",
        f"drawbox=x=0:y=h-{round(58 * scale)}:w=iw:h={round(58 * scale)}:"
        "color=0x030711@0.82:t=fill",
        f"drawtext={font_option}text='ROSClaw Soccer · G1 GOALKEEPER SHOWCASE':"
        f"expansion=none:x={left}:y={round(13 * scale)}:fontsize={round(32 * scale)}:"
        "fontcolor=white",
        f"drawtext={font_option}text='STRICT CPU MUJOCO REPLAY · REGULATION GOAL · SIM ONLY':"
        f"expansion=none:x={left}:y=h-{round(38 * scale)}:fontsize={round(18 * scale)}:"
        "fontcolor=0x8DD8FF",
    ]
    offset = 0.0
    for label, clip in zip(labels, clips, strict=True):
        end = offset + len(clip.frames) / fps
        filters.append(
            f"drawtext={font_option}textfile={escape_filtergraph_option(str(label))}:"
            f"expansion=none:x={left}:y={round(59 * scale)}:fontsize={round(20 * scale)}:"
            f"fontcolor=0x65F59A:enable='between(t,{offset:.6f},{end:.6f})'"
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


def _compose_final(
    *,
    ffmpeg: str,
    primary: Path,
    dive_video: Path,
    double_video: Path,
    output: Path,
    dive_start: float,
    dive_duration: float,
    double_start: float,
    double_duration: float,
) -> None:
    dive_title = "BONUS · CONTROLLED LOW DIVE + RECOVERY · INDIVIDUAL ROLLOUT"
    double_title = "BONUS · CONTINUOUS DOUBLE SAVE · INDIVIDUAL DEVELOPMENT ROLLOUT"
    filtergraph = (
        "[0:v]setpts=PTS-STARTPTS[v0];"
        f"[1:v]trim=start={dive_start:.6f}:duration={dive_duration:.6f},"
        "setpts=PTS-STARTPTS,"
        "drawbox=x=0:y=h-126:w=iw:h=126:color=0x030711@0.78:t=fill,"
        f"drawtext=text='{dive_title}':fontcolor=0xFFD166:fontsize=28:"
        "x=(w-text_w)/2:y=h-108,"
        "drawtext=text='POLICY TRAINING REJECTED · THIS STABLE SAVE ONLY · SIM ONLY':"
        "fontcolor=white:fontsize=22:x=(w-text_w)/2:y=h-63[v1];"
        f"[2:v]trim=start={double_start:.6f}:duration={double_duration:.6f},"
        "setpts=PTS-STARTPTS,"
        "drawbox=x=0:y=h-126:w=iw:h=126:color=0x030711@0.78:t=fill,"
        f"drawtext=text='{double_title}':fontcolor=0xFFD166:fontsize=28:"
        "x=(w-text_w)/2:y=h-108,"
        "drawtext=text='POLICY EXAM REJECTED · THIS ROLLOUT ONLY · SIM ONLY':"
        "fontcolor=white:fontsize=22:x=(w-text_w)/2:y=h-63[v2];"
        "[v0][v1][v2]concat=n=3:v=1:a=0[v]"
    )
    completed = subprocess.run(
        (
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(primary),
            "-i",
            str(dive_video),
            "-i",
            str(double_video),
            "-filter_complex",
            filtergraph,
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ),
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"goalkeeper showcase final encode failed: {completed.stderr[-3000:]}")


def _probe(ffprobe: str, path: Path) -> dict[str, int]:
    completed = subprocess.run(
        (
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = str(stream["avg_frame_rate"]).split("/", maxsplit=1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": round(int(numerator) / int(denominator)),
        "frame_count": int(stream["nb_read_frames"]),
    }


__all__ = [
    "render_collision_faithful_goalkeeper_video",
    "render_goalkeeper_showcase_video",
    "validate_collision_faithful_goalkeeper_manifest",
    "validate_goalkeeper_showcase_manifest",
]
