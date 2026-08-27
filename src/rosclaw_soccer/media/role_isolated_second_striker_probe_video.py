"""Evidence-downstream review reel for role-isolated contact candidates."""

from __future__ import annotations

import argparse
import inspect
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
from rosclaw_soccer.training.continuous_second_striker_save_exam import (
    physical_second_striker_kwargs,
)
from rosclaw_soccer.training.dynamic_corner_save import expanded_dynamic_corner_lanes
from rosclaw_soccer.training.role_isolated_second_striker_probe import (
    RoleIsolatedSecondStrikerProbeConfig,
    _role_isolated_exam_config,
    validate_role_isolated_second_striker_probe,
)
from rosclaw_soccer.world.field import build_g1_four_player_two_ball_stadium_model

_REJECTED_CLAIM = "ROLE_ISOLATED_SECOND_STRIKER_REJECTED_CANDIDATE_REVIEW_VIDEO"
_QUALIFIED_CLAIM = "ROLE_ISOLATED_SECOND_STRIKER_QUALIFIED_CONTROL_CANDIDATE_REVIEW_VIDEO"


def _implementation_hash() -> str:
    dependencies = (
        Path(__file__),
        Path(inspect.getfile(_write_frames)),
        Path(inspect.getfile(_segment)),
    )
    return str(
        hash_json(
            {path.name: hash_bytes(path.resolve().read_bytes()) for path in dependencies}
        )
    )


def validate_role_isolated_second_striker_probe_video(path: Path) -> dict[str, Any]:
    """Recompute video, source and authority bindings without using pixels as evidence."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("role-isolated video manifest must be an object")
    expected_hash = payload.pop("manifest_hash", None)
    try:
        video_value = payload.get("video_path")
        evidence_value = payload.get("evidence_path")
        sources = payload.get("source_files")
        if (
            not isinstance(video_value, str)
            or not isinstance(evidence_value, str)
            or not isinstance(sources, dict)
        ):
            raise ValueError("role-isolated video bindings are invalid")
        video = Path(video_value).expanduser().resolve()
        if not video.is_file() or payload.get("video_hash") != hash_bytes(video.read_bytes()):
            raise ValueError("role-isolated video bytes changed")
        for source_value, source_hash in sources.items():
            source = Path(source_value).expanduser().resolve()
            if not source.is_file() or hash_bytes(source.read_bytes()) != source_hash:
                raise ValueError("role-isolated video source binding changed")
        evidence = validate_role_isolated_second_striker_probe(Path(evidence_value))
        replays = cast(list[dict[str, Any]], evidence["replays"])
        diagnostics = cast(dict[str, Any], replays[0]["candidate_diagnostics"])
        plasticity = cast(dict[str, bool], evidence["plasticity_gates"])
        numeric = tuple(
            payload.get(name) for name in ("fps", "width", "height", "frame_count", "duration_sec")
        )
        promoted = evidence.get("candidate_promoted") is True
        schema = (
            "rosclaw_soccer.role_isolated_second_striker_probe_video.v2"
            if promoted
            else "rosclaw_soccer.role_isolated_second_striker_probe_video.v1"
        )
        status = (
            "QUALIFIED_DEVELOPMENT_CANDIDATE"
            if promoted
            else "REJECTED_NO_SUPPORTED_PLASTICITY"
        )
        claim = _QUALIFIED_CLAIM if promoted else _REJECTED_CLAIM
        selected_count = int(diagnostics["candidate_selected_frame_count"])
        if (
            payload.get("schema_version") != schema
            or payload.get("claim") != claim
            or payload.get("evidence_passed") is not True
            or payload.get("candidate_promoted") is not promoted
            or payload.get("candidate_status") != status
            or payload.get("evidence_report_hash") != evidence["report_hash"]
            or evidence.get("evidence_passed") is not True
            or payload.get("strict_replay") is not True
            or payload.get("strict_replay") is not evidence["evidence_gates"]["strict_replay"]
            or payload.get("complete_chain_retained") is not True
            or payload.get("complete_chain_retained") is not plasticity["complete_chain_passed"]
            or payload.get("candidate_selected_frame_count")
            != diagnostics["candidate_selected_frame_count"]
            or (selected_count <= 0 if promoted else selected_count != 0)
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
            or expected_hash != hash_json(payload)
            or any(
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in numeric
            )
        ):
            raise ValueError("role-isolated video authority or integrity contract is invalid")
    finally:
        if expected_hash is not None:
            payload["manifest_hash"] = expected_hash
    return cast(dict[str, Any], payload)


def render_role_isolated_second_striker_probe_video(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render a retained or candidate-controlled chain with explicit authority labels."""

    evidence_file = evidence_path.expanduser().resolve()
    root = asset_root.expanduser().resolve()
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
        raise ValueError("role-isolated video output contract is invalid")
    evidence = validate_role_isolated_second_striker_probe(evidence_file)
    plasticity = cast(dict[str, bool], evidence["plasticity_gates"])
    if evidence.get("evidence_passed") is not True or plasticity.get(
        "complete_chain_passed"
    ) is not True:
        raise ValueError("only a safe complete-chain role-isolated probe is render eligible")
    replays = cast(list[dict[str, Any]], evidence["replays"])
    if len(replays) != 2 or evidence["evidence_gates"]["strict_replay"] is not True:
        raise ValueError("role-isolated video requires exact replay reports")
    request_path = evidence_file.parent / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if evidence.get("request_hash") != hash_bytes(request_path.read_bytes()):
        raise ValueError("role-isolated video request binding changed")
    trajectory_path = evidence_file.parent / str(replays[0]["trajectory_file"])
    if replays[0].get("trajectory_hash") != hash_bytes(trajectory_path.read_bytes()):
        raise ValueError("role-isolated video trajectory binding changed")
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
        raise ValueError("role-isolated video trajectory is incomplete")
    trajectory = dict(raw)
    trajectory["first_ball_pose"] = raw["ball_pose"]
    trajectory["source_shooter_pelvis_pose"] = raw["shooter_pelvis_pose"]
    trajectory["source_shooter_joint_position"] = raw["shooter_joint_position"]
    result = cast(dict[str, Any], replays[0]["result"])
    diagnostics = cast(dict[str, Any], replays[0]["candidate_diagnostics"])
    promoted = evidence.get("candidate_promoted") is True
    raw_config = cast(dict[str, Any], request["config"])
    config = RoleIsolatedSecondStrikerProbeConfig(**raw_config)
    motion_curriculum = config.second_striker_foot_pitch_offset_rad is not None
    clips = _timeline(
        trajectory,
        result,
        diagnostics,
        fps,
        promoted=promoted,
        motion_curriculum=motion_curriculum,
    )
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for the S113 review video")
    lane = next(
        lane for lane in expanded_dynamic_corner_lanes() if lane.lane_id == config.lane_id
    )
    locators = cast(dict[str, str], request["artifact_locators"])
    g1_paths = {
        "g1-policy": Path("policy/robonaldo/model/policy-obs-aic.onnx"),
        "g1-motion": Path("policy/robonaldo/model/freekick_motion.npz"),
        "g1-scene": Path("g1_description/scene_with_ball.xml"),
        "g1-model": Path("g1_description/g1_liao.xml"),
        "g1-free-kick": Path("policy/robonaldo/FreeKick.py"),
    }
    if any(
        (root / relative).resolve() != Path(locators[label]).resolve()
        for label, relative in g1_paths.items()
    ):
        raise ValueError("role-isolated video asset root changed")
    assets = {
        key: Path(locators[key.replace("_", "-")])
        for key in (
            "striker_actor",
            "goalkeeper_actor",
            "gmt_model",
            "gmt_skill",
            "dive_athlete_checkpoint",
            "dive_athlete_exam",
            "recovery_athlete_checkpoint",
            "recovery_athlete_exam",
        )
    }
    assets["dive_source"] = Path(
        cast(dict[str, str], request["source_tree_locators"])["dive-source"]
    )
    exam = _role_isolated_exam_config(config)
    _, _, goal = physical_second_striker_kwargs(
        lane=lane,
        assets=assets,
        recovery_checkpoint=assets["recovery_athlete_checkpoint"],
        recovery_exam=assets["recovery_athlete_exam"],
        config=exam,
    )
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        qualification = qualify_g1_assets(root)
        qualification.require_eligible()
        import mujoco

        striker = exam.striker
        model = build_g1_four_player_two_ball_stadium_model(
            root,
            passer_origin_m=(5.10, -0.864060065039216, 0.0),
            goalkeeper_origin_m=(goal.plane_x_m - 0.30, 0.0, 0.0),
            second_striker_origin_m=striker.origin_m,
            first_ball_origin_m=(1.0, 0.0, goal.ball_radius_m),
            second_ball_origin_m=striker.ball_origin_m,
            second_ball_mass_kg=config.second_ball_mass_kg,
            spec=goal,
        )
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-s113-video-") as temporary:
                labels = _write_labels(Path(temporary), clips)
                process = subprocess.Popen(
                    _video_ffmpeg_command(
                        ffmpeg=ffmpeg,
                        output=output,
                        width=width,
                        height=height,
                        fps=fps,
                        clips=clips,
                        labels=labels,
                        promoted=promoted,
                        motion_curriculum=motion_curriculum,
                    ),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                if process.stdin is None:
                    raise RuntimeError("S113 raw-video pipe is unavailable")
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
                    raise RuntimeError(f"S113 ffmpeg failed: {stderr[-3000:]}")
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
        raise RuntimeError("role-isolated encoded video contract changed")
    manifest: dict[str, Any] = {
        "schema_version": (
            "rosclaw_soccer.role_isolated_second_striker_probe_video.v2"
            if promoted
            else "rosclaw_soccer.role_isolated_second_striker_probe_video.v1"
        ),
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "evidence_path": str(evidence_file),
        "source_files": {
            str(evidence_file): hash_bytes(evidence_file.read_bytes()),
            str(request_path): hash_bytes(request_path.read_bytes()),
            str(trajectory_path): hash_bytes(trajectory_path.read_bytes()),
        },
        "claim": _QUALIFIED_CLAIM if promoted else _REJECTED_CLAIM,
        "evidence_report_hash": evidence["report_hash"],
        "evidence_passed": True,
        "candidate_promoted": promoted,
        "candidate_status": evidence["candidate_status"],
        "strict_replay": evidence["evidence_gates"]["strict_replay"],
        "complete_chain_retained": plasticity["complete_chain_passed"],
        "conditioned_frame_count": diagnostics["conditioned_frame_count"],
        "supported_frame_count": diagnostics["supported_frame_count"],
        "candidate_selected_frame_count": diagnostics["candidate_selected_frame_count"],
        "frozen_parent_selected_frame_count": diagnostics[
            "frozen_parent_selected_frame_count"
        ],
        "four_g1_visible": True,
        "two_physical_balls_visible": True,
        "two_physical_saves": True,
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
    return validate_role_isolated_second_striker_probe_video(manifest_path)


def _timeline(
    trajectory: dict[str, np.ndarray],
    result: dict[str, Any],
    diagnostics: dict[str, Any],
    fps: int,
    *,
    promoted: bool,
    motion_curriculum: bool = False,
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
    parent_frames = int(diagnostics["frozen_parent_selected_frame_count"])
    candidate_frames = int(diagnostics["candidate_selected_frame_count"])
    title = tuple(_Frame("left-inner", start, "four") for _ in range(round(1.8 * fps)))
    final = tuple(_Frame("left-inner", end, "goal") for _ in range(round(2.0 * fps)))
    audit_title = (
        "S115 · HEAVY-BALL WHOLE-BODY CURRICULUM"
        if motion_curriculum
        else (
            "S114 · FAILURE MEMORY → CONTROL QUALIFICATION"
            if promoted
            else "S113 · STABILITY–PLASTICITY AUDIT · CANDIDATE REJECTED"
        )
    )
    approach_label = (
        f"CURRICULUM BODY PITCH + LEARNED CONTACT ACTOR · {candidate_frames} CONTACT FRAMES"
        if motion_curriculum
        else (
            f"FAILURE-UPDATED CANDIDATE SELECTED · {candidate_frames} CONTACT FRAMES"
            if promoted
            else f"TARGET CANDIDATE ABSTAINS · FROZEN PARENT {parent_frames} FRAMES"
        )
    )
    contact_label = (
        f"LEARNED RIGHT-FOOT CONTACT · {force:.0f} N · NO CANNON"
        if promoted
        else f"PARENT RIGHT-FOOT CONTACT · {force:.0f} N · NO CANNON"
    )
    decision_label = (
        "HEAVY CONTROL PASSED · NEIGHBOR HOLDOUT REQUIRED · SIM ONLY"
        if motion_curriculum
        else (
            "CONTROL PASSED · SEALED HOLDOUT STILL REQUIRED · SIM ONLY"
            if promoted
            else "RETENTION PASSED · PLASTICITY FAILED · NO PROMOTION"
        )
    )
    return (
        _Clip(audit_title, title),
        _Clip(
            "FROZEN PREFIX · PASS → HIGH SHOT → AIRBORNE SAVE",
            _segment("left-inner", 4.4, first + 0.45, 1.0, "four", fps),
        ),
        _Clip(
            f"FIRST TRUE GLOVE SAVE · {first_height:.3f} m",
            _segment("left-inner", first - 0.50, first + 0.55, 0.34, "goal", fps),
        ),
        _Clip(
            "MEASURED RECOVERY → READY → PHYSICAL HANDOFF",
            _segment("left-inner", first + 0.55, rearm + 0.10, 2.35, "goal", fps),
        ),
        _Clip(
            approach_label,
            _segment("left-inner", 12.0, strike - 0.28, 1.35, "striker", fps),
        ),
        _Clip(
            contact_label,
            _segment("left-inner", strike - 0.42, second + 0.28, 0.42, "contact", fps),
        ),
        _Clip(
            f"RETAINED SECOND SAVE · {second_height:.3f} m · BALL CLEARED",
            _segment("left-inner", second - 0.42, second + 0.78, 0.31, "goal", fps),
        ),
        _Clip(
            decision_label,
            _segment("left-inner", start, end, 2.1, "four", fps),
        ),
        _Clip("DEVELOPMENT REVIEW · SIM ONLY · PIXELS NEVER SCORE", final),
    )


def _video_ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    width: int,
    height: int,
    fps: int,
    clips: tuple[_Clip, ...],
    labels: tuple[Path, ...],
    promoted: bool,
    motion_curriculum: bool = False,
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
        (
            "ROSClaw Soccer · S115 HEAVY-BALL MOTION CURRICULUM"
            if motion_curriculum
            else (
                "ROSClaw Soccer · S114 FAILURE-DRIVEN CONTACT GROWTH"
                if promoted
                else "ROSClaw Soccer · S113 SAFE REJECTION REVIEW"
            )
        ),
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
    manifest = render_role_isolated_second_striker_probe_video(
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
    "render_role_isolated_second_striker_probe_video",
    "validate_role_isolated_second_striker_probe_video",
]
