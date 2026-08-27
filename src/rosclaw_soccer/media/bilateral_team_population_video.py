"""Render an honestly labelled S110 bilateral Growth diagnostic reel."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from rosclaw_soccer.media.continuous_second_striker_save_video import _write_frames
from rosclaw_soccer.media.three_role_save_portfolio_video import (
    _Clip,
    _Frame,
    _probe,
    _segment,
    _write_labels,
)
from rosclaw_soccer.media.trajectory_render import escape_filtergraph_option
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.bilateral_continuous_team_population import (
    validate_bilateral_continuous_team_population,
)
from rosclaw_soccer.world.field import (
    G1TrainingGoalSpec,
    build_g1_four_player_two_ball_stadium_model,
)

_CLAIM = "BILATERAL_PERTURBED_CONTINUOUS_TEAM_POPULATION_DIAGNOSTIC"
_CASE_IDS = ("right-control", "left-foot-frontier")


def _implementation_hash() -> str:
    return str(hash_bytes(Path(__file__).read_bytes()))


def validate_bilateral_team_population_video(path: Path) -> dict[str, Any]:
    """Validate bytes, source bindings and permanent rejected truth label."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("bilateral population video manifest must be an object")
    unhashed = dict(payload)
    expected = unhashed.pop("manifest_hash", None)
    sources = payload.get("source_files")
    video_value = payload.get("video_path")
    if (
        expected != hash_json(unhashed)
        or not isinstance(sources, dict)
        or not isinstance(video_value, str)
    ):
        raise ValueError("bilateral population video integrity mismatch")
    video = Path(video_value).expanduser().resolve()
    if not video.is_file() or payload.get("video_hash") != hash_bytes(video.read_bytes()):
        raise ValueError("bilateral population video bytes changed")
    for source_value, source_hash in sources.items():
        source = Path(source_value).expanduser().resolve()
        if not source.is_file() or source_hash != hash_bytes(source.read_bytes()):
            raise ValueError("bilateral population video source binding changed")
    numeric = tuple(
        payload.get(name) for name in ("fps", "width", "height", "frame_count", "duration_sec")
    )
    if (
        payload.get("schema_version") != "rosclaw_soccer.bilateral_team_population_video.v1"
        or payload.get("claim") != _CLAIM
        or payload.get("truth_label") != "REJECTED_DEVELOPMENT"
        or payload.get("evidence_passed") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("strict_replay_all_cases") is not True
        or payload.get("observed_contact_feet") != ["left", "right"]
        or payload.get("visualization_only") is not True
        or payload.get("pixels_used_for_scoring") is not False
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
        or payload.get("commercial_use_allowed") is not False
        or payload.get("implementation_hash") != _implementation_hash()
        or not all(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
            for value in numeric
        )
    ):
        raise ValueError("bilateral population video authority contract is invalid")
    return cast(dict[str, Any], payload)


def _load_trajectory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
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
        raise ValueError("bilateral population trajectory is incomplete")
    raw["first_ball_pose"] = raw["ball_pose"]
    raw["source_shooter_pelvis_pose"] = raw["shooter_pelvis_pose"]
    raw["source_shooter_joint_position"] = raw["shooter_joint_position"]
    return raw


def _case_clips(
    case_id: str,
    trajectory: dict[str, np.ndarray],
    result: dict[str, Any],
    fps: int,
) -> tuple[_Clip, ...]:
    end = float(trajectory["time"][-1])
    strike = float(result["second_striker_contact_time_sec"])
    foot = str(result["second_striker_contact_foot"]).upper()
    speed = float(result["second_striker_postcontact_peak_ball_speed_mps"])
    prefix = "FROZEN RIGHT CONTROL" if case_id == "right-control" else "LEFT-FOOT FRONTIER"
    return (
        _Clip(
            f"{prefix} · CURRENT RUNTIME RE-EXAM",
            tuple(_Frame(case_id, 0.0, "four") for _ in range(round(1.5 * fps))),
        ),
        _Clip(
            "PASS → FIRST SHOT → FIRST GLOVE CONTACT",
            _segment(case_id, 4.4, 8.65, 0.85, "goal", fps),
        ),
        _Clip(
            "NO RESET · MEASURED RECOVERY · SECOND STRIKER APPROACH",
            _segment(case_id, 8.6, strike - 0.30, 2.2, "four", fps),
        ),
        _Clip(
            f"ANATOMICAL {foot}-FOOT CONTACT · {speed:.2f} m/s",
            _segment(case_id, strike - 0.45, min(strike + 1.55, end), 0.42, "contact", fps),
        ),
        _Clip(
            "COUNTEREXAMPLE RETAINED · NOT A QUALIFIED SAVE",
            _segment(case_id, min(strike + 0.80, end - 0.1), end, 1.65, "goal", fps),
        ),
    )


def render_bilateral_team_population_video(
    *,
    evidence_path: Path,
    goal_contract_path: Path,
    asset_root: Path,
    output_path: Path,
    fps: int = 60,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render right/left counterexamples; video pixels never score policy."""

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
        raise ValueError("bilateral population video output contract is invalid")
    evidence = validate_bilateral_continuous_team_population(evidence_file)
    if evidence.get("passed") is not False or evidence.get("promotion_status") != (
        "REJECTED_BILATERAL_POPULATION"
    ):
        raise ValueError("bilateral reel requires rejected population evidence")
    request_path = evidence_file.parent / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request_unhashed = dict(request)
    request_hash = request_unhashed.pop("request_hash", None)
    if request_hash != hash_json(request_unhashed) or request_hash != evidence.get("request_hash"):
        raise ValueError("bilateral population video request binding changed")
    goal_contract = json.loads(goal_contract_file.read_text(encoding="utf-8"))
    goal_specs = goal_contract.get("lane_goal_specs")
    if (
        not isinstance(goal_specs, dict)
        or "left-inner" not in goal_specs
        or goal_contract.get("body_hash") != request.get("body_hash")
    ):
        raise ValueError("bilateral population goal contract changed")
    base_goal = G1TrainingGoalSpec(**goal_specs["left-inner"])
    cases = cast(dict[str, dict[str, Any]], evidence["cases"])
    trajectories: dict[str, dict[str, np.ndarray]] = {}
    clips_by_case: dict[str, tuple[_Clip, ...]] = {}
    trajectory_paths: dict[str, Path] = {}
    for case_id in _CASE_IDS:
        case = cases[case_id]
        if case.get("strict_replay") is not True:
            raise ValueError("bilateral reel source lacks strict replay")
        trajectory_path = evidence_file.parent / str(case["trajectory_file"])
        if case.get("trajectory_hash") != hash_bytes(trajectory_path.read_bytes()):
            raise ValueError("bilateral reel trajectory changed")
        trajectory = _load_trajectory(trajectory_path)
        result = cast(dict[str, Any], case["evaluation"])["result"]
        trajectories[case_id] = trajectory
        trajectory_paths[case_id] = trajectory_path
        clips_by_case[case_id] = _case_clips(case_id, trajectory, result, fps)
    clips = tuple(clip for case_id in _CASE_IDS for clip in clips_by_case[case_id])
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for the S110 video")
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        if qualification.body_hash != request.get("body_hash"):
            raise ValueError("bilateral population video Body hash changed")
        import mujoco

        with tempfile.TemporaryDirectory(prefix="rosclaw-s110-video-") as temp:
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
                raise RuntimeError("S110 raw-video pipe is unavailable")
            try:
                for case_id in _CASE_IDS:
                    case_config = cast(dict[str, Any], cases[case_id]["case"])
                    striker = cast(dict[str, Any], case_config["striker"])
                    goal = replace(base_goal, ball_mass_kg=float(case_config["ball_mass_kg"]))
                    model = build_g1_four_player_two_ball_stadium_model(
                        asset_root,
                        passer_origin_m=(5.10, -0.864060065039216, 0.0),
                        goalkeeper_origin_m=(goal.plane_x_m - 0.30, 0.0, 0.0),
                        second_striker_origin_m=tuple(striker["origin_m"]),
                        first_ball_origin_m=(1.0, 0.0, goal.ball_radius_m),
                        second_ball_origin_m=tuple(striker["ball_origin_m"]),
                        spec=goal,
                    )
                    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
                    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
                    data = mujoco.MjData(model)
                    renderer = mujoco.Renderer(model, height=height, width=width)
                    try:
                        _write_frames(
                            mujoco=mujoco,
                            model=model,
                            data=data,
                            renderer=renderer,
                            trajectory=trajectories[case_id],
                            clips=clips_by_case[case_id],
                            stream=cast(BinaryIO, process.stdin),
                        )
                    finally:
                        renderer.close()
            except BaseException:
                process.stdin.close()
                process.kill()
                process.wait()
                raise
            process.stdin.close()
            stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
            if process.wait():
                raise RuntimeError(f"S110 ffmpeg failed: {stderr[-3000:]}")
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
        raise RuntimeError("bilateral population encoded video contract changed")
    source_files = {
        str(evidence_file): hash_bytes(evidence_file.read_bytes()),
        str(request_path): hash_bytes(request_path.read_bytes()),
        str(goal_contract_file): hash_bytes(goal_contract_file.read_bytes()),
        **{str(path): hash_bytes(path.read_bytes()) for path in trajectory_paths.values()},
    }
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.bilateral_team_population_video.v1",
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_files": source_files,
        "claim": _CLAIM,
        "truth_label": "REJECTED_DEVELOPMENT",
        "evidence_report_hash": evidence["report_hash"],
        "evidence_passed": False,
        "strict_replay_all_cases": True,
        "observed_contact_feet": evidence["observed_contact_feet"],
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
    validate_bilateral_team_population_video(manifest_path)
    return manifest


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
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font_option = f"fontfile={escape_filtergraph_option(str(font))}:" if font.is_file() else ""
    scale = height / 720.0
    left = round(30 * scale)
    filters = [
        f"drawbox=x=0:y=0:w=iw:h={round(116 * scale)}:color=0x030711@0.84:t=fill",
        f"drawbox=x=0:y=h-{round(66 * scale)}:w=iw:h={round(66 * scale)}:"
        "color=0x5B1018@0.90:t=fill",
        f"drawtext={font_option}text='ROSClaw Soccer · S110 BILATERAL GROWTH':"
        f"expansion=none:x={left}:y={round(13 * scale)}:fontsize={round(32 * scale)}:"
        "fontcolor=white",
        f"drawtext={font_option}text='REJECTED DEVELOPMENT · COUNTEREXAMPLES FOR LEARNING · "
        f"NOT PROMOTED':expansion=none:x={left}:y=h-{round(43 * scale)}:"
        f"fontsize={round(20 * scale)}:fontcolor=0xFFD0D5",
    ]
    offset = 0.0
    for label, clip in zip(labels, clips, strict=True):
        end = offset + len(clip.frames) / fps
        filters.append(
            f"drawtext={font_option}textfile={escape_filtergraph_option(str(label))}:"
            f"expansion=none:x={left}:y={round(61 * scale)}:"
            f"fontsize={round(20 * scale)}:fontcolor=0xFFB36B:"
            f"enable='between(t,{offset:.6f},{end:.6f})'"
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


def main() -> None:
    args = _parser().parse_args()
    manifest = render_bilateral_team_population_video(
        evidence_path=args.evidence,
        goal_contract_path=args.goal_contract,
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
    "render_bilateral_team_population_video",
    "validate_bilateral_team_population_video",
]
