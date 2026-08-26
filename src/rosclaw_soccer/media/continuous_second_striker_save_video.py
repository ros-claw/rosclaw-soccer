"""Evidence-bound S109 four-G1, two-ball, two-save development reel."""

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

from rosclaw_soccer.media.second_striker_contact_video import (
    _add_second_ball_trail,
    _id,
    _sample,
)
from rosclaw_soccer.media.three_role_save_portfolio_video import (
    _Clip,
    _Frame,
    _probe,
    _segment,
    _write_labels,
)
from rosclaw_soccer.media.trajectory_render import escape_filtergraph_option
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.continuous_second_striker_save_exam import (
    validate_continuous_second_striker_save_exam,
)
from rosclaw_soccer.world.field import (
    G1TrainingGoalSpec,
    build_g1_four_player_two_ball_stadium_model,
)

_CLAIM = "CONTINUOUS_FOUR_G1_SECOND_STRIKER_HIGH_GLOVE_SAVE_AND_READY"


def _implementation_hash() -> str:
    return str(hash_bytes(Path(__file__).read_bytes()))


def validate_continuous_second_striker_save_video_manifest(
    path: Path,
) -> dict[str, Any]:
    """Validate video/source bytes and non-promotion boundaries."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("continuous second-striker video manifest must be an object")
    unhashed = dict(payload)
    claimed = unhashed.pop("manifest_hash", None)
    video_value = payload.get("video_path")
    sources = payload.get("source_files")
    if (
        claimed != hash_json(unhashed)
        or not isinstance(video_value, str)
        or not isinstance(sources, dict)
    ):
        raise ValueError("continuous second-striker video manifest integrity mismatch")
    video = Path(video_value).expanduser().resolve()
    if not video.is_file() or hash_bytes(video.read_bytes()) != payload.get("video_hash"):
        raise ValueError("continuous second-striker video bytes changed")
    for source_value, source_hash in sources.items():
        source = Path(source_value).expanduser().resolve()
        if not source.is_file() or hash_bytes(source.read_bytes()) != source_hash:
            raise ValueError("continuous second-striker video source binding changed")
    numeric = tuple(
        payload.get(name) for name in ("fps", "width", "height", "frame_count", "duration_sec")
    )
    if not (
        payload.get("schema_version") == "rosclaw_soccer.continuous_second_striker_save_video.v1"
        and payload.get("claim") == _CLAIM
        and payload.get("evidence_passed") is True
        and payload.get("strict_replay") is True
        and payload.get("four_g1_visible") is True
        and payload.get("two_physical_balls_visible") is True
        and payload.get("ball_cannon_used") is False
        and payload.get("reset_or_teleport_used") is False
        and payload.get("visualization_only") is True
        and payload.get("pixels_used_for_scoring") is False
        and payload.get("promotion_eligible") is False
        and payload.get("activation_ceiling") == "SIM_ONLY"
        and payload.get("hardware_command_sent") is False
        and payload.get("commercial_use_allowed") is False
        and payload.get("implementation_hash") == _implementation_hash()
        and all(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
            for value in numeric
        )
    ):
        raise ValueError("continuous second-striker video authority contract is invalid")
    return cast(dict[str, Any], payload)


def render_continuous_second_striker_save_video(
    *,
    evidence_path: Path,
    goal_contract_path: Path,
    asset_root: Path,
    output_path: Path,
    fps: int = 60,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render the scored S109 trajectory; pixels never participate in gates."""

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
        raise ValueError("continuous second-striker video output contract is invalid")
    evidence = validate_continuous_second_striker_save_exam(evidence_file)
    cases = cast(dict[str, dict[str, Any]], evidence["cases"])
    if len(cases) != 1:
        raise ValueError("continuous second-striker reel requires one qualified lane")
    lane_id, case = next(iter(cases.items()))
    evaluation = cast(dict[str, Any], case["evaluation"])
    result = cast(dict[str, Any], evaluation["result"])
    request_path = evidence_file.parent / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request_unhashed = dict(request)
    request_hash = request_unhashed.pop("request_hash", None)
    if request_hash != hash_json(request_unhashed) or request_hash != evidence["request_hash"]:
        raise ValueError("continuous second-striker video request binding changed")
    goal_contract = json.loads(goal_contract_file.read_text(encoding="utf-8"))
    goal_specs = goal_contract.get("lane_goal_specs")
    if (
        not isinstance(goal_specs, dict)
        or lane_id not in goal_specs
        or goal_contract.get("body_hash") != request.get("body_hash")
    ):
        raise ValueError("continuous second-striker goal contract changed")
    goal = G1TrainingGoalSpec(**goal_specs[lane_id])
    if abs(goal.width_m - 7.32) > 1.0e-9 or abs(goal.height_m - 2.44) > 1.0e-9:
        raise ValueError("continuous second-striker video requires a regulation goal")
    trajectory_name = case.get("trajectory_file")
    if not isinstance(trajectory_name, str) or Path(trajectory_name).name != trajectory_name:
        raise ValueError("continuous second-striker trajectory name is invalid")
    trajectory_path = evidence_file.parent / trajectory_name
    if hash_bytes(trajectory_path.read_bytes()) != case.get("trajectory_hash"):
        raise ValueError("continuous second-striker trajectory binding changed")
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
        raise ValueError("continuous second-striker video trajectory is incomplete")
    trajectory = dict(raw)
    trajectory["first_ball_pose"] = raw["ball_pose"]
    trajectory["source_shooter_pelvis_pose"] = raw["shooter_pelvis_pose"]
    trajectory["source_shooter_joint_position"] = raw["shooter_joint_position"]
    clips = _timeline(lane_id, trajectory, result, fps)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for the S109 video")
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        if qualification.body_hash != request.get("body_hash"):
            raise ValueError("continuous second-striker video Body hash changed")
        import mujoco

        striker_config = cast(dict[str, Any], request["config"])["striker"]
        model = build_g1_four_player_two_ball_stadium_model(
            asset_root,
            passer_origin_m=(5.10, -0.864060065039216, 0.0),
            goalkeeper_origin_m=(goal.plane_x_m - 0.30, 0.0, 0.0),
            second_striker_origin_m=tuple(striker_config["origin_m"]),
            first_ball_origin_m=(1.0, 0.0, goal.ball_radius_m),
            second_ball_origin_m=tuple(striker_config["ball_origin_m"]),
            spec=goal,
        )
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-s109-video-") as temp:
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
                    raise RuntimeError("S109 raw-video pipe is unavailable")
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
                    raise RuntimeError(f"S109 ffmpeg failed: {stderr[-3000:]}")
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
        raise RuntimeError("continuous second-striker encoded video contract changed")
    source_files = {
        str(evidence_file): hash_bytes(evidence_file.read_bytes()),
        str(request_path): hash_bytes(request_path.read_bytes()),
        str(goal_contract_file): hash_bytes(goal_contract_file.read_bytes()),
        str(trajectory_path): hash_bytes(trajectory_path.read_bytes()),
    }
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.continuous_second_striker_save_video.v1",
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_files": source_files,
        "claim": _CLAIM,
        "evidence_report_hash": evidence["report_hash"],
        "evidence_passed": True,
        "strict_replay": case["strict_replay"],
        "four_g1_visible": True,
        "two_physical_balls_visible": True,
        "ball_cannon_used": False,
        "reset_or_teleport_used": False,
        "second_striker_contact_time_sec": result["second_striker_contact_time_sec"],
        "second_striker_contact_force_peak_n": result["second_striker_contact_force_peak_n"],
        "second_glove_contact_time_sec": result["goalkeeper_second_glove_contact_time_sec"],
        "second_glove_contact_height_m": result["goalkeeper_second_glove_contact_height_m"],
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
    validate_continuous_second_striker_save_video_manifest(manifest_path)
    return manifest


def _timeline(
    lane_id: str,
    trajectory: dict[str, np.ndarray],
    result: dict[str, Any],
    fps: int,
) -> tuple[_Clip, ...]:
    start = float(trajectory["time"][0])
    end = float(trajectory["time"][-1])
    first = float(result["goalkeeper_glove_contact_time_sec"])
    rearm = float(result["second_threat_rearm_time_sec"])
    strike = float(result["second_striker_contact_time_sec"])
    second = float(result["goalkeeper_second_glove_contact_time_sec"])
    force = float(result["second_striker_contact_force_peak_n"])
    height = float(result["goalkeeper_second_glove_contact_height_m"])
    title = tuple(_Frame(lane_id, start, "four") for _ in range(round(2.0 * fps)))
    final = tuple(_Frame(lane_id, end, "goal") for _ in range(round(2.2 * fps)))
    return (
        _Clip("S109 · FOUR G1 · TWO FOOTBALLS · TWO REAL SAVES", title),
        _Clip(
            "PASS → FIRST G1 HIGH SHOT → FIRST AIRBORNE SAVE",
            _segment(lane_id, 4.4, first + 0.45, 1.0, "four", fps),
        ),
        _Clip(
            "FIRST COLLISION-FAITHFUL GLOVE SAVE · SLOW MOTION",
            _segment(lane_id, first - 0.50, first + 0.55, 0.34, "goal", fps),
        ),
        _Clip(
            "MEASURED RECOVERY → READY → SECOND THREAT REARM",
            _segment(lane_id, first + 0.55, rearm + 0.10, 2.25, "goal", fps),
        ),
        _Clip(
            "FOURTH G1 · LEARNED ACTOR + UPPER-CORNER MUSCLE MEMORY",
            _segment(lane_id, 12.0, strike - 0.28, 1.35, "striker", fps),
        ),
        _Clip(
            f"ANATOMICAL RIGHT-FOOT STRIKE · {force:.0f} N · NO CANNON",
            _segment(lane_id, strike - 0.42, second + 0.30, 0.42, "contact", fps),
        ),
        _Clip(
            f"SECOND HIGH GLOVE SAVE AT {height:.3f} m · BALL REVERSED OUT",
            _segment(lane_id, second - 0.42, second + 0.78, 0.31, "goal", fps),
        ),
        _Clip(
            "SECOND SAVE → DOUBLE SUPPORT → READY AGAIN",
            _segment(lane_id, second + 0.55, 23.5, 1.45, "goal", fps),
        ),
        _Clip(
            "ONE CLOCK · STRICT REPLAY · ZERO RESET OR TELEPORT",
            _segment(lane_id, start, end, 2.0, "four", fps),
        ),
        _Clip("ROSCLAW SELF-GROWTH · COMPLETE PHYSICAL SUCCESSOR", final),
    )


def _write_frames(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    trajectory: dict[str, np.ndarray],
    clips: tuple[_Clip, ...],
    stream: BinaryIO,
) -> None:
    roles = (
        ("source_shooter", ""),
        ("passer", "passer_"),
        ("goalkeeper", "goalkeeper_"),
        ("second_striker", "second_striker_"),
    )
    free_qpos = {
        role: int(
            model.jnt_qposadr[
                _id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint")
            ]
        )
        for role, prefix in roles
    }
    joint_qpos = {
        role: np.asarray(
            [
                model.jnt_qposadr[_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)]
                for name in G1_DDS_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        for role, prefix in roles
    }
    ball_qpos = {
        "first_ball": int(model.jnt_qposadr[int(model.joint("ball_free").id)]),
        "second_ball": int(model.jnt_qposadr[int(model.joint("second_ball_free").id)]),
    }
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    for clip in clips:
        for frame in clip.frames:
            sample = _sample(trajectory, frame.simulation_time_sec)
            data.qpos[:] = model.qpos0
            for role, _ in roles:
                data.qpos[free_qpos[role] : free_qpos[role] + 7] = sample[f"{role}_pelvis_pose"]
                data.qpos[joint_qpos[role]] = sample[f"{role}_joint_position"]
            for ball in ("first_ball", "second_ball"):
                data.qpos[ball_qpos[ball] : ball_qpos[ball] + 7] = sample[f"{ball}_pose"]
            mujoco.mj_forward(model, data)
            _set_camera(camera, frame.view, sample)
            renderer.update_scene(data, camera=camera)
            _add_second_ball_trail(
                mujoco,
                renderer.scene,
                trajectory,
                int(sample["index"]),
            )
            stream.write(np.ascontiguousarray(renderer.render().copy()).tobytes())


def _set_camera(camera: Any, view: str, poses: dict[str, Any]) -> None:
    ball = np.asarray(poses["second_ball_pose"], dtype=np.float64)
    striker = np.asarray(poses["second_striker_pelvis_pose"], dtype=np.float64)
    if view == "striker":
        camera.lookat[:] = (float(striker[0] + 0.55), float(striker[1]), 0.72)
        camera.distance, camera.azimuth, camera.elevation = 4.8, -72.0, -7.0
    elif view == "contact":
        camera.lookat[:] = 0.52 * striker[:3] + 0.48 * ball[:3]
        camera.lookat[2] = 0.50
        camera.distance, camera.azimuth, camera.elevation = 3.3, -67.0, -5.0
    elif view == "goal":
        camera.lookat[:] = (6.4, float(ball[1]), 0.95)
        camera.distance, camera.azimuth, camera.elevation = 7.3, 155.0, -6.0
    else:
        camera.lookat[:] = (3.65, -0.25, 0.72)
        camera.distance, camera.azimuth, camera.elevation = 11.8, 92.0, -10.0


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
        f"drawbox=x=0:y=0:w=iw:h={round(116 * scale)}:color=0x030711@0.82:t=fill",
        f"drawbox=x=0:y=h-{round(62 * scale)}:w=iw:h={round(62 * scale)}:"
        "color=0x030711@0.82:t=fill",
        f"drawtext={font_option}text='ROSClaw Soccer · S109 PHYSICAL SECOND STRIKER':"
        f"expansion=none:x={left}:y={round(13 * scale)}:fontsize={round(32 * scale)}:"
        "fontcolor=white",
        f"drawtext={font_option}text='STRICT CPU MUJOCO · FOUR G1 · TWO BALLS · "
        f"NO CANNON · NO PIXEL SCORING':expansion=none:x={left}:y=h-{round(40 * scale)}:"
        f"fontsize={round(18 * scale)}:fontcolor=0x8DD8FF",
    ]
    offset = 0.0
    for label, clip in zip(labels, clips, strict=True):
        end = offset + len(clip.frames) / fps
        filters.append(
            f"drawtext={font_option}textfile={escape_filtergraph_option(str(label))}:"
            f"expansion=none:x={left}:y={round(61 * scale)}:"
            f"fontsize={round(20 * scale)}:fontcolor=0x65F59A:"
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
    manifest = render_continuous_second_striker_save_video(
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
    "render_continuous_second_striker_save_video",
    "validate_continuous_second_striker_save_video_manifest",
]
