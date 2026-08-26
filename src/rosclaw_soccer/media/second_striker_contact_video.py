"""Evidence-downstream S108 fourth-G1 physical contact showcase."""

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
from numpy.typing import NDArray

from rosclaw_soccer.media.three_role_save_portfolio_video import (
    _Clip,
    _Frame,
    _lerp,
    _pose,
    _probe,
    _segment,
    _write_labels,
)
from rosclaw_soccer.media.trajectory_render import append_sphere, escape_filtergraph_option
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.second_striker_contact_exam import (
    validate_second_striker_contact_exam,
)
from rosclaw_soccer.world.field import (
    G1TrainingGoalSpec,
    build_g1_four_player_two_ball_stadium_model,
)

_CLAIM = "FOURTH_G1_PHYSICAL_SECOND_BALL_FOOT_CONTACT_ONLY"


def _implementation_hash() -> str:
    return str(hash_bytes(Path(__file__).read_bytes()))


def validate_second_striker_contact_video_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("second-striker contact video manifest must be an object")
    unhashed = dict(payload)
    claimed = unhashed.pop("manifest_hash", None)
    video_value = payload.get("video_path")
    sources = payload.get("source_files")
    if (
        claimed != hash_json(unhashed)
        or not isinstance(video_value, str)
        or not isinstance(sources, dict)
    ):
        raise ValueError("second-striker contact video manifest integrity mismatch")
    video = Path(video_value).expanduser().resolve()
    if not video.is_file() or hash_bytes(video.read_bytes()) != payload.get("video_hash"):
        raise ValueError("second-striker contact video bytes changed")
    for source_value, source_hash in sources.items():
        source = Path(source_value).expanduser().resolve()
        if not source.is_file() or hash_bytes(source.read_bytes()) != source_hash:
            raise ValueError("second-striker contact video source binding changed")
    numeric = tuple(
        payload.get(name) for name in ("fps", "width", "height", "frame_count", "duration_sec")
    )
    if not (
        payload.get("schema_version") == "rosclaw_soccer.second_striker_contact_video.v1"
        and payload.get("claim") == _CLAIM
        and payload.get("evidence_passed") is True
        and payload.get("strict_replay") is True
        and payload.get("four_g1_visible") is True
        and payload.get("two_physical_balls_visible") is True
        and payload.get("complete_second_save_claimed") is False
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
        raise ValueError("second-striker contact video authority contract is invalid")
    return cast(dict[str, Any], payload)


def render_second_striker_contact_video(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    fps: int = 60,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    evidence_file = evidence_path.expanduser().resolve()
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
        raise ValueError("second-striker contact video output contract is invalid")
    evidence = validate_second_striker_contact_exam(evidence_file)
    request_path = evidence_file.parent / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request_unhashed = dict(request)
    request_hash = request_unhashed.pop("request_hash", None)
    if request_hash != hash_json(request_unhashed) or request_hash != evidence["request_hash"]:
        raise ValueError("second-striker contact video request binding changed")
    trajectory_path = evidence_file.parent / str(evidence["trajectory_file"])
    if hash_bytes(trajectory_path.read_bytes()) != evidence["trajectory_hash"]:
        raise ValueError("second-striker contact video trajectory binding changed")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        trajectory = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "time",
        "first_ball_pose",
        "second_ball_pose",
        "source_shooter_pelvis_pose",
        "source_shooter_joint_position",
        "passer_pelvis_pose",
        "passer_joint_position",
        "goalkeeper_pelvis_pose",
        "goalkeeper_joint_position",
        "second_striker_pelvis_pose",
        "second_striker_joint_position",
    }
    if not required <= set(trajectory):
        raise ValueError("second-striker contact video trajectory is incomplete")
    result = cast(dict[str, Any], evidence["result"])
    contact_time = float(result["contact_time_sec"])
    clips = _timeline(trajectory, contact_time, fps)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for second-striker contact video")
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        if qualification.body_hash != evidence["body_hash"]:
            raise ValueError("second-striker contact video Body hash changed")
        import mujoco

        goal = G1TrainingGoalSpec(
            plane_x_m=7.5,
            width_m=7.32,
            height_m=2.44,
            target_y_m=0.45,
            target_z_m=1.35,
            regulation_field_enabled=True,
        )
        model = build_g1_four_player_two_ball_stadium_model(
            asset_root,
            passer_origin_m=(5.10, -3.0, 0.0),
            goalkeeper_origin_m=(7.02, 0.0, 0.0),
            second_striker_origin_m=(0.0, 0.0, 0.0),
            first_ball_origin_m=(3.895, -2.84, goal.ball_radius_m),
            second_ball_origin_m=(1.285, -0.018, goal.ball_radius_m),
            spec=goal,
        )
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-second-striker-contact-") as temp:
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
                    raise RuntimeError("second-striker contact raw-video pipe is unavailable")
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
                    raise RuntimeError(f"second-striker contact ffmpeg failed: {stderr[-3000:]}")
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
        raise RuntimeError("second-striker contact encoded video contract changed")
    source_files = {
        str(evidence_file): hash_bytes(evidence_file.read_bytes()),
        str(request_path): hash_bytes(request_path.read_bytes()),
        str(trajectory_path): hash_bytes(trajectory_path.read_bytes()),
    }
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.second_striker_contact_video.v1",
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_files": source_files,
        "claim": _CLAIM,
        "evidence_report_hash": evidence["report_hash"],
        "evidence_passed": True,
        "strict_replay": evidence["strict_replay"],
        "contact_time_sec": contact_time,
        "contact_force_peak_n": result["contact_force_peak_n"],
        "postcontact_peak_ball_speed_mps": result["postcontact_peak_ball_speed_mps"],
        "four_g1_visible": True,
        "two_physical_balls_visible": True,
        "complete_second_save_claimed": False,
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
    validate_second_striker_contact_video_manifest(manifest_path)
    return manifest


def _timeline(
    trajectory: dict[str, np.ndarray], contact_time: float, fps: int
) -> tuple[_Clip, ...]:
    start = float(trajectory["time"][0])
    end = float(trajectory["time"][-1])
    title = tuple(_Frame("s108", start, "four") for _ in range(round(1.8 * fps)))
    final = tuple(_Frame("s108", end, "goal") for _ in range(round(2.0 * fps)))
    return (
        _Clip("S108 · FOUR G1 · TWO PHYSICAL FOOTBALLS · ZERO TELEPORT", title),
        _Clip(
            "FOURTH G1 · FROZEN ROBONALDO ONNX APPROACH",
            _segment("s108", 1.6, contact_time - 0.32, 1.45, "striker", fps),
        ),
        _Clip(
            "ANATOMICAL RIGHT-FOOT CONTACT · 502 N PEAK",
            _segment("s108", contact_time - 0.48, contact_time + 0.42, 0.24, "contact", fps),
        ),
        _Clip(
            "PHYSICAL BALL SPEED 0.001 → 7.098 m/s",
            _segment("s108", contact_time - 0.18, end, 0.58, "flight", fps),
        ),
        _Clip(
            "STRICT-REPLAY CONTACT CHAIN · NO UNEXPECTED PRE-CONTACT COLLISION",
            _segment("s108", start, end, 1.75, "four", fps),
        ),
        _Clip("CONTACT INTERFACE PROMOTED · COMPLETE SECOND SAVE IS NEXT", final),
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
            _add_second_ball_trail(mujoco, renderer.scene, trajectory, int(sample["index"]))
            stream.write(np.ascontiguousarray(renderer.render().copy()).tobytes())


def _sample(
    trajectory: dict[str, np.ndarray], simulation_time_sec: float
) -> dict[str, NDArray[np.float64] | int]:
    time = np.asarray(trajectory["time"], dtype=np.float64)
    upper = int(np.searchsorted(time, simulation_time_sec, side="right"))
    if upper <= 0:
        lower = upper = 0
        ratio = 0.0
    elif upper >= time.size:
        lower = upper = time.size - 1
        ratio = 0.0
    else:
        lower = upper - 1
        ratio = float((simulation_time_sec - time[lower]) / (time[upper] - time[lower]))
    output: dict[str, NDArray[np.float64] | int] = {"index": upper if ratio >= 0.5 else lower}
    for role in ("source_shooter", "passer", "goalkeeper", "second_striker"):
        output[f"{role}_pelvis_pose"] = _pose(
            trajectory[f"{role}_pelvis_pose"][lower],
            trajectory[f"{role}_pelvis_pose"][upper],
            ratio,
        )
        output[f"{role}_joint_position"] = _lerp(
            trajectory[f"{role}_joint_position"][lower],
            trajectory[f"{role}_joint_position"][upper],
            ratio,
        )
    for ball in ("first_ball", "second_ball"):
        output[f"{ball}_pose"] = _pose(
            trajectory[f"{ball}_pose"][lower], trajectory[f"{ball}_pose"][upper], ratio
        )
    return output


def _set_camera(camera: Any, view: str, poses: dict[str, NDArray[np.float64] | int]) -> None:
    ball = cast(NDArray[np.float64], poses["second_ball_pose"])
    striker = cast(NDArray[np.float64], poses["second_striker_pelvis_pose"])
    if view == "striker":
        camera.lookat[:] = (float(striker[0] + 0.55), float(striker[1]), 0.70)
        camera.distance, camera.azimuth, camera.elevation = 4.7, 112.0, -7.0
    elif view == "contact":
        camera.lookat[:] = 0.52 * striker[:3] + 0.48 * ball[:3]
        camera.lookat[2] = 0.48
        camera.distance, camera.azimuth, camera.elevation = 3.1, 104.0, -5.0
    elif view == "flight":
        camera.lookat[:] = (float(ball[0]), float(ball[1]), max(0.65, float(ball[2])))
        camera.distance, camera.azimuth, camera.elevation = 6.8, 112.0, -7.0
    elif view == "goal":
        camera.lookat[:] = (6.4, float(ball[1]), 0.95)
        camera.distance, camera.azimuth, camera.elevation = 7.3, 155.0, -6.0
    else:
        camera.lookat[:] = (3.65, -0.25, 0.72)
        camera.distance, camera.azimuth, camera.elevation = 11.8, 92.0, -10.0


def _add_second_ball_trail(
    mujoco: Any, scene: Any, trajectory: dict[str, np.ndarray], index: int
) -> None:
    start = max(0, index - 42)
    indices = np.linspace(start, index, min(18, index - start + 1), dtype=int)
    for trail_index, alpha in zip(indices, np.linspace(0.02, 0.42, len(indices)), strict=True):
        append_sphere(
            mujoco,
            scene,
            np.asarray(trajectory["second_ball_pose"][trail_index, :3]),
            0.022,
            (0.20, 0.82, 1.0, float(alpha)),
        )


def _id(mujoco: Any, model: Any, object_type: Any, name: str) -> int:
    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise ValueError(f"second-striker contact video model is missing {name}")
    return value


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
        f"drawtext={font_option}text='ROSClaw Soccer · S108 FOURTH G1 CONTACT':"
        f"expansion=none:x={left}:y={round(13 * scale)}:fontsize={round(32 * scale)}:"
        "fontcolor=white",
        f"drawtext={font_option}text='STRICT CPU MUJOCO · FOUR G1 · TWO BALLS · "
        f"CONTACT QUALIFICATION ONLY':expansion=none:x={left}:y=h-{round(40 * scale)}:"
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
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = render_second_striker_contact_video(
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
    "render_second_striker_contact_video",
    "validate_second_striker_contact_video_manifest",
]
