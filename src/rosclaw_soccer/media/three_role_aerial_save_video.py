"""Evidence-downstream promo video for the strict three-G1 aerial save."""

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
    simulation_time_sec: float
    view: str


@dataclass(frozen=True)
class _Clip:
    label: str
    frames: tuple[_Frame, ...]


def render_three_role_aerial_save_video(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render one frozen trajectory; video pixels never affect qualification."""

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
        raise ValueError("three-role aerial-save video output contract is invalid")
    evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict) or not (
        evidence.get("passed") is True
        and evidence.get("strict_replay") is True
        and evidence.get("promotion_status") == "FROZEN_SIM_DEMO"
        and evidence.get("physics_authority") == "CPU_MUJOCO"
        and evidence.get("activation_ceiling") == "SIM_ONLY"
        and evidence.get("hardware_command_sent") is False
        and evidence.get("pixels_used_for_scoring") is False
    ):
        raise ValueError("three-role aerial-save evidence is not render eligible")
    replay = evidence.get("replay")
    if not isinstance(replay, dict) or replay.get("passed") is not True:
        raise ValueError("three-role aerial-save replay did not pass")
    gates = replay.get("gates")
    if not isinstance(gates, dict) or not all(gates.values()):
        raise ValueError("three-role aerial-save replay gates are incomplete")
    request_path = evidence_file.parent / "request.json"
    trajectory_path = evidence_file.parent / "trajectory.npz"
    if hash_bytes(request_path.read_bytes()) != evidence.get("request_hash") or hash_bytes(
        trajectory_path.read_bytes()
    ) != evidence.get("trajectory_hash"):
        raise ValueError("three-role aerial-save evidence bindings changed")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    goal_value = request.get("goal_spec")
    if not isinstance(goal_value, dict):
        raise ValueError("three-role aerial-save goal contract is missing")
    goal = G1TrainingGoalSpec(**goal_value)
    if abs(goal.width_m - 7.32) > 1e-9 or abs(goal.height_m - 2.44) > 1e-9:
        raise ValueError("three-role aerial-save promo requires the regulation goal")
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
    }
    if not required <= set(trajectory):
        raise ValueError("three-role aerial-save trajectory is incomplete")
    result = replay.get("result")
    if not isinstance(result, dict):
        raise ValueError("three-role aerial-save result is missing")
    clips = _timeline(replay, trajectory, fps)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for aerial-save video")
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        if qualification.body_hash != request.get("body_hash"):
            raise ValueError("three-role aerial-save Body hash changed")
        import mujoco

        goalkeeper_depth = float(request["config"]["goalkeeper_depth_from_goal_line_m"])
        model = build_g1_three_player_stadium_model(
            asset_root.expanduser().resolve(),
            passer_origin_m=(5.10, -0.16406006503921598, 0.0),
            goalkeeper_origin_m=(goal.plane_x_m - goalkeeper_depth, 0.0, 0.0),
            spec=goal,
        )
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-three-g1-aerial-save-") as temp:
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
                    raise RuntimeError("aerial-save ffmpeg raw-video pipe is unavailable")
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
                    raise RuntimeError(f"aerial-save ffmpeg failed: {stderr[-3000:]}")
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
        raise RuntimeError("aerial-save encoded video contract changed")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.three_role_aerial_save_video.v1",
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_evidence_path": str(evidence_file),
        "source_evidence_hash": hash_bytes(evidence_file.read_bytes()),
        "request_hash": evidence["request_hash"],
        "trajectory_hash": evidence["trajectory_hash"],
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps,
        "goal_contract": "REGULATION_7.32X2.44M_GOAL",
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
    return manifest


def _timeline(
    replay: dict[str, Any], trajectory: dict[str, np.ndarray], fps: int
) -> tuple[_Clip, ...]:
    result = replay.get("result")
    if not isinstance(result, dict):
        raise ValueError("three-role aerial-save replay result is missing")
    start = float(trajectory["time"][0])
    end = float(trajectory["time"][-1])
    pass_time = float(result["pass_contact_time_sec"])
    shot_time = float(result["shot_contact_time_sec"])
    save_time = float(result["goalkeeper_glove_contact_time_sec"])
    pass_error_mm = 1_000.0 * float(result["pass_delivery_error_m"])
    incoming_speed = float(replay["incoming_speed_mps"])
    apex = float(replay["aerial_apex_m"])
    glove_height = float(result["goalkeeper_glove_contact_height_m"])
    intro = tuple(_Frame(max(start, pass_time - 1.0), "wide") for _ in range(round(1.4 * fps)))
    continuous = (
        *_segment(max(start, pass_time - 1.0), pass_time + 0.45, 1.0, "pass", fps),
        *_segment(pass_time + 0.45, shot_time - 0.45, 1.0, "roll", fps),
        *_segment(shot_time - 0.45, save_time + 1.0, 1.0, "hero", fps),
        *_segment(save_time + 1.0, end, 1.0, "wide_goal", fps),
    )
    slow_chain = (
        *_segment(pass_time - 0.45, shot_time + 0.25, 0.52, "pass_close", fps),
        *_segment(shot_time + 0.25, save_time + 0.70, 0.38, "hero", fps),
    )
    glove_replay = _segment(shot_time - 0.35, save_time + 0.60, 0.32, "goal_front", fps)
    finale = tuple(_Frame(end, "wide_goal") for _ in range(round(1.8 * fps)))
    return (
        _Clip("THREE G1 · ONE BALL · ONE CONTINUOUS WORLD", intro),
        _Clip(
            f"{pass_error_mm:.1f} mm PASS → HIGH STRIKE → REAL GLOVE SAVE",
            continuous,
        ),
        _Clip(
            f"SLOW REPLAY · {incoming_speed:.2f} m/s SHOT · {apex:.2f} m APEX",
            slow_chain,
        ),
        _Clip(
            f"BOTH HANDS UP · GLOVE CONTACT {glove_height:.3f} m · NO GOAL",
            glove_replay,
        ),
        _Clip("STRICT CPU MUJOCO REPLAY PASS · ALL THREE STABLE", finale),
    )


def _segment(start: float, end: float, speed: float, view: str, fps: int) -> tuple[_Frame, ...]:
    if end <= start or speed <= 0.0:
        raise ValueError("aerial-save video segment is invalid")
    count = max(1, int(math.ceil((end - start) / speed * fps)))
    return tuple(_Frame(min(end, start + index / fps * speed), view) for index in range(count))


def _write_frames(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    trajectory: dict[str, np.ndarray],
    clips: tuple[_Clip, ...],
    goal: G1TrainingGoalSpec,
    stream: BinaryIO,
) -> None:
    ball_body = _id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_joint = int(model.body_jntadr[ball_body])
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    free_qpos = {"shooter": 0}
    for role, prefix in (("passer", "passer_"), ("goalkeeper", "goalkeeper_")):
        joint = _id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint")
        free_qpos[role] = int(model.jnt_qposadr[joint])
    joint_qpos = {
        role: _joint_qpos(mujoco, model, prefix)
        for role, prefix in (("shooter", ""), ("passer", "passer_"), ("goalkeeper", "goalkeeper_"))
    }
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    for clip in clips:
        for frame in clip.frames:
            poses = _sample(trajectory, frame.simulation_time_sec)
            data.qpos[:] = model.qpos0
            for role in ("shooter", "passer", "goalkeeper"):
                data.qpos[free_qpos[role] : free_qpos[role] + 7] = poses[f"{role}_pelvis_pose"]
                data.qpos[joint_qpos[role]] = poses[f"{role}_joint_position"]
            data.qpos[ball_qpos : ball_qpos + 7] = poses["ball_pose"]
            mujoco.mj_forward(model, data)
            _set_camera(camera, frame.view, poses, goal)
            renderer.update_scene(data, camera=camera)
            _add_ball_trail(
                mujoco,
                renderer.scene,
                trajectory,
                int(poses["index"]),
            )
            stream.write(np.ascontiguousarray(renderer.render().copy()).tobytes())


def _set_camera(
    camera: Any, view: str, poses: dict[str, NDArray[np.float64] | int], goal: G1TrainingGoalSpec
) -> None:
    ball = cast(NDArray[np.float64], poses["ball_pose"])
    passer = cast(NDArray[np.float64], poses["passer_pelvis_pose"])
    if view == "wide":
        camera.lookat[:] = (3.65, 0.0, 0.72)
        camera.distance, camera.azimuth, camera.elevation = 10.8, 91.0, -10.0
    elif view == "pass":
        camera.lookat[:] = (3.10, -0.05, 0.70)
        camera.distance, camera.azimuth, camera.elevation = 7.7, 100.0, -8.0
    elif view == "pass_close":
        camera.lookat[:] = 0.55 * passer[:3] + 0.45 * ball[:3]
        camera.lookat[2] = 0.66
        camera.distance, camera.azimuth, camera.elevation = 4.7, 116.0, -7.0
    elif view == "roll":
        camera.lookat[:] = ball[:3] + np.asarray((0.0, 0.0, 0.42))
        camera.distance, camera.azimuth, camera.elevation = 5.1, 101.0, -6.0
    elif view == "goal_front":
        camera.lookat[:] = (goal.plane_x_m - 0.30, 0.35, 1.10)
        camera.distance, camera.azimuth, camera.elevation = 6.4, 178.0, -4.0
    elif view == "hero":
        camera.lookat[:] = (goal.plane_x_m - 1.00, 0.30, 0.95)
        camera.distance, camera.azimuth, camera.elevation = 6.6, 142.0, -6.0
    else:
        camera.lookat[:] = (goal.plane_x_m - 1.25, 0.0, 0.85)
        camera.distance, camera.azimuth, camera.elevation = 8.4, 118.0, -8.0


def _add_ball_trail(mujoco: Any, scene: Any, trajectory: dict[str, np.ndarray], index: int) -> None:
    start = max(0, index - 45)
    indices = np.linspace(start, index, min(18, index - start + 1), dtype=int)
    for trail_index, alpha in zip(indices, np.linspace(0.02, 0.42, len(indices)), strict=True):
        append_sphere(
            mujoco,
            scene,
            np.asarray(trajectory["ball_pose"][trail_index, :3]),
            0.022,
            (0.20, 0.82, 1.0, float(alpha)),
        )


def _sample(
    trajectory: dict[str, np.ndarray], simulation_time_sec: float
) -> dict[str, NDArray[np.float64] | int]:
    time = np.asarray(trajectory["time"], dtype=np.float64)
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
    value: dict[str, NDArray[np.float64] | int] = {"index": upper if ratio >= 0.5 else lower}
    for role in ("passer", "shooter", "goalkeeper"):
        value[f"{role}_pelvis_pose"] = _pose(
            trajectory[f"{role}_pelvis_pose"][lower],
            trajectory[f"{role}_pelvis_pose"][upper],
            ratio,
        )
        value[f"{role}_joint_position"] = _lerp(
            trajectory[f"{role}_joint_position"][lower],
            trajectory[f"{role}_joint_position"][upper],
            ratio,
        )
    value["ball_pose"] = _pose(
        trajectory["ball_pose"][lower], trajectory["ball_pose"][upper], ratio
    )
    return value


def _pose(left: np.ndarray, right: np.ndarray, ratio: float) -> NDArray[np.float64]:
    result: NDArray[np.float64] = np.empty(7, dtype=np.float64)
    result[:3] = _lerp(left[:3], right[:3], ratio)
    result[3:] = _slerp(left[3:], right[3:], ratio)
    return result


def _lerp(left: np.ndarray, right: np.ndarray, ratio: float) -> NDArray[np.float64]:
    start = np.asarray(left, dtype=np.float64)
    return np.asarray(start + ratio * (np.asarray(right, dtype=np.float64) - start))


def _slerp(left: np.ndarray, right: np.ndarray, ratio: float) -> NDArray[np.float64]:
    start = np.asarray(left, dtype=np.float64)
    end = np.asarray(right, dtype=np.float64)
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
        np.sin((1.0 - ratio) * angle) / scale * start + np.sin(ratio * angle) / scale * end,
        dtype=np.float64,
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
        raise ValueError(f"aerial-save video model is missing {name}")
    return value


def _write_labels(root: Path, clips: tuple[_Clip, ...]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for index, clip in enumerate(clips):
        path = root / f"label-{index}.txt"
        path.write_text(clip.label, encoding="utf-8")
        paths.append(path)
    return tuple(paths)


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
        f"drawtext={font_option}text='ROSClaw Soccer · THREE G1 AERIAL SAVE':"
        f"expansion=none:x={left}:y={round(13 * scale)}:fontsize={round(32 * scale)}:"
        "fontcolor=white",
        f"drawtext={font_option}text='STRICT CPU MUJOCO · 7.32×2.44 m GOAL · "
        "SIM ONLY · NO PIXEL SCORING':"
        f"expansion=none:x={left}:y=h-{round(40 * scale)}:fontsize={round(18 * scale)}:"
        "fontcolor=0x8DD8FF",
    ]
    offset = 0.0
    for label, clip in zip(labels, clips, strict=True):
        end = offset + len(clip.frames) / fps
        filters.append(
            f"drawtext={font_option}textfile={escape_filtergraph_option(str(label))}:"
            f"expansion=none:x={left}:y={round(61 * scale)}:fontsize={round(20 * scale)}:"
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
    numerator, denominator = (int(value) for value in stream["avg_frame_rate"].split("/"))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": round(numerator / denominator),
        "frame_count": int(stream["nb_read_frames"]),
    }


__all__ = ["render_three_role_aerial_save_video"]
