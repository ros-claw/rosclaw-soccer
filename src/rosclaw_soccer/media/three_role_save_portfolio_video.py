"""Evidence-downstream multi-lane three-G1 goalkeeper promo video."""

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


def render_three_role_save_portfolio_video(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render all frozen lanes without granting pixels scoring authority."""

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
        raise ValueError("three-role save-portfolio video output contract is invalid")
    evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
    portfolio_gates = evidence.get("portfolio_gates")
    cases_value = evidence.get("cases")
    if not isinstance(evidence, dict) or not (
        evidence.get("passed") is True
        and evidence.get("promotion_status") == "FROZEN_SIM_DEMO"
        and evidence.get("physics_authority") == "CPU_MUJOCO"
        and evidence.get("activation_ceiling") == "SIM_ONLY"
        and evidence.get("hardware_command_sent") is False
        and evidence.get("pixels_used_for_scoring") is False
        and isinstance(portfolio_gates, dict)
        and all(portfolio_gates.values())
        and isinstance(cases_value, dict)
        and len(cases_value) >= 3
    ):
        raise ValueError("three-role save-portfolio evidence is not render eligible")
    request_path = evidence_file.parent / "request.json"
    if hash_bytes(request_path.read_bytes()) != evidence.get("request_hash"):
        raise ValueError("three-role save-portfolio request binding changed")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    goal_specs = request.get("lane_goal_specs")
    if not isinstance(goal_specs, dict) or set(goal_specs) != set(cases_value):
        raise ValueError("three-role save-portfolio goal contracts are incomplete")
    trajectories: dict[str, dict[str, np.ndarray]] = {}
    cases: dict[str, dict[str, Any]] = {}
    goals: dict[str, G1TrainingGoalSpec] = {}
    trajectory_hashes: dict[str, str] = {}
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
    for lane_id, case_value in cases_value.items():
        if not isinstance(lane_id, str) or not isinstance(case_value, dict) or not (
            case_value.get("passed") is True and case_value.get("strict_replay") is True
        ):
            raise ValueError("three-role save-portfolio case did not strictly pass")
        replay = case_value.get("replay")
        if not isinstance(replay, dict) or replay.get("passed") is not True:
            raise ValueError("three-role save-portfolio replay is invalid")
        gates = replay.get("gates")
        if not isinstance(gates, dict) or not all(gates.values()):
            raise ValueError("three-role save-portfolio replay gates are incomplete")
        trajectory_name = case_value.get("trajectory_file")
        if not isinstance(trajectory_name, str) or Path(trajectory_name).name != trajectory_name:
            raise ValueError("three-role save-portfolio trajectory name is invalid")
        trajectory_path = evidence_file.parent / trajectory_name
        trajectory_hash = hash_bytes(trajectory_path.read_bytes())
        if trajectory_hash != case_value.get("trajectory_hash"):
            raise ValueError("three-role save-portfolio trajectory binding changed")
        with np.load(trajectory_path, allow_pickle=False) as archive:
            trajectory = {name: np.asarray(archive[name]) for name in archive.files}
        if not required <= set(trajectory):
            raise ValueError("three-role save-portfolio trajectory is incomplete")
        goal = G1TrainingGoalSpec(**goal_specs[lane_id])
        if abs(goal.width_m - 7.32) > 1e-9 or abs(goal.height_m - 2.44) > 1e-9:
            raise ValueError("three-role save-portfolio requires the regulation goal")
        cases[lane_id] = case_value
        trajectories[lane_id] = trajectory
        goals[lane_id] = goal
        trajectory_hashes[lane_id] = trajectory_hash
    clips = _timeline(cases, trajectories, fps)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for save-portfolio video")
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        if qualification.body_hash != request.get("body_hash"):
            raise ValueError("three-role save-portfolio Body hash changed")
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
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-three-g1-save-portfolio-") as temp:
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
                    raise RuntimeError("save-portfolio ffmpeg raw-video pipe is unavailable")
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
                    raise RuntimeError(f"save-portfolio ffmpeg failed: {stderr[-3000:]}")
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
        raise RuntimeError("save-portfolio encoded video contract changed")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.three_role_save_portfolio_video.v1",
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_evidence_path": str(evidence_file),
        "source_evidence_hash": hash_bytes(evidence_file.read_bytes()),
        "request_hash": evidence["request_hash"],
        "case_trajectory_hashes": trajectory_hashes,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps,
        "goal_contract": "REGULATION_7.32X2.44M_GOAL",
        "case_count": len(cases),
        "contact_span_m": evidence["contact_span_m"],
        "clips": [{"label": clip.label, "frame_count": len(clip.frames)} for clip in clips],
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    manifest["manifest_hash"] = hash_json(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _timeline(
    cases: dict[str, dict[str, Any]],
    trajectories: dict[str, dict[str, np.ndarray]],
    fps: int,
) -> tuple[_Clip, ...]:
    clips: list[_Clip] = []
    first_lane = next(iter(cases))
    first_time = float(trajectories[first_lane]["time"][0])
    clips.append(
        _Clip(
            "FOUR SHOT LANES · THREE G1 · FOUR STRICT PHYSICAL SAVES",
            tuple(_Frame(first_lane, first_time, "wide") for _ in range(round(1.4 * fps))),
        )
    )
    for index, (lane_id, case) in enumerate(cases.items(), start=1):
        replay = cast(dict[str, Any], case["replay"])
        result = cast(dict[str, Any], replay["result"])
        pass_time = float(result["pass_contact_time_sec"])
        shot_time = float(result["shot_contact_time_sec"])
        save_time = float(result["goalkeeper_glove_contact_time_sec"])
        position = cast(list[float], replay["glove_contact_position_m"])
        label = cast(dict[str, Any], case["lane"])["label"]
        pass_mm = 1_000.0 * float(result["pass_delivery_error_m"])
        clips.append(
            _Clip(
                f"SAVE {index}/4 · {label} · {pass_mm:.1f} mm RELAY PASS",
                (
                    *_segment(lane_id, pass_time - 0.65, shot_time + 0.18, 0.82, "chain", fps),
                    *_segment(lane_id, shot_time + 0.18, save_time + 0.82, 0.72, "hero", fps),
                ),
            )
        )
        clips.append(
            _Clip(
                f"SLOW SAVE · CONTACT y={position[1]:+.3f} m · "
                f"{float(replay['incoming_speed_mps']):.2f} m/s · NO GOAL",
                _segment(lane_id, shot_time - 0.22, save_time + 0.48, 0.30, "goal_front", fps),
            )
        )
    last_lane = next(reversed(cases))
    end = float(trajectories[last_lane]["time"][-1])
    clips.append(
        _Clip(
            "4/4 STRICT CPU MUJOCO REPLAYS · 0.885 m CONTACT SPAN · ALL THREE STABLE",
            tuple(_Frame(last_lane, end, "wide_goal") for _ in range(round(2.0 * fps))),
        )
    )
    return tuple(clips)


def _segment(
    lane_id: str,
    start: float,
    end: float,
    speed: float,
    view: str,
    fps: int,
) -> tuple[_Frame, ...]:
    if end <= start or speed <= 0.0:
        raise ValueError("save-portfolio video segment is invalid")
    count = max(1, int(math.ceil((end - start) / speed * fps)))
    return tuple(
        _Frame(lane_id, min(end, start + index / fps * speed), view)
        for index in range(count)
    )


def _write_frames(
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
            trajectory = trajectories[frame.lane_id]
            poses = _sample(trajectory, frame.simulation_time_sec)
            data.qpos[:] = model.qpos0
            for role in ("shooter", "passer", "goalkeeper"):
                data.qpos[free_qpos[role] : free_qpos[role] + 7] = poses[
                    f"{role}_pelvis_pose"
                ]
                data.qpos[joint_qpos[role]] = poses[f"{role}_joint_position"]
            data.qpos[ball_qpos : ball_qpos + 7] = poses["ball_pose"]
            mujoco.mj_forward(model, data)
            _set_camera(camera, frame.view, poses, goal)
            renderer.update_scene(data, camera=camera)
            _add_ball_trail(mujoco, renderer.scene, trajectory, int(poses["index"]))
            stream.write(np.ascontiguousarray(renderer.render().copy()).tobytes())


def _set_camera(
    camera: Any,
    view: str,
    poses: dict[str, NDArray[np.float64] | int],
    goal: G1TrainingGoalSpec,
) -> None:
    ball = cast(NDArray[np.float64], poses["ball_pose"])
    passer = cast(NDArray[np.float64], poses["passer_pelvis_pose"])
    keeper = cast(NDArray[np.float64], poses["goalkeeper_pelvis_pose"])
    lane_y = float(ball[1])
    if view == "wide":
        camera.lookat[:] = (3.65, 0.35 * lane_y, 0.72)
        camera.distance, camera.azimuth, camera.elevation = 10.8, 91.0, -10.0
    elif view == "chain":
        camera.lookat[:] = 0.42 * passer[:3] + 0.58 * ball[:3]
        camera.lookat[2] = 0.67
        camera.distance, camera.azimuth, camera.elevation = 6.1, 108.0, -7.0
    elif view == "goal_front":
        camera.lookat[:] = (goal.plane_x_m - 0.35, lane_y, 1.12)
        camera.distance, camera.azimuth, camera.elevation = 6.0, 178.0, -4.0
    elif view == "hero":
        camera.lookat[:] = (goal.plane_x_m - 1.00, 0.55 * lane_y + 0.45 * keeper[1], 0.98)
        camera.distance, camera.azimuth, camera.elevation = 6.2, 142.0, -6.0
    else:
        camera.lookat[:] = (goal.plane_x_m - 1.25, float(keeper[1]), 0.85)
        camera.distance, camera.azimuth, camera.elevation = 8.3, 118.0, -8.0


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
        np.sin((1.0 - ratio) * angle) / scale * start
        + np.sin(ratio * angle) / scale * end,
        dtype=np.float64,
    )


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
        raise ValueError(f"save-portfolio video model is missing {name}")
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
        f"drawtext={font_option}text='ROSClaw Soccer · MULTI-LANE G1 SAVE PORTFOLIO':"
        f"expansion=none:x={left}:y={round(13 * scale)}:fontsize={round(32 * scale)}:"
        "fontcolor=white",
        f"drawtext={font_option}text='STRICT CPU MUJOCO · REGULATION GOAL · "
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


__all__ = ["render_three_role_save_portfolio_video"]
