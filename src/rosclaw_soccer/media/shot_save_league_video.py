"""Evidence-downstream video for one shot-save alternating response cycle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import G1_DDS_JOINT_NAMES, hash_bytes, hash_json
from rosclaw_soccer.training.shot_save_league import validate_shot_save_growth_round
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_three_player_stadium_model


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def validate_shot_save_league_video_manifest(path: Path) -> dict[str, Any]:
    """Fail closed if the rendered artifact or its authority claims drift."""

    manifest_path = path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("shot-save video manifest must be a JSON object")
    expected = payload.pop("manifest_hash", None)
    if expected != hash_json(payload):
        raise ValueError("shot-save video manifest integrity mismatch")
    video_value = payload.get("video_path")
    if not isinstance(video_value, str):
        raise ValueError("shot-save video path is invalid")
    video_path = Path(video_value).expanduser().resolve()
    numbers = tuple(
        payload.get(name) for name in ("fps", "width", "height", "frame_count", "duration_sec")
    )
    if (
        payload.get("schema_version") != "rosclaw_soccer.shot_save_league_video.v1"
        or not video_path.is_file()
        or payload.get("video_hash") != _file_hash(video_path)
        or any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in numbers
        )
        or payload.get("visualization_only") is not True
        or payload.get("pixels_used_for_scoring") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("shot-save video manifest contract is invalid")
    payload["manifest_hash"] = expected
    return cast(dict[str, Any], payload)


def _id(mujoco: Any, model: Any, object_type: Any, name: str) -> int:
    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise ValueError(f"shot-save video model is missing {name}")
    return value


def _joint_qpos(mujoco: Any, model: Any, prefix: str) -> NDArray[np.int64]:
    result: NDArray[np.int64] = np.asarray(
        [
            model.jnt_qposadr[
                _id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
            ]
            for name in G1_DDS_JOINT_NAMES
        ],
        dtype=np.int64,
    )
    return result


def _load_trajectory(path: Path, expected_hash: str) -> dict[str, np.ndarray]:
    if hash_bytes(path.read_bytes()) != expected_hash:
        raise ValueError("shot-save video trajectory hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
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
        raise ValueError("shot-save video trajectory is incomplete")
    count = len(trajectory["time"])
    if count < 2 or any(len(trajectory[name]) != count for name in required):
        raise ValueError("shot-save video trajectory timebase is inconsistent")
    return trajectory


def _nearest(trajectory: dict[str, np.ndarray], time_sec: float) -> int:
    time = np.asarray(trajectory["time"], dtype=np.float64)
    return int(np.clip(np.searchsorted(time, time_sec, side="left"), 0, len(time) - 1))


def _write_frames(
    *,
    mujoco: Any,
    asset_root: Path,
    trajectory: dict[str, np.ndarray],
    goalkeeper_depth_m: float,
    goal: G1TrainingGoalSpec,
    times: np.ndarray,
    width: int,
    height: int,
    stream: BinaryIO,
) -> None:
    model = build_g1_three_player_stadium_model(
        asset_root,
        passer_origin_m=(5.10, -0.16406006503921598, 0.0),
        goalkeeper_origin_m=(goal.plane_x_m - goalkeeper_depth_m, 0.0, 0.0),
        spec=goal,
    )
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    ball_body = _id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_joint = int(model.body_jntadr[ball_body])
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    free_qpos = {"shooter": 0}
    for role, prefix in (("passer", "passer_"), ("goalkeeper", "goalkeeper_")):
        joint = _id(
            mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint"
        )
        free_qpos[role] = int(model.jnt_qposadr[joint])
    joint_qpos = {
        role: _joint_qpos(mujoco, model, prefix)
        for role, prefix in (
            ("shooter", ""),
            ("passer", "passer_"),
            ("goalkeeper", "goalkeeper_"),
        )
    }
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    try:
        for output_index, time_sec in enumerate(times):
            index = _nearest(trajectory, float(time_sec))
            data.qpos[:] = model.qpos0
            for role in ("shooter", "passer", "goalkeeper"):
                data.qpos[free_qpos[role] : free_qpos[role] + 7] = trajectory[
                    f"{role}_pelvis_pose"
                ][index]
                data.qpos[joint_qpos[role]] = trajectory[f"{role}_joint_position"][index]
            data.qpos[ball_qpos : ball_qpos + 7] = trajectory["ball_pose"][index]
            mujoco.mj_forward(model, data)
            ball = trajectory["ball_pose"][index, :3]
            progress = output_index / max(1, len(times) - 1)
            camera.lookat[:] = (
                float(np.clip(ball[0], 3.8, 6.7)),
                float(0.30 + 0.12 * math.sin(math.pi * progress)),
                0.72,
            )
            camera.distance = 7.5 - 0.35 * math.sin(math.pi * progress)
            camera.azimuth = 137.0 + 5.0 * math.sin(2.0 * math.pi * progress)
            camera.elevation = -8.0
            renderer.update_scene(data, camera=camera)
            stream.write(np.ascontiguousarray(renderer.render()).tobytes())
    finally:
        renderer.close()


def _ffmpeg_command(
    *, output: Path, width: int, height: int, fps: int, boundaries: tuple[float, ...]
) -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for shot-save league video")
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    labels = (
        (0.0, boundaries[0], "ROUND 1 · STRIKER LEARNS · GOAL", "0xFFB000"),
        (boundaries[0], boundaries[1], "REPLAY · STRIKER BREAKS FROZEN KEEPER", "0xFFD166"),
        (boundaries[1], boundaries[2], "ROUND 2 · GOALKEEPER LEARNS · SAVE", "0x42E8A8"),
        (boundaries[2], boundaries[3], "REPLAY · KEEPER ANSWERS THE SAME SHOT", "0x66E0FF"),
    )
    filters = [
        "drawbox=x=0:y=0:w=iw:h=116:color=black@0.64:t=fill",
        (
            f"drawtext=fontfile={font}:text='ROSClaw Shot–Save League':"
            "fontcolor=white:fontsize=42:x=52:y=18"
        ),
        (
            f"drawtext=fontfile={font}:"
            "text='SCORED PHYSICS REPLAY · ONE SHARED BALL · SIM ONLY':"
            "fontcolor=0x66E0FF:fontsize=23:x=54:y=72"
        ),
    ]
    for start, stop, label, color in labels:
        filters.append(
            f"drawtext=fontfile={font}:text='{label}':fontcolor={color}:fontsize=32:"
            "x=(w-text_w)/2:y=h-86:box=1:boxcolor=black@0.58:boxborderw=14:"
            f"enable='between(t,{start:.3f},{stop:.3f})'"
        )
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
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


def render_shot_save_league_video(
    *,
    report_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render attack and defense best responses from their scored archives."""

    report_file = report_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if (
        output.exists()
        or output.suffix.lower() != ".mp4"
        or output == checkout
        or checkout in output.parents
        or not 20 <= fps <= 60
        or not 1280 <= width <= 3840
        or not 720 <= height <= 2160
    ):
        raise ValueError("shot-save league video output contract is invalid")
    report = validate_shot_save_growth_round(report_file)
    # Asset qualification imports MuJoCo.  Select the headless backend before
    # that first import; changing MUJOCO_GL afterwards leaves GLFW selected.
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if qualification.body_hash != report.get("body_hash"):
        raise ValueError("shot-save video Body hash does not match evidence")
    selected_striker = next(
        row
        for row in report["config"]["striker_candidates"]
        if row["policy_id"] == report["selected_striker_policy_id"]
    )
    policies = {
        "attacker-best-response": report["config"]["parent_goalkeeper"],
        "defender-best-response": next(
            row
            for row in report["config"]["goalkeeper_candidates"]
            if row["policy_id"] == report["selected_goalkeeper_policy_id"]
        ),
    }
    goal = G1TrainingGoalSpec(
        plane_x_m=7.50,
        width_m=3.0,
        height_m=2.0,
        depth_m=1.2,
        target_y_m=float(selected_striker["physical_target_y_m"]),
        target_z_m=float(selected_striker["physical_target_z_m"]),
        precision_radius_m=0.10,
    )
    archives = report["trajectory_archives"]
    trajectories = {
        name: _load_trajectory(report_file.parent / contract["path"], contract["hash"])
        for name, contract in archives.items()
    }
    real_times: NDArray[np.float64] = np.arange(
        4.60, 10.60, 1.0 / fps, dtype=np.float64
    )
    slow_times: NDArray[np.float64] = np.arange(
        7.25, 8.75, 0.5 / fps, dtype=np.float64
    )
    frame_groups = (real_times, slow_times, real_times, slow_times)
    durations = tuple(len(group) / fps for group in frame_groups)
    boundaries = tuple(float(value) for value in np.cumsum(durations))
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        _ffmpeg_command(
            output=output,
            width=width,
            height=height,
            fps=fps,
            boundaries=boundaries,
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("shot-save video ffmpeg pipe is unavailable")
    try:
        for name, times in (
            ("attacker-best-response", real_times),
            ("attacker-best-response", slow_times),
            ("defender-best-response", real_times),
            ("defender-best-response", slow_times),
        ):
            _write_frames(
                mujoco=mujoco,
                asset_root=asset_root.expanduser().resolve(),
                trajectory=trajectories[name],
                goalkeeper_depth_m=float(policies[name]["depth_from_goal_line_m"]),
                goal=goal,
                times=times,
                width=width,
                height=height,
                stream=cast(BinaryIO, process.stdin),
            )
        process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        if process.wait():
            raise RuntimeError(f"shot-save video ffmpeg failed: {stderr[-3000:]}")
    except BaseException:
        process.stdin.close()
        process.kill()
        process.wait()
        raise
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.shot_save_league_video.v1",
        "video_path": str(output),
        "video_hash": _file_hash(output),
        "report_hash": report["report_hash"],
        "report_file_hash": hash_bytes(report_file.read_bytes()),
        "trajectory_hashes": {
            name: contract["hash"] for name, contract in archives.items()
        },
        "selected_striker_policy_hash": report["selected_striker_policy_hash"],
        "selected_goalkeeper_policy_hash": report["selected_goalkeeper_policy_hash"],
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": int(sum(len(group) for group in frame_groups)),
        "duration_sec": float(sum(durations)),
        "clips": [
            "STRIKER_BEST_RESPONSE_GOAL",
            "STRIKER_SLOW_REPLAY",
            "GOALKEEPER_BEST_RESPONSE_SAVE",
            "GOALKEEPER_SLOW_REPLAY",
        ],
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "goal_spec": asdict(goal),
    }
    manifest["manifest_hash"] = hash_json(manifest)
    manifest_path = output.with_suffix(".json")
    _atomic_json(manifest_path, manifest)
    return validate_shot_save_league_video_manifest(manifest_path)


__all__ = [
    "render_shot_save_league_video",
    "validate_shot_save_league_video_manifest",
]
