"""Evidence-downstream development video for a rejected Goalkeeper V2 actor."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from rosclaw_soccer.evidence.goalkeeper_v2 import goalkeeper_v2_implementation_hash
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.skills.team.development_evidence import three_role_development_kwargs
from rosclaw_soccer.skills.team.shared_world import G1GoalkeeperConfig, simulate_shared_world
from rosclaw_soccer.world.field import build_g1_three_player_stadium_model


@dataclass(frozen=True)
class GoalkeeperV2DevelopmentVideoResult:
    output_path: str
    manifest_path: str
    video_hash: str
    candidate_policy_hash: str
    candidate_evidence_hash: str
    implementation_hash: str
    frame_count: int
    fps: int
    width: int
    height: int
    duration_sec: float
    visualization_only: bool = True
    pixels_used_for_scoring: bool = False
    candidate_promoted: bool = False
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.goalkeeper_v2_development_video.v1"


def render_goalkeeper_v2_development_video(
    *,
    candidate_evidence_path: Path,
    actor_artifact_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> GoalkeeperV2DevelopmentVideoResult:
    """Render parent/candidate numerical trajectories with an explicit rejection label."""

    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("goalkeeper development video must remain outside source checkout")
    if output.suffix.lower() != ".mp4" or output.exists():
        raise ValueError("goalkeeper development video requires a new .mp4 path")
    if not 10 <= fps <= 60 or width < 640 or height < 360:
        raise ValueError("goalkeeper development video dimensions or fps are invalid")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for goalkeeper development video")
    evidence = json.loads(candidate_evidence_path.read_text(encoding="utf-8"))
    decision = evidence.get("promotion_decision", {})
    if decision.get("verdict") != "REJECTED":
        raise ValueError("development video is only for a rejected candidate")
    implementation_hash = goalkeeper_v2_implementation_hash()
    if evidence.get("implementation_hash") != implementation_hash:
        raise ValueError("development video evidence does not bind the current implementation")
    candidate_hash = str(decision.get("candidate_policy_hash", ""))
    actor = json.loads(actor_artifact_path.read_text(encoding="utf-8"))
    if actor.get("policy_hash") != candidate_hash:
        raise ValueError("development video actor does not match candidate evidence")

    # MuJoCo chooses its GL platform at first import.  Shared-world simulation
    # imports it too, so the headless boundary must be active before rollouts,
    # not merely before Renderer construction.
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    base = three_role_development_kwargs()
    goal = base["goal_spec"]
    goalkeeper = base.get("goalkeeper_config")
    if not isinstance(goalkeeper, G1GoalkeeperConfig):
        raise RuntimeError("goalkeeper development video requires a parent config")
    parent = replace(
        goalkeeper,
        actor_observation_mode="visible_ball_history_v3",
        anticipation_enabled=False,
        actor_artifact_path=None,
    )
    scenarios = (
        ("UPPER LEFT · 1.0 S COVERAGE CHALLENGE", 1.0, 0.90, 1.60),
        ("LOW RIGHT · 1.2 S DEVELOPMENT TRIAL", 1.2, -0.40, 0.60),
        ("HIGH RIGHT · 1.2 S COVERAGE CHALLENGE", 1.2, -0.40, 1.20),
        ("CENTER · 0.8 S MATCHED TRIAL", 0.8, 0.0, 1.00),
    )
    rollouts: list[tuple[str, str, dict[str, np.ndarray]]] = []
    for label, deadline, target_y, target_z in scenarios:
        start = (goal.plane_x_m - 5.0, 0.0, 0.6)
        velocity = (
            5.0 / deadline,
            target_y / deadline,
            (target_z - 0.6 + 4.905 * deadline * deadline) / deadline,
        )
        for policy_label, artifact_path in (
            ("FROZEN PARENT", None),
            ("RL CANDIDATE", actor_artifact_path),
        ):
            kwargs = dict(base)
            kwargs.update(
                goalkeeper_config=replace(parent, actor_artifact_path=artifact_path),
                ball_launcher_position_m=start,
                ball_launcher_velocity_mps=velocity,
                simulation_duration_sec=3.0,
            )
            _, trajectory = simulate_shared_world(asset_root, **kwargs)
            rollouts.append((label, policy_label, trajectory))

    import mujoco

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = output.with_suffix(".json")
    try:
        model = build_g1_three_player_stadium_model(
            asset_root,
            passer_origin_m=base["passer_origin"],
            goalkeeper_origin_m=(
                goal.plane_x_m - parent.depth_from_goal_line_m,
                0.0,
                0.0,
            ),
            spec=goal,
        )
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        frames_per_rollout = int(round(3.0 * fps))
        process = subprocess.Popen(
            _ffmpeg_command(
                ffmpeg=ffmpeg,
                output=output,
                fps=fps,
                width=width,
                height=height,
                labels=tuple((scenario, policy) for scenario, policy, _ in rollouts),
                frames_per_rollout=frames_per_rollout,
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None:
            raise RuntimeError("goalkeeper development ffmpeg pipe is unavailable")
        try:
            for _, _, trajectory in rollouts:
                _write_rollout(
                    mujoco=mujoco,
                    model=model,
                    data=data,
                    renderer=renderer,
                    trajectory=trajectory,
                    fps=fps,
                    frame_count=frames_per_rollout,
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
            raise RuntimeError(f"goalkeeper development ffmpeg failed: {stderr[-2000:]}")
        renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    result = GoalkeeperV2DevelopmentVideoResult(
        output_path=str(output),
        manifest_path=str(manifest),
        video_hash=_file_hash(output),
        candidate_policy_hash=candidate_hash,
        candidate_evidence_hash=str(evidence["evidence_hash"]),
        implementation_hash=implementation_hash,
        frame_count=len(rollouts) * frames_per_rollout,
        fps=fps,
        width=width,
        height=height,
        duration_sec=len(rollouts) * frames_per_rollout / fps,
    )
    manifest.write_text(
        json.dumps(result.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _write_rollout(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    trajectory: dict[str, np.ndarray],
    fps: int,
    frame_count: int,
    stream: BinaryIO,
) -> None:
    ball_body = _id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_joint = int(model.body_jntadr[ball_body])
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    joint = _id(
        mujoco,
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "goalkeeper_floating_base_joint",
    )
    keeper_base = int(model.jnt_qposadr[joint])
    keeper_joints = np.asarray(
        [
            model.jnt_qposadr[
                _id(
                    mujoco,
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    "goalkeeper_" + name,
                )
            ]
            for name in G1_DDS_JOINT_NAMES
        ],
        dtype=np.int64,
    )
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (7.0, 0.0, 0.82)
    camera.distance, camera.azimuth, camera.elevation = 5.8, 145.0, -7.0
    time = np.asarray(trajectory["time"], dtype=np.float64)
    for index in range(frame_count):
        timestamp = min(float(time[-1]), index / fps)
        sample = min(len(time) - 1, int(np.searchsorted(time, timestamp, side="left")))
        data.qpos[:] = model.qpos0
        data.qpos[keeper_base : keeper_base + 7] = trajectory["goalkeeper_pelvis_pose"][sample]
        data.qpos[keeper_joints] = trajectory["goalkeeper_joint_position"][sample]
        data.qpos[ball_qpos : ball_qpos + 7] = trajectory["ball_pose"][sample]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        stream.write(np.ascontiguousarray(renderer.render()).tobytes())


def _ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    fps: int,
    width: int,
    height: int,
    labels: tuple[tuple[str, str], ...],
    frames_per_rollout: int,
) -> list[str]:
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    filters = [
        "drawbox=x=0:y=0:w=iw:h=116:color=0x030711@0.86:t=fill",
        "drawbox=x=0:y=h-58:w=iw:h=58:color=0x030711@0.86:t=fill",
        f"drawtext=fontfile={font}:text='ROSClaw Goalkeeper V2 · CAUSAL HIERARCHICAL PPO':"
        "x=28:y=14:fontsize=31:fontcolor=white",
        f"drawtext=fontfile={font}:text='REJECTED CANDIDATE · NOT PROMOTED · "
        "CPU MUJOCO · SIM ONLY':"
        "x=28:y=h-39:fontsize=18:fontcolor=0xFFB45C",
    ]
    for index, (scenario, policy) in enumerate(labels):
        start = index * frames_per_rollout / fps
        # FFmpeg's between() includes both endpoints.  Exclude half a frame
        # from the boundary so adjacent rollout labels never overlap.
        end = ((index + 1) * frames_per_rollout - 0.5) / fps
        text = f"{scenario} · {policy}"
        filters.append(
            f"drawtext=fontfile={font}:text='{text}':x=28:y=65:fontsize=22:fontcolor=0x65F59A:"
            f"enable='between(t,{start:.6f},{end:.6f})'"
        )
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


def _id(mujoco: Any, model: Any, object_type: Any, name: str) -> int:
    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise ValueError(f"goalkeeper development model is missing {name}")
    return value


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "GoalkeeperV2DevelopmentVideoResult",
    "render_goalkeeper_v2_development_video",
]
