"""Render an honest multi-camera reel from one continuous 3v3 match trace."""

from __future__ import annotations

import argparse
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

from rosclaw_soccer.media.independent_team_video import (
    _addresses,
    _color_player,
    _PlayerAddresses,
    _probe,
    _sample,
)
from rosclaw_soccer.media.trajectory_render import append_sphere, escape_filtergraph_option
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.continuous_competitive_match_growth import (
    build_continuous_competitive_fixture,
    validate_continuous_competitive_match_growth,
)
from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model

_CLAIM = "CONTINUOUS_PHYSICAL_3V3_DEVELOPMENT_REPLAY"
_REQUIRED_SKILLS = frozenset({"pass", "ball_win", "dribble", "sprint"})
_MATCH_AGENT_IDS = (
    "blue.finisher",
    "blue.goalkeeper",
    "blue.playmaker",
    "red.finisher",
    "red.goalkeeper",
    "red.playmaker",
)


@dataclass(frozen=True)
class _Frame:
    simulation_time_sec: float
    camera: str


@dataclass(frozen=True)
class _Clip:
    title: str
    frames: tuple[_Frame, ...]


def render_continuous_competitive_match_video(
    *,
    exam_path: Path,
    asset_root: Path,
    output_path: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render only integrity-checked physics evidence; pixels never score the run."""

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
        raise ValueError("continuous-match video output contract is invalid")
    report_path = exam_path.expanduser().resolve()
    report = validate_continuous_competitive_match_growth(report_path)
    assessment = cast(dict[str, Any], report.get("primary_assessment"))
    gates = cast(dict[str, bool], assessment.get("gates"))
    events = cast(list[dict[str, Any]], assessment.get("events"))
    observed = {str(event.get("skill")) for event in events}
    if (
        report.get("exact_replay") is not True
        or report.get("status") != "REJECTED_CONTINUOUS_MATCH_CHAIN"
        or report.get("passed") is not False
        or report.get("primary_result", {}).get("safe") is not True
        or assessment.get("safe") is not True
        or assessment.get("passed") is not False
        or assessment.get("first_failed_skill") != "shot"
        or not _REQUIRED_SKILLS.issubset(observed)
        or gates.get("world_safe") is not True
        or gates.get("strict_replay") is not True
        or gates.get("shot_observed") is not False
        or gates.get("save_observed") is not False
    ):
        raise ValueError("continuous-match video requires the safe rejected development rung")
    artifact = cast(dict[str, str], report.get("primary_artifact"))
    trajectory_path = report_path.parent / str(artifact.get("file"))
    if not trajectory_path.is_file() or hash_bytes(trajectory_path.read_bytes()) != artifact.get(
        "file_hash"
    ):
        raise ValueError("continuous-match video trajectory file changed")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        trajectory = {name: np.asarray(archive[name]) for name in archive.files}
    digest = trajectory_digest(trajectory)
    if digest != artifact.get("trajectory_digest") or digest != assessment.get("trajectory_hash"):
        raise ValueError("continuous-match video trajectory identity changed")
    clips = _timeline(trajectory, fps=fps)
    first_touch_time = _first_touch_time(trajectory, events, agent_ids=_MATCH_AGENT_IDS)
    source_files = {
        str(report_path): hash_bytes(report_path.read_bytes()),
        str(trajectory_path): hash_bytes(trajectory_path.read_bytes()),
    }
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for continuous-match video")
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ["MUJOCO_GL"] = "egl"
    try:
        fixture = build_continuous_competitive_fixture(asset_root)
        if tuple(sorted(player.agent_id for player in fixture.players)) != _MATCH_AGENT_IDS:
            raise ValueError("continuous-match video fixture roster changed")
        import mujoco

        model = build_g1_multi_player_stadium_model(
            asset_root,
            players=fixture.players,
            spec=fixture.goal,
        )
        for player in fixture.players:
            _color_player(
                model,
                root_body_name=player.body_prefix + "pelvis",
                rgba=(0.88, 0.05, 0.04, 1.0)
                if player.agent_id.startswith("red.")
                else (0.02, 0.20, 0.92, 1.0),
            )
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-continuous-match-video-") as temp:
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
                    raise RuntimeError("continuous-match raw video pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        renderer=renderer,
                        players=fixture.players,
                        trajectory=trajectory,
                        clips=clips,
                        stream=cast(BinaryIO, process.stdin),
                    )
                except BrokenPipeError as error:
                    process.stdin.close()
                    process.wait()
                    stderr = (
                        process.stderr.read().decode(errors="replace") if process.stderr else ""
                    )
                    raise RuntimeError(
                        f"continuous-match ffmpeg closed its input: {stderr[-3000:]}"
                    ) from error
                except BaseException:
                    process.stdin.close()
                    process.kill()
                    process.wait()
                    raise
                process.stdin.close()
                stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
                if process.wait():
                    raise RuntimeError(f"continuous-match ffmpeg failed: {stderr[-3000:]}")
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl
    frame_count = sum(len(clip.frames) for clip in clips)
    probe = _probe(ffprobe, output)
    if (
        probe["width"] != width
        or probe["height"] != height
        or probe["fps"] != fps
        or abs(probe["frame_count"] - frame_count) > 1
    ):
        raise RuntimeError("continuous-match encoded video contract changed")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.continuous_competitive_match_video.v1",
        "claim": _CLAIM,
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_files": source_files,
        "source_report_hash": report["report_hash"],
        "source_report_status": report["status"],
        "source_trajectory_digest": digest,
        "observed_skills": sorted(observed),
        "first_touch_time_sec": first_touch_time,
        "first_failed_skill": assessment["first_failed_skill"],
        "strict_replay": True,
        "world_safe": True,
        "whole_body_g1_count": 6,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps,
        "camera_views": sorted({frame.camera for clip in clips for frame in clip.frames}),
        "ball_trail_is_visualization": True,
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "shot_or_save_claimed": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "renderer_hash": hash_bytes(Path(__file__).read_bytes()),
    }
    manifest["manifest_hash"] = hash_json(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_continuous_competitive_match_video_manifest(manifest_path)
    return manifest


def validate_continuous_competitive_match_video_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("continuous-match video manifest must be an object")
    declared = payload.pop("manifest_hash", None)
    try:
        video = Path(str(payload.get("video_path"))).expanduser().resolve()
        sources = payload.get("source_files")
        if (
            declared != hash_json(payload)
            or payload.get("schema_version")
            != "rosclaw_soccer.continuous_competitive_match_video.v1"
            or payload.get("claim") != _CLAIM
            or not video.is_file()
            or hash_bytes(video.read_bytes()) != payload.get("video_hash")
            or not isinstance(sources, dict)
            or any(
                not Path(source).is_file() or hash_bytes(Path(source).read_bytes()) != expected
                for source, expected in sources.items()
            )
            or payload.get("source_report_status") != "REJECTED_CONTINUOUS_MATCH_CHAIN"
            or payload.get("observed_skills") != sorted(_REQUIRED_SKILLS)
            or payload.get("first_failed_skill") != "shot"
            or payload.get("strict_replay") is not True
            or payload.get("world_safe") is not True
            or payload.get("whole_body_g1_count") != 6
            or payload.get("ball_trail_is_visualization") is not True
            or payload.get("visualization_only") is not True
            or payload.get("pixels_used_for_scoring") is not False
            or payload.get("promotion_eligible") is not False
            or payload.get("shot_or_save_claimed") is not False
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("commercial_use_allowed") is not False
            or payload.get("renderer_hash") != hash_bytes(Path(__file__).read_bytes())
        ):
            raise ValueError("continuous-match video authority contract is invalid")
    finally:
        payload["manifest_hash"] = declared
    return cast(dict[str, Any], payload)


def _timeline(trajectory: dict[str, NDArray[Any]], *, fps: int) -> tuple[_Clip, ...]:
    start = float(trajectory["time"][0])
    end = float(trajectory["time"][-1])

    def hold(timestamp: float, duration_sec: float, camera: str) -> tuple[_Frame, ...]:
        return tuple(_Frame(timestamp, camera) for _ in range(round(duration_sec * fps)))

    def play(first: float, last: float, speed: float, camera: str) -> tuple[_Frame, ...]:
        duration = max(0.0, last - first) / speed
        return tuple(
            _Frame(min(last, first + index / fps * speed), camera)
            for index in range(max(1, math.ceil(duration * fps)))
        )

    segments = (
        (start, 0.85, "RED BUILD-UP · SIX PRIVATE ROSCLAW AGENT CELLS"),
        (0.85, 2.35, "PASS · RED PLAYMAKER TO RED FINISHER · FOOT CONTACT"),
        (2.35, 4.25, "RUN-ONTO FIRST TOUCH · NO TELEPORT · LIVE BALL PHYSICS"),
        (4.25, 7.35, "BLUE PRESS · RED SUPPORT RUN · ROLE-AWARE TRANSITION"),
        (7.35, 8.75, "BALL WIN · BLUE PLAYMAKER INTERCEPTS"),
        (8.75, end, "COUNTERATTACK · DRIBBLE + SPRINT"),
    )
    clips: list[_Clip] = [
        _Clip(
            "CONTINUOUS 3v3 PHYSICS MATCH · DEVELOPMENT EVIDENCE",
            hold(start, 1.0, "wide"),
        )
    ]
    for first, last, title in segments:
        lower = min(max(first, start), end)
        upper = min(max(last, lower), end)
        clips.append(_Clip(title, play(lower, upper, 1.0, "broadcast")))
    clips.extend(
        (
            _Clip(
                "TACTICAL REPLAY · PASS + PHYSICAL FIRST TOUCH · 0.60x",
                play(1.20, min(4.15, end), 0.60, "touchline"),
            ),
            _Clip(
                "TACTICAL REPLAY · PRESS + BALL WIN + DRIBBLE · 0.60x",
                play(min(7.20, end), end, 0.60, "counter"),
            ),
            _Clip(
                "RUNG RESULT · PASS / WIN / DRIBBLE / SPRINT PASSED · SHOT NEXT",
                hold(end, 1.5, "wide"),
            ),
        )
    )
    return tuple(clips)


def _first_touch_time(
    trajectory: dict[str, NDArray[Any]],
    events: list[dict[str, Any]],
    *,
    agent_ids: tuple[str, ...],
) -> float:
    pass_event = next((event for event in events if event.get("skill") == "pass"), None)
    if pass_event is None or not isinstance(pass_event.get("target_agent_id"), str):
        raise ValueError("continuous-match video lacks its pass target")
    target = str(pass_event["target_agent_id"])
    if len(agent_ids) < 2 or len(agent_ids) != len(set(agent_ids)) or target not in agent_ids:
        raise ValueError("continuous-match video pass target is absent")
    target_code = agent_ids.index(target) + 1
    contact_codes = np.asarray(trajectory["ball_contact_agent_code"], dtype=np.int64)
    effector_codes = np.asarray(trajectory["ball_contact_effector_code"], dtype=np.int64)
    time = np.asarray(trajectory["time"], dtype=np.float64)
    candidates = np.flatnonzero(
        (contact_codes == target_code)
        & np.isin(effector_codes, np.asarray((1, 2), dtype=np.int64))
        & (time > float(pass_event["time_sec"]))
    )
    if not len(candidates):
        raise ValueError("continuous-match video pass has no physical first touch")
    return float(time[int(candidates[0])])


def _write_frames(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    players: tuple[Any, ...],
    trajectory: dict[str, NDArray[Any]],
    clips: tuple[_Clip, ...],
    stream: BinaryIO,
) -> None:
    addresses: tuple[_PlayerAddresses, ...] = tuple(
        _addresses(mujoco, model, player.agent_id, player.body_prefix) for player in players
    )
    ball_joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"))
    if ball_joint < 0:
        raise ValueError("continuous-match video model is missing ball_free")
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    time = np.asarray(trajectory["time"], dtype=np.float64)
    ball_path = np.asarray(trajectory["ball_pose"], dtype=np.float64)[:, :3]
    total = max(1, sum(len(clip.frames) for clip in clips) - 1)
    global_frame = 0
    for clip in clips:
        for frame in clip.frames:
            sample = _sample(trajectory, frame.simulation_time_sec, addresses)
            data.qpos[:] = model.qpos0
            for address in addresses:
                key = address.agent_id.replace(".", "_").replace(":", "_").replace("-", "_")
                data.qpos[address.free_qpos : address.free_qpos + 7] = sample[f"{key}_pelvis_pose"]
                data.qpos[address.joint_qpos] = sample[f"{key}_joint_position"]
            data.qpos[ball_qpos : ball_qpos + 7] = sample["ball_pose"]
            mujoco.mj_forward(model, data)
            progress = global_frame / total
            ball = np.asarray(sample["ball_pose"][:3], dtype=np.float64)
            _set_camera(camera, frame.camera, ball=ball, progress=progress)
            renderer.update_scene(data, camera=camera)
            trail_end = int(np.searchsorted(time, frame.simulation_time_sec, side="right"))
            for trail_frame in range(max(0, trail_end - 36), trail_end, 6):
                age = (trail_end - trail_frame) / 36.0
                append_sphere(
                    mujoco,
                    renderer.scene,
                    ball_path[trail_frame] + np.asarray((0.0, 0.0, 0.012)),
                    0.025,
                    (1.0, 0.72, 0.08, max(0.10, 0.48 * (1.0 - age))),
                )
            stream.write(np.ascontiguousarray(renderer.render().copy()).tobytes())
            global_frame += 1


def _set_camera(camera: Any, name: str, *, ball: NDArray[np.float64], progress: float) -> None:
    if name == "touchline":
        camera.lookat[:] = (float(ball[0]), float(ball[1]), 0.72)
        camera.distance = 5.1
        camera.azimuth = 126.0
        camera.elevation = -13.0
    elif name == "counter":
        camera.lookat[:] = (float(ball[0]), float(ball[1]), 0.70)
        camera.distance = 5.4
        camera.azimuth = 54.0
        camera.elevation = -14.0
    else:
        camera.lookat[:] = (3.25, -0.35, 0.74)
        camera.distance = 11.2 - 0.25 * math.sin(math.pi * progress)
        camera.azimuth = 91.0 + 3.0 * math.sin(2.0 * math.pi * progress)
        camera.elevation = -11.0


def _write_labels(directory: Path, clips: tuple[_Clip, ...]) -> tuple[Path, ...]:
    labels: list[Path] = []
    for index, clip in enumerate(clips):
        path = directory / f"label-{index:02d}.txt"
        path.write_text(clip.title, encoding="utf-8")
        labels.append(path)
    return tuple(labels)


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
    scale = height / 720.0
    left = round(28 * scale)
    filters = [
        f"drawbox=x=0:y=0:w=iw:h={round(124 * scale)}:color=0x020714@0.82:t=fill",
        f"drawbox=x=0:y=h-{round(72 * scale)}:w=iw:h={round(72 * scale)}:"
        "color=0x020714@0.84:t=fill",
        f"drawtext=font='DejaVu Sans':text='ROSClaw Soccer · CONTINUOUS 3v3 PHYSICS MATCH':"
        f"expansion=none:x={left}:y={round(12 * scale)}:fontsize={round(31 * scale)}:"
        "fontcolor=white",
        f"drawtext=font='DejaVu Sans':text='STRICT REPLAY PASS · WORLD SAFE PASS · "
        "DEVELOPMENT RUNG - NOT PROMOTED':"
        f"expansion=none:x={left}:y={round(50 * scale)}:fontsize={round(18 * scale)}:"
        "fontcolor=0xFFC857",
        f"drawtext=font='DejaVu Sans':text='PASS  |  FIRST TOUCH  |  PRESS / BALL WIN  |  "
        "DRIBBLE  |  SPRINT     ·     SHOT + SAVE - NEXT RUNG':"
        f"expansion=none:x={left}:y=h-{round(47 * scale)}:fontsize={round(18 * scale)}:"
        "fontcolor=0x8ED9FF",
    ]
    offset = 0.0
    for clip, label in zip(clips, labels, strict=True):
        end = offset + len(clip.frames) / fps
        filters.append(
            f"drawtext=font='DejaVu Sans':textfile={escape_filtergraph_option(str(label))}:"
            f"expansion=none:x={left}:y={round(84 * scale)}:fontsize={round(19 * scale)}:"
            f"fontcolor=0x68F596:enable='between(t,{offset:.6f},{end:.6f})'"
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
    parser.add_argument("--exam", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    value = render_continuous_competitive_match_video(
        exam_path=arguments.exam,
        asset_root=arguments.asset_root,
        output_path=arguments.output,
        fps=arguments.fps,
        width=arguments.width,
        height=arguments.height,
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "render_continuous_competitive_match_video",
    "validate_continuous_competitive_match_video_manifest",
]
