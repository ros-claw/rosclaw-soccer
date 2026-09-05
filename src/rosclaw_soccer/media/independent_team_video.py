"""Render the evidence-bound six-G1 independent-team development reel."""

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

from rosclaw_soccer.media.trajectory_render import escape_filtergraph_option
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import G1_DDS_JOINT_NAMES, hash_bytes, hash_json
from rosclaw_soccer.training.independent_team_growth import (
    build_independent_three_vs_three_fixture,
    validate_independent_team_growth,
)
from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model

_CLAIM = "INDEPENDENT_ROSCLAW_AGENT_CELLS_3V3_DEVELOPMENT_REPLAY"


@dataclass(frozen=True)
class _Clip:
    case_id: str
    title: str
    times_sec: tuple[float, ...]


@dataclass(frozen=True)
class _PlayerAddresses:
    agent_id: str
    free_qpos: int
    joint_qpos: NDArray[np.int64]


def render_independent_team_video(
    *,
    evidence_dir: Path,
    asset_root: Path,
    output_path: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    root = evidence_dir.expanduser().resolve()
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
        raise ValueError("independent-team video output contract is invalid")
    report_path = root / "retention-exam.json"
    report = validate_independent_team_growth(report_path)
    if report.get("status") != "PASS_INDEPENDENT_3V3_FOUNDATION":
        raise ValueError("independent-team video requires passing evidence")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("independent-team video requires four retention cases")
    source_files = {str(report_path): hash_bytes(report_path.read_bytes())}
    trajectories: dict[str, dict[str, NDArray[Any]]] = {}
    titles = (
        "RED BUILD-UP · PLAYMAKER NEGOTIATES PASS · BLUE KEEPER TRACKS",
        "BLUE COUNTER · RED KEEPER DEFENDS THEN DISTRIBUTES",
        "FINISHERS COMPETE FOR THE SHOOTING LANE",
        "KEEPER DISTRIBUTION · BOTH TEAMS RE-SHAPE AUTONOMOUSLY",
    )
    clips: list[_Clip] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("exact_replay") is not True:
            raise ValueError("independent-team video selected an unverified row")
        result = row.get("result")
        artifact = row.get("primary_artifact")
        scenario = row.get("scenario")
        if (
            not isinstance(result, dict)
            or result.get("passed") is not True
            or not isinstance(artifact, dict)
            or not isinstance(scenario, dict)
        ):
            raise ValueError("independent-team video row is incomplete")
        path = root / f"case-{index:03d}" / str(artifact.get("file"))
        if hash_bytes(path.read_bytes()) != artifact.get("file_hash"):
            raise ValueError("independent-team video trajectory changed")
        with np.load(path, allow_pickle=False) as archive:
            trajectory = {name: np.asarray(archive[name]) for name in archive.files}
        if trajectory_digest(trajectory) != result.get("trajectory_hash"):
            raise ValueError("independent-team trajectory digest is not result-bound")
        case_id = str(scenario.get("scenario_id"))
        trajectories[case_id] = trajectory
        source_files[str(path)] = hash_bytes(path.read_bytes())
        start = float(trajectory["time"][0])
        end = float(trajectory["time"][-1])
        hold = tuple(start for _ in range(round(0.65 * fps)))
        playback = tuple(
            min(end, start + frame / fps) for frame in range(math.ceil((end - start) * fps))
        )
        clips.append(_Clip(case_id, titles[index], (*hold, *playback)))
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for independent-team video")
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        # MuJoCo selects its GL backend at first import, so qualification and
        # scene construction must also happen inside the headless EGL scope.
        fixture = build_independent_three_vs_three_fixture(asset_root)
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
                rgba=(0.84, 0.08, 0.07, 1.0)
                if player.agent_id.startswith("red.")
                else (0.04, 0.24, 0.88, 1.0),
            )
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-independent-3v3-") as temporary:
                labels = _write_labels(Path(temporary), clips)
                process = subprocess.Popen(
                    _ffmpeg_command(
                        ffmpeg=ffmpeg,
                        output=output,
                        width=width,
                        height=height,
                        fps=fps,
                        clips=tuple(clips),
                        labels=labels,
                    ),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                if process.stdin is None:
                    raise RuntimeError("independent-team raw video pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        renderer=renderer,
                        players=fixture.players,
                        trajectories=trajectories,
                        clips=tuple(clips),
                        stream=cast(BinaryIO, process.stdin),
                    )
                except BrokenPipeError as error:
                    process.stdin.close()
                    process.wait()
                    stderr = (
                        process.stderr.read().decode(errors="replace") if process.stderr else ""
                    )
                    raise RuntimeError(
                        f"independent-team ffmpeg closed its input: {stderr[-3000:]}"
                    ) from error
                except BaseException:
                    process.stdin.close()
                    process.kill()
                    process.wait()
                    raise
                process.stdin.close()
                stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
                if process.wait():
                    raise RuntimeError(f"independent-team ffmpeg failed: {stderr[-3000:]}")
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl
    probe = _probe(ffprobe, output)
    frame_count = sum(len(clip.times_sec) for clip in clips)
    if (
        probe["width"] != width
        or probe["height"] != height
        or probe["fps"] != fps
        or abs(probe["frame_count"] - frame_count) > 1
    ):
        raise RuntimeError("independent-team encoded video contract changed")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.independent_team_video.v1",
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_files": source_files,
        "source_report_hash": report["report_hash"],
        "claim": _CLAIM,
        "strict_replay": True,
        "cases_shown": [clip.case_id for clip in clips],
        "whole_body_g1_count": 6,
        "red_agent_count": 3,
        "blue_agent_count": 3,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps,
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "contact_skill_router_complete": False,
        "commercial_use_allowed": False,
    }
    manifest["manifest_hash"] = hash_json(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_independent_team_video_manifest(manifest_path)
    return manifest


def _write_frames(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    players: tuple[Any, ...],
    trajectories: dict[str, dict[str, NDArray[Any]]],
    clips: tuple[_Clip, ...],
    stream: BinaryIO,
) -> None:
    addresses = tuple(
        _addresses(mujoco, model, player.agent_id, player.body_prefix) for player in players
    )
    ball_joint = _id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    total = max(1, sum(len(clip.times_sec) for clip in clips) - 1)
    global_frame = 0
    for clip in clips:
        trajectory = trajectories[clip.case_id]
        for timestamp in clip.times_sec:
            sample = _sample(trajectory, timestamp, addresses)
            data.qpos[:] = model.qpos0
            for address in addresses:
                key = _agent_key(address.agent_id)
                data.qpos[address.free_qpos : address.free_qpos + 7] = sample[f"{key}_pelvis_pose"]
                data.qpos[address.joint_qpos] = sample[f"{key}_joint_position"]
            data.qpos[ball_qpos : ball_qpos + 7] = sample["ball_pose"]
            mujoco.mj_forward(model, data)
            progress = global_frame / total
            camera.lookat[:] = (3.65, 0.0, 0.74)
            camera.distance = 11.0 - 0.25 * math.sin(math.pi * progress)
            camera.azimuth = 91.0 + 4.0 * math.sin(2.0 * math.pi * progress)
            camera.elevation = -10.0
            renderer.update_scene(data, camera=camera)
            stream.write(np.ascontiguousarray(renderer.render().copy()).tobytes())
            global_frame += 1


def _sample(
    trajectory: dict[str, NDArray[Any]],
    timestamp: float,
    addresses: tuple[_PlayerAddresses, ...],
) -> dict[str, NDArray[np.float64]]:
    time = np.asarray(trajectory["time"], dtype=np.float64)
    upper = int(np.searchsorted(time, timestamp, side="right"))
    if upper <= 0:
        lower = upper = 0
        ratio = 0.0
    elif upper >= len(time):
        lower = upper = len(time) - 1
        ratio = 0.0
    else:
        lower = upper - 1
        ratio = float((timestamp - time[lower]) / (time[upper] - time[lower]))
    result = {
        "ball_pose": _pose(trajectory["ball_pose"][lower], trajectory["ball_pose"][upper], ratio)
    }
    for address in addresses:
        key = _agent_key(address.agent_id)
        result[f"{key}_pelvis_pose"] = _pose(
            trajectory[f"{key}_pelvis_pose"][lower],
            trajectory[f"{key}_pelvis_pose"][upper],
            ratio,
        )
        result[f"{key}_joint_position"] = _lerp(
            trajectory[f"{key}_joint_position"][lower],
            trajectory[f"{key}_joint_position"][upper],
            ratio,
        )
    return result


def _pose(first: NDArray[Any], second: NDArray[Any], ratio: float) -> NDArray[np.float64]:
    value = _lerp(first, second, ratio)
    value[3:7] /= max(float(np.linalg.norm(value[3:7])), 1.0e-12)
    return value


def _lerp(first: NDArray[Any], second: NDArray[Any], ratio: float) -> NDArray[np.float64]:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    return np.asarray(a + ratio * (b - a), dtype=np.float64)


def _addresses(mujoco: Any, model: Any, agent_id: str, prefix: str) -> _PlayerAddresses:
    free = _id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint")
    joints = np.asarray(
        [
            model.jnt_qposadr[_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, prefix + joint_name)]
            for joint_name in G1_DDS_JOINT_NAMES
        ],
        dtype=np.int64,
    )
    return _PlayerAddresses(agent_id, int(model.jnt_qposadr[free]), joints)


def _color_player(
    model: Any, *, root_body_name: str, rgba: tuple[float, float, float, float]
) -> None:
    root = int(model.body(root_body_name).id)
    for geom_id in range(int(model.ngeom)):
        body = int(model.geom_bodyid[geom_id])
        while body > 0 and body != root:
            body = int(model.body_parentid[body])
        if body == root:
            model.geom_matid[geom_id] = -1
            model.geom_rgba[geom_id] = rgba


def _write_labels(directory: Path, clips: list[_Clip]) -> tuple[Path, ...]:
    paths = []
    for index, clip in enumerate(clips):
        path = directory / f"label-{index:02d}.txt"
        path.write_text(clip.title, encoding="utf-8")
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
    scale = height / 720.0
    left = round(28 * scale)
    filters = [
        f"drawbox=x=0:y=0:w=iw:h={round(119 * scale)}:color=0x030711@0.80:t=fill",
        f"drawbox=x=0:y=h-{round(63 * scale)}:w=iw:h={round(63 * scale)}:"
        "color=0x030711@0.80:t=fill",
        f"drawtext=font='DejaVu Sans':text='ROSClaw Soccer · SIX INDEPENDENT AGENTS · 3v3':"
        f"expansion=none:x={left}:y={round(12 * scale)}:fontsize={round(31 * scale)}:"
        "fontcolor=white",
        f"drawtext=font='DejaVu Sans':text='RED / BLUE - GK · PLAYMAKER · FINISHER   |   "
        "50 Hz NEURAL LOCOMOTION · CPU MUJOCO · SIM ONLY':"
        f"expansion=none:x={left}:y=h-{round(41 * scale)}:fontsize={round(17 * scale)}:"
        "fontcolor=0x8DD8FF",
    ]
    offset = 0.0
    for clip, label in zip(clips, labels, strict=True):
        end = offset + len(clip.times_sec) / fps
        filters.append(
            f"drawtext=font='DejaVu Sans':textfile={escape_filtergraph_option(str(label))}:"
            f"expansion=none:x={left}:y={round(64 * scale)}:fontsize={round(18 * scale)}:"
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
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
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
        "frame_count": int(stream["nb_frames"]),
    }


def validate_independent_team_video_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("independent-team video manifest must be an object")
    expected = payload.pop("manifest_hash", None)
    if expected != hash_json(payload):
        raise ValueError("independent-team video manifest integrity mismatch")
    video = Path(str(payload.get("video_path"))).expanduser().resolve()
    sources = payload.get("source_files")
    if (
        not video.is_file()
        or hash_bytes(video.read_bytes()) != payload.get("video_hash")
        or payload.get("schema_version") != "rosclaw_soccer.independent_team_video.v1"
        or payload.get("claim") != _CLAIM
        or payload.get("whole_body_g1_count") != 6
        or payload.get("red_agent_count") != 3
        or payload.get("blue_agent_count") != 3
        or payload.get("strict_replay") is not True
        or payload.get("visualization_only") is not True
        or payload.get("promotion_eligible") is not False
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
        or payload.get("contact_skill_router_complete") is not False
        or payload.get("commercial_use_allowed") is not False
        or not isinstance(sources, dict)
    ):
        raise ValueError("independent-team video authority contract is invalid")
    for source_name, source_hash in sources.items():
        source = Path(source_name).expanduser().resolve()
        if not source.is_file() or hash_bytes(source.read_bytes()) != source_hash:
            raise ValueError("independent-team video source binding changed")
    payload["manifest_hash"] = expected
    return cast(dict[str, Any], payload)


def _agent_key(agent_id: str) -> str:
    return agent_id.replace(".", "_").replace(":", "_").replace("-", "_")


def _id(mujoco: Any, model: Any, object_type: Any, name: str) -> int:
    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise ValueError(f"independent-team video model is missing {name}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    value = render_independent_team_video(
        evidence_dir=arguments.evidence_dir,
        asset_root=arguments.asset_root,
        output_path=arguments.output,
        fps=arguments.fps,
        width=arguments.width,
        height=arguments.height,
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["render_independent_team_video", "validate_independent_team_video_manifest"]
