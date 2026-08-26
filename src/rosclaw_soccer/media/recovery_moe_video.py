"""Evidence-downstream showcase for a source-scene recovery MoE exam."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.opentrack_teacher_source_exam import (
    validate_opentrack_teacher_source_exam_report,
)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_recovery_moe_video_manifest(path: Path) -> dict[str, Any]:
    """Fail closed when a showcase drifts from its scored replay contract."""

    manifest_path = path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery MoE video manifest must be a JSON object")
    expected_hash = payload.pop("manifest_hash", None)
    if expected_hash != hash_json(payload):
        raise ValueError("recovery MoE video manifest integrity mismatch")
    video_path_value = payload.get("video_path")
    indices = payload.get("selected_state_indices")
    labels = payload.get("labels")
    durations = payload.get("segment_duration_sec")
    if not isinstance(video_path_value, str):
        raise ValueError("recovery MoE video path is invalid")
    video_path = Path(video_path_value).expanduser().resolve()
    finite_values = (
        payload.get("duration_sec"),
        payload.get("fps"),
        payload.get("width"),
        payload.get("height"),
        payload.get("frame_count"),
    )
    if (
        payload.get("schema_version") != "rosclaw.recovery_moe_showcase_video.v1"
        or not video_path.is_file()
        or payload.get("video_hash") != _file_hash(video_path)
        or not isinstance(indices, list)
        or not indices
        or any(not isinstance(index, int) or index < 0 for index in indices)
        or len(set(indices)) != len(indices)
        or not isinstance(labels, list)
        or len(labels) != len(indices)
        or not isinstance(durations, list)
        or len(durations) != len(indices)
        or any(
            not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in (*finite_values, *durations)
        )
        or payload.get("visualization_only") is not True
        or payload.get("pixels_used_for_scoring") is not False
        or payload.get("development_oracle") is not True
        or payload.get("promotion_eligible") is not False
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("recovery MoE video manifest contract is invalid")
    payload["manifest_hash"] = expected_hash
    return cast(dict[str, Any], payload)


def render_recovery_moe_video(
    *,
    exam_path: Path,
    trajectory_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    selected_state_indices: tuple[int, ...] = (0, 2, 6, 7),
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render scored states without changing or re-evaluating physics."""

    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from rosclaw_soccer.world.field import build_g1_stadium_model

    exam_file = exam_path.expanduser().resolve()
    trajectory_file = trajectory_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    report = validate_opentrack_teacher_source_exam_report(exam_file)
    archive_contract = report.get("trajectory_archive")
    if (
        report.get("decision") != "SOURCE_SCENE_TEACHER_REACHABILITY_SUPPORTED"
        or not isinstance(archive_contract, dict)
        or archive_contract.get("hash") != hash_bytes(trajectory_file.read_bytes())
        or archive_contract.get("contains_scored_qpos_only") is not True
    ):
        raise ValueError("recovery MoE video requires bound passing trajectory evidence")
    if (
        output.suffix.lower() != ".mp4"
        or output.exists()
        or output == checkout
        or checkout in output.parents
        or not 1280 <= width <= 3840
        or not 720 <= height <= 2160
        or not selected_state_indices
        or len(set(selected_state_indices)) != len(selected_state_indices)
    ):
        raise ValueError("recovery MoE video output contract is invalid")
    rows = report["rows"]
    if any(not 0 <= index < len(rows) for index in selected_state_indices):
        raise ValueError("recovery MoE video selected state is invalid")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for recovery MoE video")
    with np.load(trajectory_file, allow_pickle=False) as archive:
        if set(archive.files) != {"time", "qpos"}:
            raise ValueError("recovery MoE trajectory arrays are invalid")
        time = np.asarray(archive["time"], dtype=np.float64)
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
    if (
        time.ndim != 1
        or qpos.ndim != 3
        or qpos.shape[:2] != (len(time), len(rows))
        or len(time) != archive_contract.get("frame_count")
        or len(time) < 2
        or not np.all(np.isfinite(qpos))
        or not np.all(np.diff(time) > 0.0)
        or not np.isclose(1.0 / np.median(np.diff(time)), 25.0, atol=1.0e-6)
    ):
        raise ValueError("recovery MoE trajectory shape or timebase is invalid")
    segment_end_times = tuple(
        min(
            float(time[-1]),
            float(time[-1])
            - float(rows[index]["final_continuous_stable_sec"])
            + 5.0,
        )
        for index in selected_state_indices
    )
    segment_frames = tuple(
        int(np.searchsorted(time, end, side="right")) for end in segment_end_times
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    model = build_g1_stadium_model(asset_root.expanduser().resolve())
    if qpos.shape[2] != model.nq:
        raise ValueError("recovery MoE trajectory does not match the source stadium")
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    renderer: Any | None = None
    labels = tuple(
        f"STATE {index}  {rows[index]['posture_cluster']}  ROUTE {rows[index]['entry_frame']}"
        for index in selected_state_indices
    )
    phase_windows: list[tuple[float, float, float]] = []
    offset = 0.0
    for index, end in zip(selected_state_indices, segment_end_times, strict=True):
        capture = float(rows[index]["capture_entry_step"]) * 0.02
        stable = float(time[-1]) - float(rows[index]["final_continuous_stable_sec"])
        phase_windows.append((offset + capture, offset + stable, offset + end))
        offset += end
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
        process = subprocess.Popen(
            _ffmpeg_command(
                ffmpeg=ffmpeg,
                output=output,
                width=width,
                height=height,
                labels=labels,
                segment_end_times=segment_end_times,
                phase_windows=tuple(phase_windows),
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None:
            raise RuntimeError("recovery MoE ffmpeg pipe is unavailable")
        try:
            for index, frame_count in zip(
                selected_state_indices, segment_frames, strict=True
            ):
                _write_segment(
                    mujoco=mujoco,
                    model=model,
                    data=data,
                    renderer=renderer,
                    qpos=qpos[:frame_count, index],
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
            raise RuntimeError(f"recovery MoE ffmpeg failed: {stderr[-3000:]}")
    finally:
        if renderer is not None:
            renderer.close()
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl
    manifest_path = output.with_suffix(".json")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw.recovery_moe_showcase_video.v1",
        "video_path": str(output),
        "video_hash": _file_hash(output),
        "exam_report_hash": report["report_hash"],
        "exam_file_hash": hash_bytes(exam_file.read_bytes()),
        "trajectory_hash": hash_bytes(trajectory_file.read_bytes()),
        "selected_state_indices": list(selected_state_indices),
        "labels": list(labels),
        "segment_duration_sec": list(segment_end_times),
        "frame_count": sum(segment_frames),
        "fps": 25,
        "width": width,
        "height": height,
        "duration_sec": sum(segment_frames) / 25.0,
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "development_oracle": True,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    manifest["manifest_hash"] = hash_json(manifest)
    _atomic_json(manifest_path, manifest)
    return validate_recovery_moe_video_manifest(manifest_path)


def _write_segment(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    qpos: NDArray[np.float64],
    stream: BinaryIO,
) -> None:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.elevation = -8.0
    camera.distance = 3.25
    for frame, state in enumerate(qpos):
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        progress = frame / max(1, len(qpos) - 1)
        camera.lookat[:] = (float(state[0]), float(state[1]), 0.72)
        camera.azimuth = 151.0 + 5.0 * np.sin(2.0 * np.pi * progress)
        camera.distance = 3.25 - 0.20 * np.sin(np.pi * progress)
        renderer.update_scene(data, camera=camera)
        stream.write(np.ascontiguousarray(renderer.render()).tobytes())


def _ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    width: int,
    height: int,
    labels: tuple[str, ...],
    segment_end_times: tuple[float, ...],
    phase_windows: tuple[tuple[float, float, float], ...],
) -> list[str]:
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    filters = [
        "drawbox=x=0:y=0:w=iw:h=112:color=black@0.62:t=fill",
        (
            f"drawtext=fontfile={font}:"
            "text='ROSClaw Athlete Foundation · Recovery MoE':"
            "fontcolor=white:fontsize=40:x=52:y=18"
        ),
        (
            f"drawtext=fontfile={font}:"
            "text='SOURCE STADIUM · SCORED QPOS REPLAY · SIM ONLY':"
            "fontcolor=0x66E0FF:fontsize=24:x=54:y=68"
        ),
    ]
    offset = 0.0
    for label, duration, (capture, stable, end) in zip(
        labels, segment_end_times, phase_windows, strict=True
    ):
        safe = label.replace("'", "")
        filters.append(
            f"drawtext=fontfile={font}:text='{safe}':fontcolor=white:fontsize=30:"
            "x=(w-text_w)/2:y=h-82:box=1:boxcolor=black@0.56:boxborderw=14:"
            f"enable='between(t,{offset:.3f},{offset + duration:.3f})'"
        )
        phases = (
            (offset, capture, "GET-UP EXPERT", "0xFFD166"),
            (capture, stable, "CAPTURE + HANDOFF", "0xFF8C42"),
            (stable, end, "LOCOMOTION READY", "0x5CFF95"),
        )
        for start, stop, text, color in phases:
            filters.append(
                f"drawtext=fontfile={font}:text='{text}':fontcolor={color}:fontsize=28:x=58:y=132:box=1:boxcolor=black@0.48:boxborderw=10:enable='between(t,{start:.3f},{stop:.3f})'"
            )
        offset += duration
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
        "25",
        "-i",
        "-",
        "-vf",
        ",".join(filters),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exam", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    args = parser.parse_args()
    result = render_recovery_moe_video(
        exam_path=args.exam,
        trajectory_path=args.trajectory,
        asset_root=args.asset_root,
        output_path=args.output,
        source_checkout=args.source_checkout,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["render_recovery_moe_video"]
