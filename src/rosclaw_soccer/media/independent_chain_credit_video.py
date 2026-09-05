"""Render the S201 physical-pass failure and role-local learning decision."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

from rosclaw_soccer.media.independent_option_video import (
    validate_independent_option_video_manifest,
)
from rosclaw_soccer.media.trajectory_render import escape_filtergraph_option
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.independent_chain_credit_growth import (
    validate_independent_chain_credit_growth,
)

_CLAIM = "S201_ROLE_LOCAL_PASS_FAILURE_CREDIT_VISUALIZATION"


def render_independent_chain_credit_video(
    *,
    evidence_dir: Path,
    source_video_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    root = evidence_dir.expanduser().resolve()
    source_video = source_video_path.expanduser().resolve()
    source_manifest_path = source_video.with_suffix(".json")
    output = output_path.expanduser().resolve()
    manifest_path = output.with_suffix(".json")
    if output.exists() or manifest_path.exists() or output.suffix.lower() != ".mp4":
        raise ValueError("S201 video output must be a new MP4 path")
    report_path = root / "chain-credit-exam.json"
    report = validate_independent_chain_credit_growth(report_path)
    source_manifest = validate_independent_option_video_manifest(source_manifest_path)
    source = cast(dict[str, Any], report["source"])
    if (
        not source_video.is_file()
        or hash_bytes(source_video.read_bytes()) != source_manifest.get("video_hash")
        or source_manifest.get("source_report_hash") != source.get("report_hash")
    ):
        raise ValueError("S201 source video is not bound to its physical evidence")
    assessment = cast(dict[str, Any], report["assessment"])
    event = cast(dict[str, Any], report["pass_event"])
    actor = cast(dict[str, Any], report["actor"])
    if (
        assessment.get("earliest_failure") != "pass_inaccurate"
        or assessment.get("focal_agent_id") != "red.playmaker"
        or actor.get("deployment_ready") is not False
    ):
        raise ValueError("S201 video requires the preliminary playmaker failure route")

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for S201 video")
    output.parent.mkdir(parents=True, exist_ok=True)
    target_error = float(event["target_error_m"])
    speed = float(event["post_contact_ball_speed_mps"])
    residual = cast(list[float], actor["residual_mean_m"])
    with tempfile.TemporaryDirectory(prefix="rosclaw-s201-video-") as temporary:
        tmp = Path(temporary)
        title = tmp / "title.txt"
        metric = tmp / "metric.txt"
        credit = tmp / "credit.txt"
        boundary = tmp / "boundary.txt"
        title.write_text("ROSClaw Growth · CAUSAL FAILURE CREDIT", encoding="utf-8")
        metric.write_text(
            f"Physical contact: PASS · {speed:.2f} m/s   |   delivery error: {target_error:.3f} m",
            encoding="utf-8",
        )
        credit.write_text(
            "Earliest failure → red.playmaker PLASTIC · other 5 cells FROZEN",
            encoding="utf-8",
        )
        boundary.write_text(
            "Learned aim residual: "
            f"({residual[0]:+.2f}, {residual[1]:+.2f}, {residual[2]:+.2f}) m · "
            "1 sample → NOT DEPLOYABLE",
            encoding="utf-8",
        )
        filters = (
            "drawbox=x=0:y=0:w=iw:h=150:color=0x030711:t=fill,"
            "drawbox=x=0:y=950:w=iw:h=130:color=0x030711:t=fill,"
            f"drawtext=font='DejaVu Sans':textfile={escape_filtergraph_option(str(title))}:"
            "expansion=none:x=36:y=18:fontsize=34:fontcolor=white,"
            f"drawtext=font='DejaVu Sans':textfile={escape_filtergraph_option(str(metric))}:"
            "expansion=none:x=36:y=67:fontsize=22:fontcolor=0x8DD8FF,"
            f"drawtext=font='DejaVu Sans':textfile={escape_filtergraph_option(str(credit))}:"
            "expansion=none:x=36:y=105:fontsize=22:fontcolor=0x65F59A,"
            f"drawtext=font='DejaVu Sans':textfile={escape_filtergraph_option(str(boundary))}:"
            "expansion=none:x=36:y=985:fontsize=19:fontcolor=0xFFD166"
        )
        completed = subprocess.run(
            (
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source_video),
                "-t",
                "9.55",
                "-vf",
                filters,
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
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(f"S201 ffmpeg failed: {completed.stderr[-3000:]}")
    probe = _probe(ffprobe, output)
    if probe["width"] != 1920 or probe["height"] != 1080 or probe["duration_sec"] < 9.0:
        raise RuntimeError("S201 video encoding contract changed")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.independent_chain_credit_video.v1",
        "claim": _CLAIM,
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_video_hash": source_manifest["video_hash"],
        "source_video_manifest_hash": source_manifest["manifest_hash"],
        "source_report_hash": report["report_hash"],
        "width": probe["width"],
        "height": probe["height"],
        "duration_sec": probe["duration_sec"],
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "continuous_chain_complete": False,
        "actor_deployment_ready": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
    }
    manifest["manifest_hash"] = hash_json(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_independent_chain_credit_video_manifest(manifest_path)
    return manifest


def validate_independent_chain_credit_video_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S201 video manifest must be an object")
    expected = payload.pop("manifest_hash", None)
    try:
        video = Path(str(payload.get("video_path"))).expanduser().resolve()
        if (
            expected != hash_json(payload)
            or not video.is_file()
            or hash_bytes(video.read_bytes()) != payload.get("video_hash")
            or payload.get("claim") != _CLAIM
            or payload.get("width") != 1920
            or payload.get("height") != 1080
            or payload.get("visualization_only") is not True
            or payload.get("promotion_eligible") is not False
            or payload.get("continuous_chain_complete") is not False
            or payload.get("actor_deployment_ready") is not False
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("commercial_use_allowed") is not False
        ):
            raise ValueError("S201 video manifest integrity or authority contract is invalid")
    finally:
        if expected is not None:
            payload["manifest_hash"] = expected
    return payload


def _probe(ffprobe: str, path: Path) -> dict[str, float | int]:
    completed = subprocess.run(
        (
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    stream = value["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration_sec": float(value["format"]["duration"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = render_independent_chain_credit_video(
        evidence_dir=args.evidence_dir,
        source_video_path=args.source_video,
        output_path=args.output,
    )
    print(json.dumps({"video_hash": manifest["video_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "render_independent_chain_credit_video",
    "validate_independent_chain_credit_video_manifest",
]
