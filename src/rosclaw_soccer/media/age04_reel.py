"""Build the Claw-7 Academy Age-4 hero reel from verified physics media."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw_soccer.evidence.age04 import (
    EvidenceCase,
    ReelSegment,
    load_age04_manifest,
    validate_age04_manifest,
)

_WIDTH = 1920
_HEIGHT = 1080
_FPS = 30
_TITLE_DURATION_SEC = 4.0
_FINALE_DURATION_SEC = 4.5


@dataclass(frozen=True)
class Age04ReelResult:
    output_path: str
    output_hash: str
    media_manifest_path: str
    width: int
    height: int
    fps: int
    duration_sec: float
    source_case_ids: tuple[str, ...]
    certified_segment_count: int
    development_segment_count: int
    validation_manifest_hash: str
    activation_ceiling: str = "SIM_ONLY"
    physics_authority: str = "SOURCE_EVIDENCE_ONLY"
    pixels_used_for_promotion: bool = False
    schema_version: str = "rosclaw_soccer.age04_reel.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_age04_reel(
    *,
    manifest_path: Path,
    evidence_root: Path,
    output_path: Path,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> Age04ReelResult:
    """Verify evidence, render a coherent 1080p reel, and bind its derivative hash."""

    manifest_file = manifest_path.expanduser().resolve()
    validation = validate_age04_manifest(manifest_file, evidence_root)
    manifest = load_age04_manifest(manifest_file)
    output = output_path.expanduser().resolve()
    project_root = _project_root(manifest_file)
    if output == project_root or project_root in output.parents:
        raise ValueError("full Age-4 MP4 output must stay outside the source repository")
    ffmpeg_path = shutil.which(ffmpeg)
    ffprobe_path = shutil.which(ffprobe)
    if ffmpeg_path is None or ffprobe_path is None:
        raise RuntimeError("ffmpeg and ffprobe are required for the Age-4 reel")
    cases = {case.case_id: case for case in manifest.cases}
    validated = {case.case_id: case for case in validation.cases}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rosclaw-soccer-age04-") as temporary:
        temp = Path(temporary)
        pieces: list[Path] = []
        pieces.append(
            _render_card(
                ffmpeg_path,
                temp,
                index=0,
                title="CLAW-7  •  ACADEMY AGE 4",
                subtitle="THE FIRST FOOTBALLER",
                footer="EVIDENCE-BOUND  •  STRICT REPLAY  •  SIM ONLY",
                duration_sec=_TITLE_DURATION_SEC,
                accent="38E6A5",
            )
        )
        for index, segment in enumerate(manifest.reel, start=1):
            case = cases[segment.case_id]
            source = Path(validated[case.case_id].media_path)
            source_duration = _probe_duration(ffprobe_path, source)
            if segment.start_sec + segment.duration_sec > source_duration + 0.05:
                raise ValueError(f"{segment.case_id} reel segment exceeds source duration")
            pieces.append(
                _render_segment(
                    ffmpeg_path,
                    temp,
                    index=index,
                    source=source,
                    case=case,
                    segment=segment,
                )
            )
        pieces.append(
            _render_card(
                ffmpeg_path,
                temp,
                index=len(pieces),
                title="AGE 4 IS A BEGINNING",
                subtitle="NEXT: FIRST TOUCH  •  CONTROL  •  CONTINUOUS FOOTBALL",
                footer="PRACTICE  →  LEARN  →  PROVE  →  GROW",
                duration_sec=_FINALE_DURATION_SEC,
                accent="F5C451",
            )
        )
        concat_file = temp / "concat.txt"
        concat_file.write_text(
            "".join(f"file '{piece.as_posix()}'\n" for piece in pieces),
            encoding="utf-8",
        )
        _run(
            [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output),
            ],
            "Age-4 reel assembly",
        )
    duration = _probe_duration(ffprobe_path, output)
    result = Age04ReelResult(
        output_path=str(output),
        output_hash=_hash_file(output),
        media_manifest_path=str(output.with_suffix(".manifest.json")),
        width=_WIDTH,
        height=_HEIGHT,
        fps=_FPS,
        duration_sec=duration,
        source_case_ids=tuple(dict.fromkeys(segment.case_id for segment in manifest.reel)),
        certified_segment_count=sum(
            cases[segment.case_id].verdict.value == "PASS" for segment in manifest.reel
        ),
        development_segment_count=sum(
            cases[segment.case_id].verdict.value == "DEVELOPMENT" for segment in manifest.reel
        ),
        validation_manifest_hash=validation.manifest_hash,
    )
    derivative_manifest = {
        **result.to_dict(),
        "source_validation": validation.to_dict(),
        "render_only": True,
        "physics_recomputed": False,
        "certification_authority": False,
    }
    Path(result.media_manifest_path).write_text(
        json.dumps(derivative_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _render_card(
    ffmpeg: str,
    temp: Path,
    *,
    index: int,
    title: str,
    subtitle: str,
    footer: str,
    duration_sec: float,
    accent: str,
) -> Path:
    output = temp / f"{index:02d}-card.mp4"
    title_file = _text_file(temp, f"{index:02d}-title", title)
    subtitle_file = _text_file(temp, f"{index:02d}-subtitle", subtitle)
    footer_file = _text_file(temp, f"{index:02d}-footer", footer)
    font = _font_file()
    fade_out = max(0.0, duration_sec - 0.45)
    filters = (
        f"drawbox=x=0:y=0:w=iw:h=ih:color=0x07111D:t=fill,"
        f"drawbox=x=260:y=330:w=1400:h=5:color=0x{accent}:t=fill,"
        f"drawtext=fontfile='{font}':textfile='{title_file}':fontcolor=white:"
        "fontsize=70:x=(w-text_w)/2:y=395,"
        f"drawtext=fontfile='{font}':textfile='{subtitle_file}':fontcolor=0x{accent}:"
        "fontsize=34:x=(w-text_w)/2:y=505,"
        f"drawtext=fontfile='{font}':textfile='{footer_file}':fontcolor=0xAAB7C4:"
        "fontsize=24:x=(w-text_w)/2:y=780,"
        f"fade=t=in:st=0:d=0.45,fade=t=out:st={fade_out:.3f}:d=0.45"
    )
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x07111D:s={_WIDTH}x{_HEIGHT}:r={_FPS}:d={duration_sec}",
            "-vf",
            filters,
            "-t",
            f"{duration_sec:.3f}",
            *_encode_args(),
            str(output),
        ],
        "Age-4 title card",
    )
    return output


def _render_segment(
    ffmpeg: str,
    temp: Path,
    *,
    index: int,
    source: Path,
    case: EvidenceCase,
    segment: ReelSegment,
) -> Path:
    output = temp / f"{index:02d}-{case.case_id}.mp4"
    headline_file = _text_file(
        temp,
        f"{index:02d}-headline",
        f"{segment.chapter}  |  {segment.subtitle}",
    )
    status = (
        "CERTIFIED AGE-4 PHYSICS  •  STRICT REPLAY  •  SIM ONLY"
        if case.verdict.value == "PASS"
        else "DEVELOPMENT  •  NOT CERTIFIED  •  SIM ONLY"
    )
    status_file = _text_file(temp, f"{index:02d}-status", status)
    accent = "38E6A5" if case.verdict.value == "PASS" else "F5A742"
    font = _font_file()
    fade_out = max(0.0, segment.duration_sec - 0.40)
    filters = (
        "crop=1920:1000:0:55,scale=1920:1080:flags=lanczos,setsar=1,"
        "setpts=PTS-STARTPTS,"
        "eq=contrast=1.03:saturation=1.04:brightness=-0.005,"
        "unsharp=5:5:0.22:5:5:0.0,"
        "drawbox=x=0:y=0:w=iw:h=118:color=0x07111D:t=fill,"
        f"drawbox=x=0:y=114:w=iw:h=4:color=0x{accent}:t=fill,"
        f"drawtext=fontfile='{font}':textfile='{headline_file}':fontcolor=0xFFFFFF:"
        "fontsize=30:x=58:y=44,"
        "drawbox=x=0:y=ih-112:w=iw:h=112:color=0x07111D:t=fill,"
        f"drawtext=fontfile='{font}':textfile='{status_file}':fontcolor=0xD7E0E9:"
        "fontsize=21:x=(w-text_w)/2:y=h-61,"
        f"fade=t=in:st=0:d=0.30,fade=t=out:st={fade_out:.3f}:d=0.40"
    )
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{segment.start_sec:.3f}",
            "-i",
            str(source),
            "-t",
            f"{segment.duration_sec:.3f}",
            "-vf",
            filters,
            "-an",
            *_encode_args(),
            str(output),
        ],
        f"Age-4 segment {case.case_id}",
    )
    return output


def _encode_args() -> list[str]:
    return [
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level:v",
        "4.1",
        "-r",
        str(_FPS),
        "-g",
        str(_FPS * 2),
        "-movflags",
        "+faststart",
    ]


def _text_file(temp: Path, stem: str, value: str) -> str:
    path = temp / f"{stem}.txt"
    path.write_text(value, encoding="utf-8")
    return path.as_posix()


def _font_file() -> str:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.as_posix()
    raise RuntimeError("a supported local font is required for the Age-4 reel")


def _probe_duration(ffprobe: str, path: Path) -> float:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
    duration = float(completed.stdout.strip())
    if duration <= 0.0:
        raise ValueError(f"media has an invalid duration: {path}")
    return duration


def _run(command: Sequence[str], label: str) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed: {completed.stderr.strip()}")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _project_root(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError("Age-4 manifest is not inside a ROSClaw Soccer checkout")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=(
            Path(os.environ["ROSCLAW_SOCCER_EVIDENCE"])
            if "ROSCLAW_SOCCER_EVIDENCE" in os.environ
            else None
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.evidence_root is None:
        raise SystemExit("--evidence-root or ROSCLAW_SOCCER_EVIDENCE is required")
    result = build_age04_reel(
        manifest_path=args.manifest,
        evidence_root=args.evidence_root,
        output_path=args.output,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Age04ReelResult", "build_age04_reel", "main"]
