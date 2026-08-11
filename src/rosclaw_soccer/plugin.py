"""ROSClaw CLI extension adapter; contains no independent runtime."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from rosclaw_soccer.academy.player_card import load_player_card
from rosclaw_soccer.evidence.age04 import validate_age04_manifest
from rosclaw_soccer.media.age04_reel import build_age04_reel
from rosclaw_soccer.physics.reality_pack import run_reality_pack
from rosclaw_soccer.training.age04_regulation import (
    Age04RegulationAssets,
    run_age04_regulation_training,
)


def register_cli(subparsers: Any) -> None:
    """Register the downstream ``rosclaw soccer`` command tree."""

    soccer = subparsers.add_parser("soccer", help="ROSClaw Soccer Academy")
    commands = soccer.add_subparsers(dest="soccer_command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect the local SIM_ONLY academy setup")
    doctor.add_argument("--evidence-root", type=Path)
    doctor.set_defaults(rosclaw_extension_handler=_doctor)

    academy = commands.add_parser("academy", help="Inspect academy progression")
    academy_commands = academy.add_subparsers(dest="academy_command", required=True)
    academy_status = academy_commands.add_parser("status", help="Show the current flagship age")
    academy_status.set_defaults(rosclaw_extension_handler=_academy_status)
    train_age04 = academy_commands.add_parser(
        "train-age04", help="Train a fresh Age-4 actor under regulation physics"
    )
    train_age04.add_argument("--asset-root", type=Path, required=True)
    train_age04.add_argument("--gait-policy-root", type=Path, required=True)
    train_age04.add_argument("--sonic-model-root", type=Path, required=True)
    train_age04.add_argument("--seed-request", type=Path, required=True)
    train_age04.add_argument("--approach-strike-candidate", type=Path, required=True)
    train_age04.add_argument(
        "--football-motion-prior",
        type=Path,
        help="optional MotionDecode prior; the stability-first curriculum defaults to no blend",
    )
    train_age04.add_argument("--output-dir", type=Path, required=True)
    train_age04.add_argument("--source-checkout", type=Path, required=True)
    train_age04.set_defaults(rosclaw_extension_handler=_train_age04)

    player = commands.add_parser("player", help="Inspect a player growth card")
    player_commands = player.add_subparsers(dest="player_command", required=True)
    player_show = player_commands.add_parser("show", help="Show one player")
    player_show.add_argument("player_id")
    player_show.set_defaults(rosclaw_extension_handler=_player_show)

    media = commands.add_parser("media", help="Build evidence-downstream media")
    media_commands = media.add_subparsers(dest="media_command", required=True)
    age04 = media_commands.add_parser("age04", help="Build the verified Age-4 hero reel")
    age04.add_argument("--manifest", type=Path, default=_default_manifest_path())
    age04.add_argument("--evidence-root", type=Path)
    age04.add_argument("--output", type=Path, required=True)
    age04.set_defaults(rosclaw_extension_handler=_build_age04_media)

    physics = commands.add_parser("physics", help="Validate football physics before training")
    physics_commands = physics.add_subparsers(dest="physics_command", required=True)
    benchmark = physics_commands.add_parser(
        "benchmark", help="Run the CPU MuJoCo Soccer Reality Pack"
    )
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.set_defaults(rosclaw_extension_handler=_physics_benchmark)


def _doctor(args: Any) -> int:
    evidence_root = _evidence_root(args.evidence_root)
    report: dict[str, Any] = {
        "project": "ROSClaw Soccer Academy",
        "activation_ceiling": "SIM_ONLY",
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "evidence_root": None if evidence_root is None else str(evidence_root),
        "age04_evidence_valid": False,
    }
    if evidence_root is not None:
        validation = validate_age04_manifest(_default_manifest_path(), evidence_root)
        report["age04_evidence_valid"] = validation.passed
        report["age04_manifest_hash"] = validation.manifest_hash
    report["ready"] = bool(
        report["ffmpeg"] and report["ffprobe"] and report["age04_evidence_valid"]
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 2


def _academy_status(_args: Any) -> int:
    player = load_player_card(_default_player_path())
    print(
        json.dumps(
            {
                "flagship_player": player.player_id,
                "academy_age": player.academy_age,
                "age_title": player.age_title,
                "certification_status": player.certification_status,
                "next_exam": player.next_exam,
                "activation_ceiling": player.activation_ceiling,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _train_age04(args: Any) -> int:
    report = run_age04_regulation_training(
        assets=Age04RegulationAssets(
            asset_root=args.asset_root,
            gait_policy_root=args.gait_policy_root,
            sonic_model_root=args.sonic_model_root,
            seed_request=args.seed_request,
            approach_strike_candidate=args.approach_strike_candidate,
            football_motion_prior=args.football_motion_prior,
        ),
        output_dir=args.output_dir,
        source_checkout=args.source_checkout,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 2


def _player_show(args: Any) -> int:
    if args.player_id.lower().replace("-", "") != "claw7":
        raise ValueError("the bootstrap academy currently contains only Claw-7")
    player = load_player_card(_default_player_path())
    print(json.dumps(player.to_dict(), indent=2, sort_keys=True))
    return 0


def _build_age04_media(args: Any) -> int:
    evidence_root = _evidence_root(args.evidence_root)
    if evidence_root is None:
        raise ValueError("--evidence-root or ROSCLAW_SOCCER_EVIDENCE is required")
    result = build_age04_reel(
        manifest_path=args.manifest,
        evidence_root=evidence_root,
        output_path=args.output,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _physics_benchmark(args: Any) -> int:
    report = run_reality_pack(
        output_dir=args.output_dir,
        source_checkout=_repository_root(),
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 2


def _evidence_root(argument: Path | None) -> Path | None:
    if argument is not None:
        return argument.expanduser().resolve()
    configured = os.environ.get("ROSCLAW_SOCCER_EVIDENCE")
    return None if configured is None else Path(configured).expanduser().resolve()


def _repository_root() -> Path:
    configured = os.environ.get("ROSCLAW_SOCCER_REPO")
    if configured is not None:
        return Path(configured).expanduser().resolve()
    candidate = Path(__file__).resolve().parents[2]
    if not (candidate / "pyproject.toml").is_file():
        raise RuntimeError("set ROSCLAW_SOCCER_REPO for this installation")
    return candidate


def _default_manifest_path() -> Path:
    return _repository_root() / "evidence" / "manifests" / "age04.json"


def _default_player_path() -> Path:
    return _repository_root() / "academy" / "players" / "claw7.json"


__all__ = ["register_cli"]
