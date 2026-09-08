"""Render complete baseline and active-team episodes at real-time speed."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from rosclaw_soccer.media.continuous_competitive_match_video import (
    _Clip,
    _write_frames,
    _write_labels,
)
from rosclaw_soccer.media.dynamic_strike_coordination_video import _ffmpeg_command, _play
from rosclaw_soccer.media.independent_team_video import _color_player
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.active_team_probe import validate_probe
from rosclaw_soccer.training.continuous_competitive_match_growth import (
    build_continuous_competitive_fixture,
)
from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model


def render(*, sources: tuple[Path, ...], asset_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    reports = [validate_probe(p) for p in sources]
    os.environ["MUJOCO_GL"] = "egl"
    import mujoco

    fixture = build_continuous_competitive_fixture(asset_root)
    model = build_g1_multi_player_stadium_model(
        asset_root, players=fixture.players, spec=fixture.goal
    )
    for player in fixture.players:
        _color_player(
            model,
            root_body_name=player.body_prefix + "pelvis",
            rgba=(0.88, 0.05, 0.04, 1.0)
            if player.agent_id.startswith("red.")
            else (0.02, 0.20, 0.92, 1.0),
        )
    model.vis.global_.offwidth = 1920
    model.vis.global_.offheight = 1080
    clips, traces = [], []
    for source, report in zip(sources, reports, strict=True):
        with np.load(source.parent / "primary.npz", allow_pickle=False) as archive:
            trajectory = {k: archive[k] for k in archive.files}
        mode = "ACTIVE TEAM" if report["active_competition"] else "BASELINE"
        state = "SAFE" if report["results"][0]["safe"] else "SAFETY GATE FAILED"
        clips.append(
            _Clip(
                f"{mode} | {state} | FULL EPISODE - REAL TIME",
                _play(
                    float(trajectory["time"][0]),
                    float(trajectory["time"][-1]),
                    1.0,
                    30,
                    "broadcast",
                ),
            )
        )
        traces.append(trajectory)
    with tempfile.TemporaryDirectory(prefix="s211-labels-") as directory:
        labels = _write_labels(Path(directory), tuple(clips))
        command = _ffmpeg_command(
            ffmpeg="ffmpeg",
            output=output,
            width=1920,
            height=1080,
            fps=30,
            clips=tuple(clips),
            labels=labels,
        )
        # Reuse encoding/camera primitives with this experiment's factual titles.
        command = [
            item.replace("S209 LEARNED QUICK STRIKE", "S211 ACTIVE TEAM")
            .replace("DATA-BOUND ACTOR", "ROLE OBJECTIVES")
            .replace(
                "PASS  |  RECEIVE  |  QUICK STRIKE  |  PHYSICAL SHIN BLOCK  |  RECOVERY",
                "COMPLETE EPISODES | PHYSICAL MOTION | DEVELOPMENT ONLY",
            )
            for item in command
        ]
        with mujoco.Renderer(model, height=1080, width=1920) as renderer:
            data = mujoco.MjData(model)
            with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
                assert process.stdin is not None
                try:
                    for trajectory, clip in zip(traces, clips, strict=True):
                        _write_frames(
                            mujoco=mujoco,
                            model=model,
                            data=data,
                            renderer=renderer,
                            players=fixture.players,
                            trajectory=trajectory,
                            clips=(clip,),
                            stream=cast(BinaryIO, process.stdin),
                        )
                finally:
                    process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError("video encoding failed")
    manifest = {
        "sources": {str(p): hash_bytes(p.read_bytes()) for p in sources},
        "video_hash": hash_bytes(output.read_bytes()),
        "fps": 30,
        "width": 1920,
        "height": 1080,
        "playback_speed": 1.0,
        "complete_episodes": True,
        "promotion_eligible": False,
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
    }
    manifest["manifest_hash"] = hash_json(manifest)
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(render(sources=tuple(args.source), asset_root=args.asset_root, output=args.output))


if __name__ == "__main__":
    main()
