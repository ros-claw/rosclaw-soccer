"""Read-only R1 evidence import; expose duplicate scenes before RSI training."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contracts import EpisodePartition, PhysicalEpisode
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_SCENE_ID = re.compile(r"s[0-9]{1,8}\Z")


def _receipt(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    commitment = value.pop("manifest_hash", None)
    if commitment != hash_json(value):
        raise ValueError(f"receipt commitment mismatch: {path}")
    return value


def import_r1_parent(evidence_dir: Path) -> dict[str, Any]:
    """Verify archived teacher executions, count unique physics, and mark consumed."""
    root = evidence_dir.expanduser().resolve()
    protocol = _receipt(root / "protocol.json")
    bank = _receipt(root / "bank.json")
    if (
        protocol["training_authorized"]
        or protocol["promoted"]
        or bank["training_authorized"]
        or bank["promoted"]
    ):
        raise ValueError("R1 bank must remain an unpromoted development baseline")
    bindings = protocol["bindings"]
    if not isinstance(bindings, dict) or not isinstance(bindings.get("source_hash"), str):
        raise ValueError("R1 source binding missing")
    scene_rows = protocol["scenes"]
    scenes = {scene["name"]: scene for scene in scene_rows}
    if len(scenes) != len(scene_rows) or any(not _SCENE_ID.fullmatch(name) for name in scenes):
        raise ValueError("R1 protocol has duplicate or unsafe scene identifiers")
    rows = []
    used_scenes: set[str] = set()
    for candidate in bank["bank"]:
        scene_id = candidate["scene"]
        stance = candidate["stance_lateral_m"]
        if (
            scene_id not in scenes
            or scene_id in used_scenes
            or type(stance) not in (int, float)
            or not math.isfinite(stance)
            or abs(stance) > 2
            or not candidate["goals"]
            or not candidate["repeats_identical"]
        ):
            raise ValueError("R1 bank contains an unqualified candidate")
        used_scenes.add(scene_id)
        folder = root / scene_id / f"lat-{stance:+.2f}"
        trials = []
        for label in ("sweep-0", "confirm-1", "confirm-2"):
            stem = folder / label
            receipt = _receipt(stem.with_suffix(".json"))
            physics = _receipt(stem.with_name(label + "-physics.json"))
            trajectory = stem.with_suffix(".npz")
            if (
                receipt["file_hash"] != hash_bytes(trajectory.read_bytes())
                or receipt["physics_file_hash"]
                != hash_bytes(stem.with_name(label + "-physics.json").read_bytes())
                or receipt["scene"] != scene_id
                or receipt["stance_lateral_m"] != stance
                or not receipt["safe"]
                or not receipt["chain_clean"]
                or not receipt["shot"]["in_frame"]
                or receipt["shot"]["keeper_touched"]
                or receipt["shot"] != candidate["shot"]
                or not physics["result"]["safe"]
                or len(physics["rows"]) == 0
            ):
                raise ValueError(
                    f"R1 trajectory, contact chain or goal failed reauthentication: {stem}"
                )
            with np.load(trajectory, allow_pickle=False) as trace:
                poses = {
                    key: trace[key][0].tolist()
                    for key in sorted(trace.files)
                    if key.endswith("_pelvis_pose")
                }
                initial_hash = hash_json({"ball": trace["ball_pose"][0].tolist(), "poses": poses})
            course_hash = hash_json({"scene": scenes[scene_id], "stance": stance})
            episode = PhysicalEpisode(
                episode_id=f"r1.{scene_id}.{label}",
                partition=EpisodePartition.CONSUMED_DEV,
                course_hash=course_hash,
                initial_state_hash=initial_hash,
                policy_hash=hash_json({"source": bindings["source_hash"], "stance": stance}),
                code_hash=bindings["source_hash"],
                environment_hash=hash_json(bindings["kick_assets"]),
                physics_hash=hash_json(
                    {
                        "scenario": physics["result"]["scenario_hash"],
                        "config": physics["result"]["config_hash"],
                    }
                ),
                trajectory_hash=receipt["file_hash"],
                event_hash=receipt["physics_file_hash"],
                safe=True,
                task_success=True,
                teacher_active=True,
            )
            trials.append(episode)
        if (
            len({item.trajectory_hash for item in trials}) != 1
            or len({item.initial_state_hash for item in trials}) != 1
            or len({item.physics_hash for item in trials}) != 1
        ):
            raise ValueError("R1 claimed deterministic repeats differ")
        if list(candidate["evidence_hashes"]) != [item.trajectory_hash for item in trials]:
            raise ValueError("R1 bank evidence hashes differ from authenticated executions")
        rows.append(
            {
                "scene": scene_id,
                "effective_initial_state_hash": trials[0].initial_state_hash,
                "trajectory_hash": trials[0].trajectory_hash,
                "episode": asdict(trials[0]),
                "execution_count": len(trials),
            }
        )
    if not rows:
        raise ValueError("R1 parent bank is empty")
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(row["trajectory_hash"], []).append(row["scene"])
    report = {
        "schema": "rosclaw_soccer.rsi.r1_parent_import.v1",
        "source_protocol_hash": hash_bytes((root / "protocol.json").read_bytes()),
        "source_bank_hash": hash_bytes((root / "bank.json").read_bytes()),
        "partition": EpisodePartition.CONSUMED_DEV.value,
        "teacher_active": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "declared_scene_count": len(rows),
        "physical_execution_count": sum(row["execution_count"] for row in rows),
        "unique_physical_trajectory_count": len(groups),
        "effective_scenario_groups": sorted(groups.values()),
        "episodes": rows,
    }
    report["report_hash"] = hash_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = import_r1_parent(args.evidence_dir)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "declared_scene_count",
                    "physical_execution_count",
                    "unique_physical_trajectory_count",
                    "effective_scenario_groups",
                    "report_hash",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
