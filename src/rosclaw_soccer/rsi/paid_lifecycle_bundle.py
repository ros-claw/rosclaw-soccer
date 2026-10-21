"""Reusable independently audited SIM world reader plus exact phase integration."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.paid_continuity_audit import identical, snapshot_fields
from rosclaw_soccer.rsi.paid_standing_lifecycle_audit import audit_standing_lifecycle


def audit_saved_lifecycle_replica(
    root: Path,
    world: dict[str, Any],
    repeat: int,
    case: dict[str, Any],
    *,
    auditor: Any,
    observer: Any,
    actor: Any,
    replay: Any,
    motion: dict[str, Any],
    schedule: tuple[str, ...],
    pins: dict[str, str],
    count_inference: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    def digest(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    if repeat not in (0, 1) or world["case"] != case["case"]:
        raise ValueError("explicit own case and repeat required")
    folder = root / f"case{case['case']}-repeat{repeat}"
    wp = folder / "complete.json"
    actual = json.loads(wp.read_text())
    pins[str(wp)] = digest(wp)
    artifacts = world["artifacts"] if repeat == 0 else world["repeat_artifacts"]
    if actual["artifacts"] != artifacts or (
        repeat == 0 and actual != {k: v for k, v in world.items() if k != "repeat_artifacts"}
    ):
        raise ValueError("sealed world differs")
    links = {Path(row["raw_path"]).stem: row for row in artifacts}
    if len(links) != len(artifacts) or any(
        (root / row["raw_path"]).parent != folder for row in artifacts
    ):
        raise ValueError("unique confined phase list required")
    rows = [actual["initial"]] + [x[k] for x in actual["kicks"] for k in ("result", "terminal")]
    rows += [x["result"] for x in actual["stands"]]
    rows += [
        row for boundary in actual["boundaries"] for row in boundary.values() if row is not None
    ]
    for row in rows:
        link = links[Path(row["raw_path"]).stem]
        if any(link[k] != row[k] for k in ("raw_path", "raw_hash")):
            raise ValueError("phase link changed")
    raw = {}
    for key, row in links.items():
        path = root / row["raw_path"]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or digest(path) != row["raw_hash"]
        ):
            raise ValueError("exact confined native data required")
        pins[str(path)] = row["raw_hash"]
        with np.load(path, allow_pickle=False) as z:
            raw[key] = {k: z[k].copy() for k in z.files}
    initial = raw["world-initial"]
    q, v = auditor.canonical(initial["qpos"], initial["qvel"])
    if (
        not np.array_equal(q[36:39], case["ball_world_initial"])
        or not np.array_equal(v[35:37], case["velocity_world_initial"])
        or not np.array_equal(initial["mocap_pos"][auditor.goal_mocap_id], case["goal_world"])
    ):
        raise ValueError("declared entry not actually executed")
    result = audit_standing_lifecycle(
        auditor,
        observer,
        actor,
        actual,
        raw,
        motion,
        raw["kick0"],
        variant=2,
        skills=3,
        count_inference=count_inference,
        motor_schedule=schedule,
    )
    phases = []
    for key, z in raw.items():
        if key.startswith("kick") and key[4:].isdigit():
            k = int(key[4:])
            entry = (
                initial
                if k == 0
                else snapshot_fields(raw[f"between{k - 1}-{k}-boundary"], "after_")
            )
            terminal = raw[f"kick-terminal{k}"]
        elif (key.startswith("stand") and key[5:].isdigit()) or key.endswith("-settle"):
            entry, terminal = snapshot_fields(z, "initial_"), snapshot_fields(z, "terminal_")
        else:
            continue
        certificate = replay.replay(z, entry=entry, terminal=terminal)
        phases.append(dict(phase=key, raw_hash=links[key]["raw_hash"], **certificate))
    return result, raw, phases


def require_same_prefix(
    first: dict[str, dict[str, np.ndarray]],
    second: dict[str, dict[str, np.ndarray]],
    *,
    before_kick: int,
) -> None:
    """Compare complete measured prefix including actual-action boundary."""
    if before_kick not in (1, 2):
        raise ValueError("one or two shared-prefix kicks required")
    allowed = {"world-initial"}
    for k in range(before_kick):
        allowed.update(
            {
                f"kick{k}",
                f"kick-terminal{k}",
                f"stand{k}",
                f"between{k}-{k + 1}-drop",
                f"between{k}-{k + 1}-settle",
                f"between{k}-{k + 1}-boundary",
            }
        )
    a, b = set(first) & allowed, set(second) & allowed
    if a != b or not {"world-initial", "kick0", "kick-terminal0"} <= a:
        raise ValueError("shared prefix coverage differs")
    if (
        before_kick == 2
        and ("kick2" in first or "kick2" in second)
        and not {"kick1", "kick-terminal1", "stand1", "between1-2-boundary"} <= a
    ):
        raise ValueError("actual third-entry prefix is incomplete")
    for key in a:
        identical(first[key], second[key])
