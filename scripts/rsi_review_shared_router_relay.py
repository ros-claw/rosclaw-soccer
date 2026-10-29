"""Review exact own-replayed paired router physical trajectories, SIM only."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_shared_cftrack_relay import PHYSICAL_FIELDS, compare_prefix, outcome


def bind_world(root, row, name):
    item = row["artifacts"].get(name)
    if item is None:
        return None
    path = root / item["path"]
    if (
        path.is_symlink()
        or path.resolve().parent != (root / f"trial{row['index']}").resolve()
        or digest(path) != item["hash"]
    ):
        raise ValueError("exact confined world artifact required")
    return path


def admit(source, replay, source_hash, replay_hash):
    paths = (source / "complete.json", replay / "complete.json")
    if (
        any(p.is_symlink() for p in paths)
        or digest(paths[0]) != source_hash
        or digest(paths[1]) != replay_hash
        or (source / "failed.json").exists()
        or (replay / "failed.json").exists()
    ):
        raise ValueError("exact successful source and own replay required")
    old, new = [json.loads(p.read_text()) for p in paths]
    if (
        old["status"] != "ROUTER_SHARED_RELAY_SOURCE_PENDING_REPLAY"
        or new["status"] != "ROUTER_SHARED_RELAY_OWN_REPLAY_EXACT"
        or new["source_hash"] != source_hash
        or old["source_head"] != new["source_head"]
        or old["worlds"] != 16
        or new["worlds"] != 16
        or len(old["records"]) != 16
        or len(new["records"]) != 16
        or any(
            r.get(k) is not False
            for r in (old, new)
            for k in ("hardware_authorized", "promotion_authorized", "team_game_verified")
        )
    ):
        raise ValueError("matching unpromoted 16-world replay required")
    declarations = []
    for root, report in ((source, old), (replay, new)):
        path = root / "declaration.json"
        if path.is_symlink() or digest(path) != report["declaration_hash"]:
            raise ValueError("source-bound declarations required")
        declarations.append(json.loads(path.read_text()))
    if (
        declarations[0]["input_file_hashes"] != declarations[1]["input_file_hashes"]
        or declarations[0]["protocol_hash"] != declarations[1]["protocol_hash"]
    ):
        raise ValueError("source/replay input commitments differ")
    for i, (x, y) in enumerate(zip(old["records"], new["records"], strict=True)):
        if (
            x["index"] != i
            or y["index"] != i
            or {k: v for k, v in x.items() if k != "artifacts"}
            != {k: v for k, v in y.items() if k != "artifacts"}
            or set(x["artifacts"]) != set(y["artifacts"])
        ):
            raise ValueError("source/replay world metadata differs")
        for name in x["artifacts"]:
            xp, yp = bind_world(source, x, name), bind_world(replay, y, name)
            with np.load(xp, allow_pickle=False) as a, np.load(yp, allow_pickle=False) as b:
                if set(a.files) != set(b.files) or any(
                    not np.array_equal(a[k], b[k]) for k in a.files
                ):
                    raise ValueError("source/replay actual histories differ")
    return old, declarations[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "replay", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("source-hash", "replay-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external review required")
    source, declaration = admit(a.source, a.replay, a.source_hash, a.replay_hash)
    pins = declaration["input_file_hashes"]
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1 or digest(scenes[0]) != pins[str(scenes[0])]:
        raise ValueError("exact source-bound standard body required")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/", "b/"))
    )
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    rows = []
    for i in range(0, 16, 2):
        base, candidate = source["records"][i : i + 2]
        if (
            base["arm"] != "dual"
            or candidate["arm"] != "router"
            or any(base[k] != candidate[k] for k in ("context", "lateral_m"))
        ):
            raise ValueError("fixed paired roles and contexts required")
        ba, ca = base["handoff"], candidate["handoff"]
        if (ba is None) != (ca is None) or (ba and ba["control"] != ca["control"]):
            raise ValueError("same causal readiness required")
        prefix = ba["control"] if ba else min(base["controls"], candidate["controls"])
        with (
            np.load(bind_world(a.source, base, "physics"), allow_pickle=False) as x,
            np.load(bind_world(a.source, candidate, "physics"), allow_pickle=False) as y,
        ):
            compare_prefix(x, y, PHYSICAL_FIELDS, prefix)
        common = min(base["controls"], candidate["controls"])
        rows.append(
            dict(
                context=base["context"],
                lateral_m=base["lateral_m"],
                prefix_controls=prefix,
                base_termination=base["termination"],
                candidate_termination=candidate["termination"],
                base_full=outcome(a.source, base, base["controls"], model, layout),
                candidate_full=outcome(a.source, candidate, candidate["controls"], model, layout),
                base_common=outcome(
                    a.source, base, common, model, layout, proximity_start_control=prefix
                ),
                candidate_common=outcome(
                    a.source, candidate, common, model, layout, proximity_start_control=prefix
                ),
                candidate_neural_controls=candidate["controls"] - ba["control"] if ba else 0,
            )
        )
    clean(a.expected_head)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="PAIRED_ROUTER_REPLAYED_DEVELOPMENT_ONLY",
            source_hash=a.source_hash,
            replay_hash=a.replay_hash,
            code_head=a.expected_head,
            source_head=source["source_head"],
            protocol_hash=declaration["protocol_hash"],
            compiled_scene_hash=compiled_model_hash(model),
            rows=rows,
            hardware_authorized=False,
            promotion_authorized=False,
            team_game_verified=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
