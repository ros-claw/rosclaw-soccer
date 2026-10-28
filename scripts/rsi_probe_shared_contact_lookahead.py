"""Bounded multi-body scratch-physics authority probe; not a team rollout."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.shared_contact_lookahead import lookahead_candidates
from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_fresh_common import clean


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("snapshots", "scene", "velocity-config", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    for key in ("snapshot-hash", "scene-hash", "velocity-config-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    receipt = a.snapshots / "complete.json"
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or digest(receipt) != a.snapshot_hash
        or digest(a.scene) != a.scene_hash
        or digest(a.velocity_config) != a.velocity_config_hash
    ):
        raise ValueError("pinned snapshot, scene, config and unused external output required")
    report = json.loads(receipt.read_text())
    if (
        report["scene_hash"] != a.scene_hash
        or report["promotion_authorized"] is not False
        or report["saved_torque_is_not_policy"] is not True
    ):
        raise ValueError("unpromoted measured shared-world snapshot required")
    ids = (0, 4, 8, 12)
    rows = {row["index"]: row for row in report["rows"]}
    if not set(ids).issubset(rows):
        raise ValueError("complete declared source snapshot indices required")
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    params = VelocityPolicyParameters.load(a.velocity_config, expected_hash=a.velocity_config_hash)
    pd = NativePDParameters(
        np.asarray(params.stiffness),
        np.asarray(params.damping),
        0.85 * np.asarray(G1_HARD_TORQUE_LIMITS),
    )
    layouts = {
        p: NativeVelocityLayout(model, robot_prefix=p, ball_body_name="soccer_ball", player_count=2)
        for p in ("a/", "b/")
    }
    corrections = np.zeros((9, 29))
    for index, joint in enumerate((0, 3, 6, 9)):
        corrections[1 + 2 * index, joint] = 0.08
        corrections[2 + 2 * index, joint] = -0.08
    results = []
    for index in ids:
        source = a.snapshots / rows[index]["snapshot"]
        if (
            source.is_symlink()
            or source.resolve().parent != a.snapshots.resolve()
            or digest(source) != rows[index]["snapshot_hash"]
        ):
            raise ValueError("exact confined historical snapshot required")
        data = mujoco.MjData(model)
        with np.load(source, allow_pickle=False) as saved:
            mujoco.mj_setState(model, data, saved["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        targets = {prefix: data.qpos[layout.qi].copy() for prefix, layout in layouts.items()}
        candidates = lookahead_candidates(
            model,
            data,
            players={"a/": pd, "b/": pd},
            ball_body_name="soccer_ball",
            targets=targets,
            receiver="b/",
            corrections=corrections,
            horizon_controls=30,
        )
        results.append(
            dict(index=index, snapshot_hash=rows[index]["snapshot_hash"], candidates=candidates)
        )
        if digest(source) != rows[index]["snapshot_hash"]:
            raise ValueError("snapshot changed while forecasting")
    if (
        digest(receipt) != a.snapshot_hash
        or digest(a.scene) != a.scene_hash
        or digest(a.velocity_config) != a.velocity_config_hash
    ):
        raise ValueError("inputs changed during forecast")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="SHARED_HELD_TARGET_PHYSICAL_PROBE_ONLY",
            source_head=a.expected_head,
            snapshot_hash=a.snapshot_hash,
            scene_hash=a.scene_hash,
            velocity_config_hash=a.velocity_config_hash,
            indices=list(ids),
            results=results,
            correction_vectors=corrections.tolist(),
            horizon_controls=30,
            both_players_hold_initial_measured_joint_positions=True,
            future_controls_read=False,
            new_neural_calls=0,
            optimizer_updates=0,
            independent_fresh=False,
            physical_teacher_qualified=False,
            promotion_authorized=False,
            hardware_authorized=False,
            warning="scratch branches; not continued live actor, game, training or replay",
        ),
    )
    print(
        json.dumps({"cases": len(results), "branches": sum(len(r["candidates"]) for r in results)})
    )


if __name__ == "__main__":
    main()
