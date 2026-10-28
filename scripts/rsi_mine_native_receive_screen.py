"""Screen all previously replayed CEM development trajectories, without retraining."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.strict_receive_screen import screen_receive
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("pilot", "shards", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("pilot-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.pilot.is_symlink()
        or digest(a.pilot) != a.pilot_hash
    ):
        raise ValueError("exact pilot and unused external output required")
    pilot = json.loads(a.pilot.read_text())
    if (
        pilot.get("status") != "SELF_TEACHER_CEM_FOUR_STATE_PILOT_CLOSED_NOT_NEURAL_TRAINING"
        or len(pilot.get("proofs", [])) != 4
        or any(
            pilot.get(k) is not False
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("exact unpromoted source pilot required")
    rows = []
    for proof in pilot["proofs"]:
        state = proof["state_id"]
        source = a.shards / f"state{state}" / "source"
        own = a.shards / f"state{state}" / "own" / "complete.json"
        receipt = source / "complete.json"
        scene = source / "scene.xml"
        if (
            any(p.is_symlink() for p in (receipt, own, scene))
            or digest(receipt) != proof["source_hash"]
            or digest(own) != proof["own_hash"]
        ):
            raise ValueError("exact source and independent recorded proof required")
        doc = json.loads(receipt.read_text())
        replay = json.loads(own.read_text())
        if (
            doc.get("state_id") != state
            or replay.get("producer_hash") != proof["source_hash"]
            or digest(scene) != doc.get("scene_hash")
            or len(doc.get("records", [])) != 68
        ):
            raise ValueError("full source replay lineage and scene required")
        model = mujoco.MjModel.from_xml_path(str(scene))
        b = NativeVelocityLayout(
            model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
        )
        for r in doc["records"]:
            item = r["artifacts"]["physical"]
            path = source / item["path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(source.resolve())
                or digest(path) != item["hash"]
            ):
                raise ValueError("confined exact native history required")
            with np.load(path, allow_pickle=False) as z:
                result = screen_receive(
                    z["ball_forces"].reshape(-1, 2, 3),
                    z["qpos"].reshape(-1, model.nq),
                    z["qvel"].reshape(-1, model.nv),
                    z["body_failed"],
                    ball_q=b.bq,
                    ball_v=b.bv,
                    root_q=b.rq,
                    root_v=b.rv,
                )
            if digest(path) != item["hash"]:
                raise ValueError("native history changed")
            rows.append(
                dict(
                    state_id=state,
                    task=r["search_task"],
                    tag=r["tag"],
                    physics_hash=item["hash"],
                    **result,
                )
            )
        if (
            digest(receipt) != proof["source_hash"]
            or digest(own) != proof["own_hash"]
            or digest(scene) != doc["scene_hash"]
        ):
            raise ValueError("source changed during screen")
    if len(rows) != 272 or digest(a.pilot) != a.pilot_hash:
        raise ValueError("all four correlated development search banks required")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="REPLAYED_CEM_RECEIVE_SCREEN_NOT_POSSESSION",
            source_head=a.expected_head,
            pilot_hash=a.pilot_hash,
            rows=rows,
            distinct_source_states=4,
            physical_steps_this_screen=0,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
