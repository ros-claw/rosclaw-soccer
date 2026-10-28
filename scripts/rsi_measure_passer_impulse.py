"""Measure archived paired pass impulses; no live actor or new physics."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.pass_impulse_window import measure_first_pass_impulse
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, action="append", required=True)
    p.add_argument("--source-hash", action="append", required=True)
    p.add_argument("--scene-hash", required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--review-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        len(a.source) != 3
        or len(a.source_hash) != 3
        or len(set(a.source)) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.review.is_symlink()
        or digest(a.review) != a.review_hash
    ):
        raise ValueError("three pinned sources and unused external output required")
    review = json.loads(a.review.read_text())
    if (
        review.get("status") != "PASSER_RESIDUAL_PAIR_REPLAY_REVIEWED"
        or review.get("conditions") != 24
        or any(review.get(k) is not False for k in ("hardware_authorized", "promotion_authorized"))
    ):
        raise ValueError("reviewed development source required")
    rows = []
    for block, (source, expected) in enumerate(zip(a.source, a.source_hash, strict=True)):
        receipt = source / "complete.json"
        scene = source / "scene.xml"
        if receipt.is_symlink() or scene.is_symlink() or digest(receipt) != expected:
            raise ValueError("exact source receipt required")
        if digest(scene) != a.scene_hash:
            raise ValueError("exact native scene required")
        doc = json.loads(receipt.read_text())
        if (
            doc.get("status") != "NEURAL_RELAY_PASSER_CONTACT_EXIT16_PENDING_OWN_REPLAY"
            or doc.get("source_head") != review.get("producer_head")
            or len(doc.get("records", [])) != 16
        ):
            raise ValueError("exact paired passer source required")
        model = mujoco.MjModel.from_xml_path(str(scene))
        layout = NativeVelocityLayout(
            model, robot_prefix="a/", ball_body_name="soccer_ball", player_count=2
        )
        for record in doc["records"]:
            item = record["artifacts"]["physics"]
            path = source / item["path"]
            if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
                raise ValueError("confined physical trajectory required")
            if digest(path) != item["hash"]:
                raise ValueError("exact physical trajectory required")
            with np.load(path, allow_pickle=False) as z:
                force = z["ball_forces"].reshape(-1, 2, 3)
                velocity = z["qvel"].reshape(-1, model.nv)[:, layout.bv : layout.bv + 3]
                result = measure_first_pass_impulse(force, velocity)
            if digest(path) != item["hash"]:
                raise ValueError("physical trajectory changed during measurement")
            course = record["course"]
            rows.append(
                dict(
                    block=block,
                    context=course["context"],
                    side=course["lateral_m"],
                    passer_residual_fraction=course["arm"]["passer_residual_fraction"],
                    physics_hash=item["hash"],
                    **result,
                )
            )
        if digest(receipt) != expected or digest(scene) != a.scene_hash:
            raise ValueError("source changed during measurement")
    if len(rows) != 48 or digest(a.review) != a.review_hash:
        raise ValueError("48 unchanged developmental paths required")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="PASS_IMPULSE_EXIT_OFFLINE_NOT_QUALIFICATION",
            source_head=a.expected_head,
            source_hashes=a.source_hash,
            scene_hash=a.scene_hash,
            review_hash=a.review_hash,
            quiet_steps=5,
            step_seconds=float(model.opt.timestep),
            rows=rows,
            new_physics=False,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
