"""Screen every replay-verified shared-world exploration path for reception."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("evidence-root", "review", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("review-hash", "scene-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    if a.output.exists() or a.review.is_symlink() or digest(a.review) != a.review_hash:
        raise ValueError("unused output and exact exploration review required")
    review = json.loads(a.review.read_text())
    if (
        review.get("status") != "RELAY_RECEIVER_EXPLORATION_PAIRED_NOT_TRAINED"
        or review.get("own_physics_neural_and_density_audited") is not True
        or review.get("promotion_authorized") is not False
        or [x["block"] for x in review.get("source_pins", [])] != [0, 1, 2]
    ):
        raise ValueError("all three exact unpromoted exploration blocks required")
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.strict_receive_screen import screen_receive

    rows = []
    for proof in review["source_pins"]:
        block = proof["block"]
        root = a.evidence_root / f"rsi-relay-receiver-exploration-b{block}-20261027"
        source_path, own_path = root / "source/complete.json", root / "audit/complete.json"
        scene = root / "source/scene.xml"
        if (
            any(p.is_symlink() for p in (source_path, own_path, scene))
            or digest(source_path) != proof["source_hash"]
            or digest(own_path) != proof["own_audit_hash"]
            or digest(scene) != a.scene_hash
        ):
            raise ValueError("exact verified source, replay and scene required")
        source, own = json.loads(source_path.read_text()), json.loads(own_path.read_text())
        if (
            source.get("status") != "NEURAL_RELAY_RECEIVER_EXPLORATION32_PENDING_OWN_REPLAY"
            or own.get("status") != "ALL_NEURAL_RELAY_OWN_HISTORY_NATIVE_REPLAY_MATCH"
            or own.get("producer_hash") != proof["source_hash"]
            or len(source.get("records", [])) != 32
            or len(own.get("records", [])) != 32
        ):
            raise ValueError("complete source and own-feedback physical replay required")
        model = mujoco.MjModel.from_xml_path(str(scene))
        layout = NativeVelocityLayout(
            model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
        )
        for record, replay in zip(source["records"], own["records"], strict=True):
            if record["index"] != replay["index"] or record["course"] != replay["course"]:
                raise ValueError("matched own-replay course identity required")
            item = record["artifacts"]["physics"]
            path = root / "source" / item["path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to((root / "source").resolve())
                or digest(path) != item["hash"]
            ):
                raise ValueError("exact confined actual-physics trajectory required")
            with np.load(path, allow_pickle=False) as z:
                result = screen_receive(
                    z["ball_forces"].reshape(-1, 2, 3),
                    z["qpos"].reshape(-1, model.nq),
                    z["qvel"].reshape(-1, model.nv),
                    z["body_failed"],
                    ball_q=layout.bq,
                    ball_v=layout.bv,
                    root_q=layout.rq,
                    root_v=layout.rv,
                )
            if digest(path) != item["hash"]:
                raise ValueError("physical source changed during read")
            rows.append(
                dict(
                    block=block,
                    index=record["index"],
                    course=record["course"],
                    termination=record["termination"],
                    physics_hash=item["hash"],
                    **result,
                )
            )
        if (
            digest(source_path) != proof["source_hash"]
            or digest(own_path) != proof["own_audit_hash"]
            or digest(scene) != a.scene_hash
        ):
            raise ValueError("block evidence changed")
    if len(rows) != 96 or digest(a.review) != a.review_hash:
        raise ValueError("all 96 source paths required")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="RELAY_EXPLORATION_ENDPOINT_SCREEN_NOT_POSSESSION",
            source_head=a.expected_head,
            review_hash=a.review_hash,
            scene_hash=a.scene_hash,
            rows=rows,
            independent_initial_contexts_unproven=True,
            endpoint_not_continuous_control=True,
            new_physics=False,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
