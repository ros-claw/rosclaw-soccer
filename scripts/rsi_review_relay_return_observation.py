"""Inspect actual teammate proximity/contact after a replayed relay return."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "audit", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("source-hash", "audit-hash", "scene-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--precontact-continuation", action="store_true")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    source_path, scene = a.source_root / "complete.json", a.source_root / "scene.xml"
    files = {source_path: a.source_hash, scene: a.scene_hash, a.audit: a.audit_hash}
    if a.output.exists() or any(p.is_symlink() or digest(p) != h for p, h in files.items()):
        raise ValueError("unused output and exact source, scene, and own replay required")
    source, audit = (json.loads(p.read_text()) for p in (source_path, a.audit))
    if (
        source.get("status")
        != (
            "NEURAL_RELAY_RECEIVER_PRECONTACT_CONTINUATION16_PENDING_OWN_REPLAY"
            if a.precontact_continuation
            else "NEURAL_RELAY_RECEIVER_COORDINATED_INTERCEPT16_PENDING_OWN_REPLAY"
        )
        or audit.get("status") != "ALL_NEURAL_RELAY_OWN_HISTORY_NATIVE_REPLAY_MATCH"
        or audit.get("producer_hash") != a.source_hash
        or len(source.get("records", [])) != 16
        or len(audit.get("records", [])) != 16
        or audit.get("promotion_authorized") is not False
        or audit.get("hardware_authorized") is not False
    ):
        raise ValueError("complete exact unpromoted own replay required")

    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.relay_return_evidence import return_observation

    model = mujoco.MjModel.from_xml_path(str(scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="a/", ball_body_name="soccer_ball", player_count=2
    )
    rows = []
    for r, replay in zip(source["records"], audit["records"], strict=True):
        if r["index"] != replay["index"] or r["course"] != replay["course"]:
            raise ValueError("own replay identities must match")
        item = r["artifacts"]["physics"]
        path = a.source_root / item["path"]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(a.source_root.resolve())
            or digest(path) != item["hash"]
        ):
            raise ValueError("exact confined physical trajectory required")
        observation = None
        if r["handoff"] is not None:
            with np.load(path, allow_pickle=False) as z:
                q = z["qpos"].reshape(-1, model.nq)
                v = z["qvel"].reshape(-1, model.nv)
                forces = z["ball_forces"].reshape(-1, 2, 3)
                start = r["handoff"]["control"] * 10
                goal = np.asarray(r["handoff"]["return_goal"])
                direction = goal[:2] - np.asarray(r["handoff"]["kick_origin"])[:2]
                norm = float(np.linalg.norm(direction))
                if not np.isfinite(norm) or norm <= 0:
                    raise ValueError("finite nonzero return direction required")
                strike = np.flatnonzero(
                    (np.max(forces[start:, 1, :2], axis=1) > 10)
                    & (v[start:, layout.bv : layout.bv + 2] @ (direction / norm) >= 1)
                )
                if len(strike):
                    observation = return_observation(
                        q,
                        v,
                        forces,
                        start=start + int(strike[0]),
                        goal=goal,
                        ball_q=layout.bq,
                        ball_v=layout.bv,
                        teammate_q=layout.rq,
                    )
        if digest(path) != item["hash"]:
            raise ValueError("trajectory changed during observation")
        rows.append(
            dict(
                index=r["index"],
                course=r["course"],
                termination=r["termination"],
                producer_result=r["result"],
                observation=observation,
                source_physics_hash=item["hash"],
            )
        )
    if any(p.is_symlink() or digest(p) != h for p, h in files.items()):
        raise ValueError("source evidence changed during review")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="RELAY_TEAMMATE_RETURN_OBSERVATIONS_NOT_POSSESSION",
            source_hash=a.source_hash,
            audit_hash=a.audit_hash,
            scene_hash=a.scene_hash,
            rows=rows,
            fresh=False,
            new_physics=False,
            precontact_continuation=a.precontact_continuation,
            closest_search_begins_at_recorded_outgoing_foot_strike=True,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
