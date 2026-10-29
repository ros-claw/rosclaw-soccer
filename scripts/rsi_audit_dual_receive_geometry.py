"""Inspect saved post-pass contact geometry; no new simulation or model update."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "audit", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "audit-hash", "scene-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    source_file, scene = a.source_root / "complete.json", a.source_root / "scene.xml"
    files = {source_file: a.source_hash, scene: a.scene_hash, a.audit: a.audit_hash}
    if a.output.exists() or any(p.is_symlink() or digest(p) != h for p, h in files.items()):
        raise ValueError("unused output and exact bound source, scene and replay required")
    source, audit = json.loads(source_file.read_text()), json.loads(a.audit.read_text())
    if (
        source.get("status") != "NEURAL_RELAY_RECEIVER_ADAPTIVE_INTERCEPT32_PENDING_OWN_REPLAY"
        or audit.get("status") != "ALL_NEURAL_RELAY_OWN_HISTORY_NATIVE_REPLAY_MATCH"
        or audit.get("producer_hash") != a.source_hash
        or len(source.get("records", [])) != 32
        or len(audit.get("records", [])) != 32
        or audit.get("hardware_authorized") is not False
        or audit.get("promotion_authorized") is not False
    ):
        raise ValueError("all 32 source and own-replay records required")

    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.dual_receive_geometry_audit import first_receiver_nonfoot_after_pass

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    ball_id = model.body("soccer_ball").id
    ball_geoms = np.flatnonzero(model.geom_bodyid == ball_id)
    if len(ball_geoms) != 1 or model.geom_type[ball_geoms[0]] != mujoco.mjtGeom.mjGEOM_SPHERE:
        raise ValueError("one spherical ball geometry required")
    foot, nonfoot = [], []
    for i in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[i])).name or ""
        if not body.startswith("b/") or not (model.geom_contype[i] or model.geom_conaffinity[i]):
            continue
        (foot if "ankle_roll" in body else nonfoot).append(i)
    if not foot or not nonfoot:
        raise ValueError("receiver foot and nonfoot collision geometries required")

    def min_gap(geom_ids):
        distances = [
            float(mujoco.mj_geomDistance(model, data, int(ball_geoms[0]), i, 2.0, None))
            for i in geom_ids
        ]
        j = int(np.argmin(distances))
        geom = geom_ids[j]
        return dict(
            distance_m=distances[j],
            geom=model.geom(geom).name,
            body=model.body(int(model.geom_bodyid[geom])).name,
        )

    rows = []
    for record, own in zip(source["records"], audit["records"], strict=True):
        if record["index"] != own["index"] or record["course"] != own["course"]:
            raise ValueError("source and own-replay identity mismatch")
        artifact = record["artifacts"]["physics"]
        path = a.source_root / artifact["path"]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(a.source_root.resolve())
            or digest(path) != artifact["hash"]
        ):
            raise ValueError("exact confined physical trajectory required")
        with np.load(path, allow_pickle=False) as z:
            qpos, qvel, forces = z["qpos"].copy(), z["qvel"].copy(), z["ball_forces"].copy()
        if (
            qpos.shape[1:] != (10, model.nq)
            or qvel.shape != (len(qpos), 10, model.nv)
            or forces.shape != (len(qpos), 10, 2, 3)
            or not np.isfinite(qpos).all()
            or not np.isfinite(qvel).all()
        ):
            raise ValueError("aligned finite native-state trajectory required")
        f = forces.reshape(-1, 2, 3)
        passer, event = first_receiver_nonfoot_after_pass(f)
        row = dict(
            index=record["index"],
            course=record["course"],
            termination=record["termination"],
            passer_foot_step=passer,
            receiver_nonfoot_step=event,
            source_physics_hash=artifact["hash"],
            event_geometry=None,
            before_100ms_geometry=None,
        )
        for label, index in (
            ("event_geometry", event),
            ("before_100ms_geometry", None if event is None or event < 50 else event - 50),
        ):
            if index is None:
                continue
            data.qpos[:] = qpos.reshape(-1, model.nq)[index]
            data.qvel[:] = qvel.reshape(-1, model.nv)[index]
            mujoco.mj_forward(model, data)
            row[label] = dict(foot=min_gap(foot), nonfoot=min_gap(nonfoot))
        rows.append(row)
    if any(p.is_symlink() or digest(p) != h for p, h in files.items()):
        raise ValueError("source evidence changed during audit")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="DUAL_RECEIVE_GEOMETRY_OBSERVATION_NOT_CONTROL",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            audit_hash=a.audit_hash,
            scene_hash=a.scene_hash,
            geometry_is_not_per_body_measured_force=True,
            kinematic_reconstruction_only=True,
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
