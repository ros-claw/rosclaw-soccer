"""Bounded readback of failed memory transfer: force timing and scratch FK only.

Never integrates physics, trains, changes an exam, or promotes a controller.
Reconstructed overlap is NOT attribution of recorded contact force to a body.
"""

import argparse
import json
import subprocess
from pathlib import Path


def contact_times(forces):
    import numpy as np

    if (
        forces.ndim != 3
        or forces.shape[1:] != (1, 3)
        or not 1 <= len(forces) <= 3600
        or not np.isfinite(forces).all()
        or (forces < 0).any()
    ):
        raise ValueError("bounded actual native left/right/nonfoot force trace required")
    first = []
    for column in range(3):
        indices = np.flatnonzero(forces[:, 0, column] > 1)
        first.append(None if not len(indices) else int(indices[0]))
    return dict(left=first[0], right=first[1], nonfoot=first[2])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("pilot", "audit", "motor-memory", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("audit-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and clean exact frozen source required")
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if digest(a.audit) != a.audit_hash:
        raise ValueError("exact independent audit required")
    audit = json.loads(a.audit.read_text())
    if (
        audit["status"] != "FULL_NN_PARTIAL_MEMORY_RAW_AND_RETRIEVAL_VERIFIED"
        or audit["memory_nominal_transfer"] is not True
        or audit["qualified_for_expansion"] is not False
        or len(audit["results"]) != 16
    ):
        raise ValueError("audited bounded nominal16 diagnostic scope required")
    bound = audit["input_file_hashes"]
    pins = {str(a.audit): a.audit_hash}

    def admit(path):
        expected = bound.get(str(path))
        if expected is None or digest(path) != expected:
            raise ValueError("audit-bound readback source changed: " + str(path))
        pins[str(path)] = expected

    admit(a.pilot / "complete.json")
    admit(a.motor_memory)
    memory = json.loads(a.motor_memory.read_text())
    scenes = [Path(k) for k in bound if k.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("one exact scene source required")
    admit(scenes[0])
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    if compiled_model_hash(model) != memory["compiled_model_hash"]:
        raise ValueError("same compiled body required; no replacement physics")
    ball = model.body("soccer_ball").id
    data = mujoco.MjData(model)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.partial_memory_contact_readback.v1",
            audit_hash=a.audit_hash,
            source_head=a.expected_head,
            maximum_trajectories=48,
            maximum_fk_calls=144,
            new_physical_steps=0,
            neural_optimizer_updates=0,
            reconstructed_contact_is_not_force_attribution=True,
            hardware_authorized=False,
        ),
    )
    results, fk_calls = [], 0
    for result in audit["results"]:
        if result["C"]["task_pass"]:
            continue
        index = result["course_index"]
        record = a.pilot / f"course-{index:02d}" / "C.json"
        admit(record)
        row = json.loads(record.read_text())
        traces = [("actual", row["artifact"], 0)] + [
            (f["name"], f["artifact"], f["donor_control_frame"] * 10)
            for f in row["candidate_forecasts"]
            if f["name"].startswith("memory-")
        ]
        for name, artifact, origin in traces:
            if len(results) >= 48:
                raise ValueError("readback trace ceiling exceeded")
            path = record.parent / artifact["path"]
            if path.parent != record.parent or digest(path) != artifact["hash"]:
                raise ValueError("local exact trajectory artifact required")
            admit(path)
            with np.load(path, allow_pickle=False) as raw:
                qpos, qvel, force = raw["qpos"], raw["qvel"], raw["ball_forces"]
                times = contact_times(force)
                if qpos.shape != (len(force), model.nq) or qvel.shape != (len(force), model.nv):
                    raise ValueError("same native model trajectory dimensions required")
                snapshots = []
                for tick in sorted({t for t in times.values() if t is not None}):
                    if fk_calls >= 144:
                        raise ValueError("scratch FK ceiling exceeded")
                    data.qpos[:] = qpos[tick]
                    data.qvel[:] = qvel[tick]
                    mujoco.mj_forward(model, data)
                    fk_calls += 1
                    nearby = []
                    for contact in data.contact:
                        g1, g2 = int(contact.geom1), int(contact.geom2)
                        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
                        if ball not in (b1, b2):
                            continue
                        other = b2 if b1 == ball else b1
                        if not model.body(other).name.startswith("a/"):
                            continue
                        nearby.append(
                            dict(
                                robot_body=model.body(other).name,
                                robot_geom=model.geom(g2 if b1 == ball else g1).name,
                                signed_distance_m=float(contact.dist),
                                force_attribution_proven=False,
                            )
                        )
                    snapshots.append(dict(global_native_index=origin + tick, nearby=nearby))
                results.append(
                    dict(
                        course_index=index,
                        trace=name,
                        actual_execution=name == "actual",
                        first_force_native_indices={
                            k: None if v is None else origin + v for k, v in times.items()
                        },
                        scratch_post_substep_fk=snapshots,
                        force_geometry_time_offset_at_most_s=0.002,
                    )
                )
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="AUDITED_MEMORY_FAILURE_FORCE_TIMING_AND_FK_READBACK_NOT_NEW_SKILL",
            source_head=a.expected_head,
            input_file_hashes=pins,
            results=results,
            scratch_fk_calls=fk_calls,
            new_physical_steps=0,
            neural_optimizer_updates=0,
            qualified_for_expansion=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
