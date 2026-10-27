"""Read-only diagnostic of classroom-actor support on actual-pass traces.

The hypothetical decision is 19 controls after measured passer foot contact.
It does not install the actor, simulate a response, or prove pass transfer.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import ReceivingTemporalMotorExpert
from rosclaw_soccer.rsi.team_coverage_actor import TeamCoverageActor
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "audit", "models", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("source-hash", "audit-hash", "models-hash", "scene-hash", "expected-head"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    source_path, audit_path, model_path = (
        args.source / "complete.json",
        args.audit / "complete.json",
        args.models / "complete.json",
    )
    scene_path = args.source / "scene.xml"
    paths = (source_path, audit_path, model_path, scene_path)
    hashes = (args.source_hash, args.audit_hash, args.models_hash, args.scene_hash)
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() or digest(p) != h for p, h in zip(paths, hashes, strict=True))
    ):
        raise ValueError("exact source, audit, actors and unused external output required")
    report, audit, training = (json.loads(p.read_text()) for p in paths[:3])
    if (
        report["status"] != "PREALIGNED_FOOT_CAPTURE_TEACHER32_COLLECTED_PENDING_OWN_REPLAY"
        or len(report["records"]) != 32
        or audit["status"] != "PREALIGNED_FOOT_CAPTURE32_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS"
        or audit["producer_hash"] != args.source_hash
        or training["status"] != "TEAM_COVERAGE_SUPERVISED_ACTORS_NOT_EVALUATED"
    ):
        raise ValueError("32 audited actual-pass development traces and actor lineage required")
    models = {}
    for record in training["records"]:
        path = args.models / record["model_path"]
        if path.is_symlink() or path.parent != args.models or digest(path) != record["model_hash"]:
            raise ValueError("exact confined actor required")
        models[str(record["seed"])] = TeamCoverageActor.from_dict(json.loads(path.read_text()))
    if set(models) != {"17", "43", "71"}:
        raise ValueError("three frozen actors required")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    data = mujoco.MjData(model)
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            source_hash=args.source_hash,
            audit_hash=args.audit_hash,
            models_hash=args.models_hash,
            scene_hash=args.scene_hash,
            source_head=args.expected_head,
            alignment="hypothetical decision 19 controls after first passer-foot force over 10N",
            own_foot_history="force over 1N in recorded prior receiver substeps",
            physical_integrations=0,
            actor_installed=False,
            simulation_transfer_proven=False,
            hardware_authorized=False,
        ),
    )
    rows = []
    for record in report["records"]:
        path = args.source / record["artifact"]["path"]
        if (
            path.is_symlink()
            or path.parent != args.source
            or digest(path) != record["artifact"]["hash"]
        ):
            raise ValueError("exact confined native physical trace required")
        with np.load(path, allow_pickle=False) as f:
            qpos, qvel, force, times = (
                f[k].copy() for k in ("qpos", "qvel", "ball_forces", "time_before")
            )
        if (
            qpos.ndim != 3
            or qpos.shape[1:] != (10, model.nq)
            or qvel.shape != (len(qpos), 10, model.nv)
            or force.shape != (len(qpos), 10, 2, 3)
            or times.shape != (len(qpos),)
            or not all(np.isfinite(x).all() for x in (qpos, qvel, force, times))
        ):
            raise ValueError("complete finite actual native trace required")
        events = np.flatnonzero((force[:, :, 0, :2] > 10.0).any(axis=(1, 2)))
        item = dict(
            index=record["index"],
            course=record["course"],
            trace_hash=record["artifact"]["hash"],
            observed_passer_contact=bool(len(events)),
            hypothetical_decision_available=False,
        )
        if len(events) and int(events[0]) + 19 < len(qpos):
            decision = int(events[0]) + 19
            data.qpos[:], data.qvel[:] = qpos[decision - 1, -1], qvel[decision - 1, -1]
            q, v = layout.canonical(data)
            own_events = np.flatnonzero((force[:decision, :, 1, :2] > 1.0).any(axis=(1, 2)))
            own_foot = bool(len(own_events))
            observation = SimpleNamespace(
                qpos=q,
                qvel=v,
                frame=19,
                last_own_foot_contact_time_sec=float(times[own_events[-1]]) if own_foot else None,
            )
            features = ReceivingTemporalMotorExpert.features(observation)
            item.update(
                hypothetical_decision_available=True,
                contact_control=int(events[0]),
                decision_control=decision,
                measured_time_sec=float(times[decision]),
                own_foot_contact_observed=own_foot,
                features=list(features),
                selectors={
                    seed: dict(
                        action=actor.select(features)[0],
                        fallback=actor.select(features)[1],
                        nearest_training_distance=float(
                            np.min(np.linalg.norm(actor.support - np.asarray(features), axis=1))
                        ),
                    )
                    for seed, actor in models.items()
                },
            )
        rows.append(item)
        write_once(args.output / f"course-{record['index']}.json", item)
    if any(digest(p) != h for p, h in zip(paths, hashes, strict=True)):
        raise ValueError("source changed during diagnostic")
    write_once(
        args.output / "complete.json",
        dict(
            status="ACTUAL_PASS_RECORDED_ACTOR_SUPPORT_DIAGNOSTIC_NOT_TRANSFER",
            source_hash=args.source_hash,
            audit_hash=args.audit_hash,
            models_hash=args.models_hash,
            scene_hash=args.scene_hash,
            source_head=args.expected_head,
            rows=rows,
            available=sum(r["hypothetical_decision_available"] for r in rows),
            within_support_radius={
                seed: sum(
                    r["hypothetical_decision_available"]
                    and r["selectors"][seed]["nearest_training_distance"] <= models[seed].radius
                    for r in rows
                )
                for seed in models
            },
            actor_nonfallback={
                seed: sum(
                    r["hypothetical_decision_available"] and not r["selectors"][seed]["fallback"]
                    for r in rows
                )
                for seed in models
            },
            physical_integrations=0,
            actor_installed=False,
            simulation_transfer_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
