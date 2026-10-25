"""Read saved force witnesses and reconstruct PRE-step collision geometry only."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview", nargs=2, action="append", required=True, metavar=("ROOT", "HASH")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(args.expected_head)
    if args.output.exists() or len(args.preview) != 4:
        raise ValueError("unused output and exactly four declared public failures required")
    pins, reports = {}, []
    for root, expected in args.preview:
        root = Path(root)
        path = root / "complete.json"
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("exact closed preview required")
        report = json.loads(path.read_text())
        if (
            report["status"] != "THREE_MEASURED_SWITCH_PREVIEWS_RECORDED_NOT_ACTUAL_GAIN"
            or report["public_baseline_and_teacher_failed"] is not True
        ):
            raise ValueError("closed public-failure preview required")
        for name, commitment in report["input_file_hashes"].items():
            if name in pins and pins[name] != commitment:
                raise ValueError("conflicting source commitments")
            pins[name] = commitment
        pins[str(path)] = expected
        reports.append((root, report))
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    scenes = [Path(p) for p in pins if p.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("one pinned compiled body source required")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    body_hash = compiled_model_hash(model)
    if body_hash != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d":
        raise ValueError("exact previously qualified compiled body required")
    ball = model.body("soccer_ball").id
    data = mujoco.MjData(model)
    results = []
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            source_head=args.expected_head,
            input_file_hashes=pins,
            model_hash=body_hash,
            new_physical_steps=0,
            new_neural_calls=0,
            optimizer_updates=0,
            reconstructs_force_solver=False,
            geometry_uses_previous_saved_poststep_qpos=True,
            hardware_authorized=False,
        ),
    )
    for root, report in reports:
        with np.load(root / "actual-unvalidated.npz", allow_pickle=False) as actual:
            for velocity, score in zip((1, 7, 13), report["scores"], strict=True):
                path = root / f"future-v{velocity:02d}.npz"
                meta = json.loads(path.with_suffix(".json").read_text())
                origin = meta["donor_control_frame"]
                with np.load(path, allow_pickle=False) as raw:
                    force = raw["ball_forces"][:, 0, 2]
                    events = np.flatnonzero(force > 1)
                    witnesses = []
                    for index in sorted(
                        {int(events[0]), int(force.argmax())} if len(events) else set()
                    ):
                        data.qpos[:] = (
                            raw["qpos"][index - 1] if index else actual["qpos"][origin * 10 - 1]
                        )
                        mujoco.mj_kinematics(model, data)
                        mujoco.mj_comPos(model, data)
                        mujoco.mj_collision(model, data)
                        contacts = []
                        for contact in data.contact[: data.ncon]:
                            geom = [int(g) for g in contact.geom]
                            bodies = [int(model.geom_bodyid[g]) for g in geom]
                            if ball not in bodies:
                                continue
                            other = bodies[1] if bodies[0] == ball else bodies[0]
                            name = model.body(other).name
                            if not name.startswith("a/"):
                                continue
                            contacts.append(
                                dict(
                                    body=name,
                                    geoms=[model.geom(g).name for g in geom],
                                    signed_distance_m=float(contact.dist),
                                    position=contact.pos.tolist(),
                                )
                            )
                        witnesses.append(
                            dict(
                                native_index=index,
                                absolute_control=origin + index // 10,
                                measured_aggregate_nonfoot_force_n=float(force[index]),
                                geometry_contacts=contacts,
                            )
                        )
                    results.append(
                        dict(
                            root=str(root),
                            delay=velocity - 1,
                            complete_prediction=score["eligible"],
                            body_failed=meta["body_failed"],
                            admission_rejection=meta["admission_rejection"],
                            nonfoot_peak_force_n=float(force.max()),
                            witnesses=witnesses,
                        )
                    )
    check_pins(pins)
    clean(args.expected_head)
    write_once(
        args.output / "complete.json",
        dict(
            status="SAVED_NONFOOT_FORCE_AND_PRESTEP_COLLISION_GEOMETRY_REVIEWED",
            source_head=args.expected_head,
            input_file_hashes=pins,
            model_hash=body_hash,
            results=results,
            new_physical_steps=0,
            new_neural_calls=0,
            optimizer_updates=0,
            contact_forces_are_saved_witnesses_not_new_solver_outputs=True,
            geometry_is_not_a_new_dynamic_counterfactual=True,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
