"""One declared native tracking case: reference PD versus predictive teacher.

Nominal future: no scheduled external velocity impulses. Initial disturbances
come unchanged from the recorded teacher world. This is NOT the previous GPU
disturbance exam, a neural policy, or a football promotion certificate.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("export-root", "reference-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("export-unit", "export-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_qualify_mjlab_motion_dataset import permutations

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("frozen native teacher and unused output required")
    wait_for_terminal_collection(a.export_unit)
    receipt_path = a.export_root / "complete.json"
    if digest(receipt_path) != a.export_hash:
        raise ValueError("exact original native-export receipt required")
    receipt = json.loads(receipt_path.read_text())
    if receipt["status"] != "MJLAB_NATIVE_EXPORT_ONE_CONTROL_DIAGNOSTIC_NOT_LONG_RUN_QUALIFIED":
        raise ValueError("actual original compiled-model export required")
    pins = dict(receipt["input_file_hashes"])
    pins[str(receipt_path)] = a.export_hash
    model_path = a.export_root / "compiled_model.mjb"
    pins[str(model_path)] = receipt["model_hash"]
    boundary = a.export_root / "native-gpu-control1.npz"
    pins[str(boundary)] = receipt["state_artifact_hash"]
    source_path = (
        a.reference_root / "src/assets/soccer/motions/shooter/soccer-standard-001_right.npz"
    )
    pins[str(source_path)] = digest(source_path)
    for directory in (Path.cwd() / "src", Path.cwd() / "scripts"):
        for source in directory.rglob("*.py"):
            pins[str(source)] = digest(source)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_name=source_path.name,
            world_index=0,
            physical_horizon_controls=300,
            future_external_impulses=False,
            initial_recorded_perturbations_unchanged=True,
            planning_samples=64,
            planning_horizon_controls=12,
            planning_improvements=2,
            planning_threads=8,
            planning_seed=1701,
            planning_sigma_rad=0.1,
            planning_delta_limit_rad=0.3,
            maximum_requested_planning_substeps=4608000,
            maximum_main_native_steps=6000,
            new_neural_calls=0,
            new_optimizer_steps=0,
            hardware_authorized=False,
        ),
    )
    tracker = None
    rows = []
    main_steps = 0
    try:
        import mujoco
        import numpy as np

        from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
        from rosclaw_soccer.rsi.native_predictive_tracking import (
            NativePredictiveTracker,
            PredictiveTrackingBudget,
        )

        if mujoco.__version__ != "3.5.0":
            raise ValueError("exact exported native engine required")
        model = mujoco.MjModel.from_binary_path(str(model_path))
        joint_map, body_map = permutations(
            a.reference_root / "src/tasks/soccer/mdp/shooter_commands.py"
        )
        names = tuple("robot/" + name for name in G1_DDS_JOINT_NAMES)
        body_names = [
            model.body(i).name
            for i in range(model.nbody)
            if model.body(i).name.startswith("robot/")
        ]
        if len(body_names) != 30:
            raise ValueError("qualified natural 30-body robot ordering required")
        torso = body_names.index("robot/torso_link")
        root_id = model.body("robot/pelvis").id
        torso_id = model.body("robot/torso_link").id
        ee_names = (
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        )
        ee_local = [body_names.index("robot/" + name) for name in ee_names]
        ee_global = [model.body("robot/" + name).id for name in ee_names]
        # Mocap base is static throughout this native pilot. Keep it frozen in
        # every planning USER stream; no tracking target is moved physically.
        if model.neq or model.na:
            raise ValueError("pilot requires the exported equality-free, activation-free model")
        with np.load(source_path, allow_pickle=False) as source:
            qref = source["joint_pos"][:, joint_map].astype(np.float64)
            positions = source["body_pos_w"][:, body_map].astype(np.float64)
            quaternions = source["body_quat_w"][:, body_map].astype(np.float64)
        with np.load(boundary, allow_pickle=False) as data:
            q0 = data["qpos"][0].copy()
            v0 = data["qvel"][0].copy()
            w0 = data["qacc_warmstart"][0].copy()
            default = data["initial_default_target"][0].copy()
            if int(data["motion_id"][0]) != 0 or int(data["phase"][0]) != 0:
                raise ValueError("source001 recorded frame0 required")
        ids = tuple(receipt["actuator_ids_hg"])
        tracker = NativePredictiveTracker(
            model,
            joint_names_hg=names,
            actuator_ids_hg=ids,
            root_joint_name="robot/floating_base_joint",
            budget=PredictiveTrackingBudget(),
        )
        if np.any(qref < tracker.bounds[:, 0] - 1e-6) or np.any(qref > tracker.bounds[:, 1] + 1e-6):
            raise ValueError("source reference exceeds actual compiled physical joint bounds")
        results = []
        initial_snapshot = None
        for arm in ("reference", "predictive"):
            data = mujoco.MjData(model)
            data.qpos[:] = q0
            data.qvel[:] = v0
            data.qacc_warmstart[:] = w0
            data.ctrl[list(ids)] = default
            mujoco.mj_forward(model, data)
            snapshot = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(model, data, snapshot, mujoco.mjtState.mjSTATE_INTEGRATION)
            if initial_snapshot is None:
                initial_snapshot = snapshot.copy()
            elif not np.array_equal(initial_snapshot, snapshot):
                raise ValueError("native arms must have exactly identical full integration states")
            rows = []
            completed = False
            reason = None
            for frame in range(min(300, len(qref))):
                row = dict(
                    qpos_before=data.qpos.copy(),
                    qvel_before=data.qvel.copy(),
                    reference_joint=qref[frame].copy(),
                    phase=frame,
                )
                if arm == "reference":
                    target = qref[frame].copy()
                    best = nominal = 0.0
                else:
                    phases = np.minimum(frame + np.arange(12), len(qref) - 1)
                    plan = tracker.plan(
                        data, qref[phases], positions[phases, 0], quaternions[phases, 0]
                    )
                    target = plan["target_hg"]
                    best = plan["best_cost"]
                    nominal = plan["nominal_cost"]
                row.update(
                    target_hg=target.copy(),
                    best_planning_cost=best,
                    nominal_planning_cost=nominal if nominal is not None else np.nan,
                )
                data.ctrl[list(ids)] = target
                subq = []
                subv = []
                force = []
                for _ in range(10):
                    before = data.time
                    mujoco.mj_step(model, data)
                    main_steps += 1
                    if not np.isclose(data.time - before, 0.002, atol=1e-10, rtol=0):
                        raise ValueError("main native integration reset or failed to advance")
                    subq.append(data.qpos.copy())
                    subv.append(data.qvel.copy())
                    force.append(data.actuator_force.copy())
                mujoco.mj_forward(model, data)
                expected = np.empty(9)
                mujoco.mju_quat2Mat(expected, quaternions[frame, torso])
                flags = dict(
                    fell_over=bool(data.xmat[root_id, 8] < np.cos(np.deg2rad(70.0))),
                    anchor_pos_z=bool(
                        abs(data.xpos[torso_id, 2] - positions[frame, torso, 2]) > 0.25
                    ),
                    anchor_ori=bool(abs(data.xmat[torso_id, 8] - expected[8]) > 0.8),
                    ee_body_pos=bool(
                        np.any(
                            np.abs(data.xpos[ee_global, 2] - positions[frame, ee_local, 2]) > 0.25
                        )
                    ),
                )
                row.update(
                    qpos_after=data.qpos.copy(),
                    qvel_after=data.qvel.copy(),
                    native_substep_qpos=np.asarray(subq),
                    native_substep_qvel=np.asarray(subv),
                    native_actuator_force=np.asarray(force),
                    failure_flags=np.array(list(flags.values())),
                )
                rows.append(row)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError("nonfinite native main body")
                if any(flags.values()):
                    reason = [key for key, value in flags.items() if value]
                    break
                if frame == len(qref) - 1:
                    completed = True
                    break
                if (frame + 1) % 20 == 0:
                    print(
                        json.dumps(
                            dict(
                                arm=arm,
                                completed_controls=frame + 1,
                                requested_planning_substeps=tracker.requested_planning_substeps,
                            )
                        ),
                        flush=True,
                    )
            path = a.output / (arm + "-raw.npz")
            np.savez_compressed(path, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
            result = dict(
                arm=arm,
                control_steps=len(rows),
                reference_completed=completed,
                failure_reason=reason,
                right_censored=not completed and reason is None,
                raw_hash=digest(path),
                new_neural_calls=0,
                new_optimizer_steps=0,
            )
            write_once(a.output / (arm + ".json"), result)
            results.append(result)
            print(json.dumps(result), flush=True)
            rows = []
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE_PREDICTIVE_TEACHER_ONE_CASE_NOT_NEURAL_OR_FOOTBALL_QUALIFIED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                results=results,
                main_native_steps=main_steps,
                requested_planning_substeps=tracker.requested_planning_substeps,
                observed_advancing_planning_substeps=tracker.observed_advancing_substeps,
                planning_updates=tracker.planning_updates,
                initial_integration_states_paired_exact=True,
                future_external_impulses=False,
                not_the_prior_gpu_disturbance_exam=True,
                new_neural_calls=0,
                new_optimizer_steps=0,
                football_skill_proven=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        if rows:
            import numpy as np

            np.savez_compressed(
                a.output / "partial-failed-arm.npz",
                **{k: np.array([r[k] for r in rows]) for k in rows[0]},
            )
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                main_native_steps=main_steps,
                requested_planning_substeps=tracker.requested_planning_substeps
                if tracker is not None
                else 0,
                observed_advancing_planning_substeps=tracker.observed_advancing_substeps
                if tracker is not None
                else 0,
                automatic_retry_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise
    finally:
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    main()
