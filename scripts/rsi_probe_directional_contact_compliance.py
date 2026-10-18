"""Four matched-state directional-compliance interventions, not full G1 policies.

Only the qualified snapshot and one known bootstrap interval precede the plan.
No saved future target is read after entry or used by candidate forecasts.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source = Path(__file__).resolve().parents[1]
    root = Path(cfg["output_root"])
    if (
        root.exists()
        or cfg["snapshot_frames"] != [78, 78, 75, 75]
        or cfg["maximum_forecast_steps"] != 80
        or cfg["bootstrap_steps"] != 40
        or cfg["actual_probe_steps"] != 80
        or cfg["directional_contraction_fraction"] != 0.5
        or cfg.get("hardware_authorized") is not False
        or cfg.get("promotion_authorized") is not False
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("new frozen four-entry one-interval bounded SIM probe required")
    pins = {str(args.config): digest(args.config)}
    for folder in (source / "src", source / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    from rosclaw_soccer.physics.contact_wrench import read_geom_contact_wrenches
    from rosclaw_soccer.physics.g1_joint_task_kinematics import G1JointTaskKinematicSampler
    from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecaster
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.contact_exit_torque_mpc import pressure_side
    from rosclaw_soccer.rsi.directional_contact_impedance import vertical_compliance_stiffness
    from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
    from scripts.rsi_atomic_artifacts import write_once

    def sealed(path, key="report_hash"):
        d = json.loads(Path(path).read_text())
        if d[key] != hash_json({k: v for k, v in d.items() if k != key}):
            raise ValueError("source seal differs")
        pins[str(path)] = digest(path)
        return d

    qual = sealed(Path(cfg["qualification_complete"]))
    declaration = sealed(Path(cfg["qualification_complete"]).parent / "declaration.json")
    if (
        qual.get("schema") != "soccer.rsi.g1_pd_torque_forecast_qualification.v1"
        or qual["declaration_hash"] != declaration["report_hash"]
        or qual["zero_offset_only"] is not True
        or qual["replay_substeps"] != 12000
        or qual["forecast_substeps"] != 400
        or len(qual["rows"]) != 4
        or any(
            len(r["forecasts"]) != 10
            or not all(
                f["all_substep_state_control_and_full_end_state_bit_equal"] for f in r["forecasts"]
            )
            for r in qual["rows"]
        )
    ):
        raise ValueError("whole previously qualified forty-forecast source required")
    uniform = sealed(Path(cfg["uniform_probe_complete"]))
    uniform_declaration = sealed(Path(cfg["uniform_probe_complete"]).parent / "declaration.json")
    if (
        uniform["report_hash"] != cfg["uniform_probe_hash"]
        or uniform["schema"] != "soccer.rsi.contact_leg_half_stiffness_interval_probe.v1"
        or uniform["declaration_hash"] != uniform_declaration["report_hash"]
        or uniform_declaration["source_qualification_hash"] != qual["report_hash"]
        or len(uniform["rows"]) != 4
    ):
        raise ValueError("complete original half-stiffness reference required")
    for p, h in uniform_declaration["input_file_hashes"].items():
        if digest(p) != h:
            raise ValueError("original half-stiffness source/input bytes changed")
        pins[p] = h
    for p, h in declaration["input_file_hashes"].items():
        if digest(p) != h:
            raise ValueError("original qualification input bytes changed")
        pins[p] = h
    upstream_cfg = json.loads(Path(declaration["source_configuration"]).read_text())
    receipts = []
    snapshots = []
    for _i, (item, row, frame) in enumerate(
        zip(upstream_cfg["baseline_receipts"], qual["rows"], cfg["snapshot_frames"], strict=True)
    ):
        r = sealed(Path(item["path"]), "receipt_hash")
        if (
            r["receipt_hash"] != row["source_receipt_hash"]
            or digest(item["path"]) != item["file_hash"]
        ):
            raise ValueError("exact original endpoint identity required")
        snap = next(f["forecast"] for f in row["forecasts"] if f["frame"] == frame)
        if digest(snap["path"]) != snap["sha256"]:
            raise ValueError("qualified full-state snapshot bytes changed")
        pins[snap["path"]] = snap["sha256"]
        receipts.append(r)
        snapshots.append(snap)
    root.mkdir()
    declared = dict(
        cfg,
        input_file_hashes=pins,
        source_qualification_hash=qual["report_hash"],
        candidate_parameters_frozen_before_probe=True,
        new_neural_inferences=0,
        new_optimizer_updates=0,
    )
    declared["report_hash"] = hash_json(declared)
    write_once(root / "declaration.json", declared)
    boot, actual = 0, 0
    attempted_boot, attempted_actual = 0, 0
    teachers = []
    rows = []

    def equal(a, b, name):
        if not np.allclose(a, b, rtol=0, atol=1e-8):
            raise ValueError("checkpoint bootstrap differs: " + name)

    try:
        for case, (r, snap, saved_frame) in enumerate(
            zip(receipts, snapshots, cfg["snapshot_frames"], strict=True)
        ):
            write_once(
                root / f"case-{case}-started.json",
                dict(source_receipt_hash=r["receipt_hash"], probe_attempt_charged=True),
            )
            model = mujoco.MjModel.from_binary_path(r["witnesses"]["world"]["path"])
            commitment = json.loads(Path(r["witnesses"]["commitment"]["path"]).read_text())
            kp = np.asarray(commitment["physics"]["canonical_pd_kp"], dtype=np.float64)
            kd = np.asarray(commitment["physics"]["canonical_pd_kd"], dtype=np.float64)
            sampler = G1JointTaskKinematicSampler(
                model, expected_world_hash=r["witnesses"]["world"]["sha256"]
            )
            with np.load(r["witnesses"]["trace"]["path"], allow_pickle=False) as archive:
                trace = dict(archive)
            with np.load(snap["path"], allow_pickle=False) as archive:
                state = archive["end_integration_state"].copy()
            data = mujoco.MjData(model)
            mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
            joints = [int(model.joint(n).id) for n in G1_DDS_JOINT_NAMES]
            qi = [int(model.jnt_qposadr[j]) for j in joints]
            vi = [int(model.jnt_dofadr[j]) for j in joints]
            ai = [
                int(
                    np.flatnonzero(
                        (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                        & (model.actuator_trnid[:, 0] == j)
                    )[0]
                )
                for j in joints
            ]
            ball = int(model.body("ball").id)
            bj = int(model.body_jntadr[ball])
            bq = int(model.jnt_qposadr[bj])
            bv = int(model.jnt_dofadr[bj])
            limits = np.asarray(G1_HARD_TORQUE_LIMITS)
            bootstrap_frame = saved_frame + 1
            frame = saved_frame + 2
            target = trace["joint_target_rad"][bootstrap_frame, 0]
            for substep in range(10):
                raw = (target - data.qpos[qi]) * kp - data.qvel[vi] * kd
                tau = np.clip(raw, -limits, limits)
                equal(tau, trace["torque_nm"][bootstrap_frame, 0, substep], "original bootstrap PD")
                data.ctrl[ai] = tau
                attempted_boot += 1
                mujoco.mj_step(model, data)
                boot += 1
            q = np.concatenate((data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7]))
            v = np.concatenate((data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6]))
            equal(q, trace["canonical_qpos"][frame, 0], "entry qpos")
            equal(v, trace["canonical_qvel"][frame, 0], "entry qvel")
            past = trace["force_n"][:frame, 0]
            foot = np.flatnonzero(np.any(past[:, :2] > 1, axis=1))
            first = None if not len(foot) else int(foot[0])
            target = trace["joint_target_rad"][frame, 0].copy()
            sample = sampler.sample(data)
            feet = [int(model.body(name + "_ankle_roll_link").id) for name in ("left", "right")]
            bg = int(model.geom("ball_geom").id)
            current_forces = np.zeros(2)
            for wrench in read_geom_contact_wrenches(model, data, geom_id=bg):
                body = int(model.geom_bodyid[wrench.other_geom_id])
                if body in feet:
                    j = feet.index(body)
                    current_forces[j] = max(
                        current_forces[j], float(np.linalg.norm(wrench.force_world_n))
                    )
            side = pressure_side(
                past[-3:].copy(),
                sample.foot_ball_signed_gap_m,
                current_forces,
                prior_nonfoot_contact=bool(np.any(past[:, 2:] > 1)),
            )
            if first is None or frame - first > 60:
                side = None
            jacobian = sample.foot_material_point_jacobian_m_per_rad[0 if side is None else side]
            matrix, shaping = vertical_compliance_stiffness(kp, jacobian, side)
            donor = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(model, data, donor, mujoco.mjtState.mjSTATE_INTEGRATION)
            measures, arrays = (
                {},
                dict(
                    donor_integration_state=donor,
                    parent_kp=kp,
                    candidate_stiffness_matrix=matrix,
                    material_point_jacobian=jacobian,
                    unchanged_kd=kd,
                ),
            )
            for variant, current_matrix in (("parent", None), ("candidate", matrix)):
                forecaster = G1PDTorqueForecaster(
                    model,
                    expected_world_hash=r["witnesses"]["world"]["sha256"],
                    kp=kp,
                    coupled_stiffness_nm_per_rad=current_matrix,
                    kd=kd,
                    max_forecast_steps=10,
                )
                teachers.append(forecaster)
                forecast = forecaster.forecast(data, target, np.zeros(29))
                live = mujoco.MjData(model)
                mujoco.mj_copyData(live, model, data)
                actual_q, actual_v, controls, actuator, robot_forces, heights = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                nonfoot = False
                for _ in range(10):
                    if current_matrix is None or np.array_equal(current_matrix, np.diag(kp)):
                        raw = (target - live.qpos[qi]) * kp - live.qvel[vi] * kd
                    else:
                        raw = current_matrix @ (target - live.qpos[qi]) - live.qvel[vi] * kd
                    tau = np.clip(raw, -limits, limits)
                    live.ctrl[ai] = tau
                    attempted_actual += 1
                    mujoco.mj_step(model, live)
                    actual += 1
                    actual_q.append(
                        np.concatenate((live.qpos[:7], live.qpos[qi], live.qpos[bq : bq + 7]))
                    )
                    actual_v.append(
                        np.concatenate((live.qvel[:6], live.qvel[vi], live.qvel[bv : bv + 6]))
                    )
                    controls.append(tau.copy())
                    actuator.append(live.actuator_force[ai].copy())
                    heights.append(float(live.xpos[model.body("pelvis").id, 2]))
                    force = np.zeros(3)
                    for wrench in read_geom_contact_wrenches(model, live, geom_id=bg):
                        body = int(model.geom_bodyid[wrench.other_geom_id])
                        if body != 0:
                            force += wrench.force_world_n
                            if body not in feet and np.linalg.norm(wrench.force_world_n) > 1:
                                nonfoot = True
                    robot_forces.append(force)
                end_state = np.empty_like(donor)
                mujoco.mj_getState(model, live, end_state, mujoco.mjtState.mjSTATE_INTEGRATION)
                for observed, expected in (
                    (np.asarray(actual_q), forecast.canonical_qpos_after_substep),
                    (np.asarray(actual_v), forecast.canonical_qvel_after_substep),
                    (np.asarray(controls), forecast.applied_control_nm),
                    (np.asarray(actuator), forecast.actual_actuator_force_nm),
                    (np.asarray(robot_forces), forecast.robot_force_on_ball_world_n),
                    (end_state, forecast.end_integration_state),
                ):
                    if observed.tobytes() != expected.tobytes():
                        raise ValueError("matched-stiffness actual and forecast differ bitwise")
                end_sample = sampler.sample(live)
                measures[variant] = dict(
                    actual_forecast_bit_equal=True,
                    downward_robot_impulse_n_s=float(
                        np.maximum(-np.asarray(robot_forces)[:, 2], 0).sum() * 0.002
                    ),
                    ball_forward_velocity_m_s=float(actual_v[-1][35]),
                    ball_lateral_velocity_m_s=float(actual_v[-1][36]),
                    minimum_pelvis_height_m=min(heights),
                    nonfoot_contact_over_1n=nonfoot,
                    foot_ball_gap_m=end_sample.foot_ball_signed_gap_m.tolist(),
                    forecast_opposite_support_weight_fraction=None
                    if side is None
                    else float(
                        np.mean(forecast.completed_foot_ground_normal_n[:, 1 - side])
                        / (float(model.body_subtreemass[model.body("pelvis").id]) * 9.81)
                    ),
                )
                arrays[variant + "_end_integration_state"] = end_state
                arrays[variant + "_qpos_after_substep"] = np.asarray(actual_q)
                arrays[variant + "_qvel_after_substep"] = np.asarray(actual_v)
                arrays[variant + "_actual_torque_nm"] = np.asarray(actuator)
                arrays[variant + "_robot_force_on_ball_world_n"] = np.asarray(robot_forces)
            reference = uniform["rows"][case]
            if (
                reference["source_receipt_hash"] != r["receipt_hash"]
                or reference["frame"] != frame
                or reference["donor_state_hash"] != hash_bytes(donor.tobytes())
                or reference["pressure_side"] != side
                or digest(reference["arrays"]["path"]) != reference["arrays"]["sha256"]
            ):
                raise ValueError("exact same original half-stiffness donor and course required")
            pins[reference["arrays"]["path"]] = reference["arrays"]["sha256"]
            with np.load(reference["arrays"]["path"], allow_pickle=False) as prior:
                if (
                    prior["donor_integration_state"].tobytes() != donor.tobytes()
                    or prior["parent_end_integration_state"].tobytes()
                    != arrays["parent_end_integration_state"].tobytes()
                    or prior["parent_kp"].tobytes() != kp.tobytes()
                    or prior["unchanged_kd"].tobytes() != kd.tobytes()
                ):
                    raise ValueError("new default law must reproduce the qualified original parent")
            if not shaping["enabled"] and (
                arrays["parent_end_integration_state"].tobytes()
                != arrays["candidate_end_integration_state"].tobytes()
            ):
                raise ValueError("inactive stiffness intervention must be exactly inert")
            equal(
                arrays["parent_qpos_after_substep"][-1],
                trace["canonical_qpos"][frame + 1, 0],
                "original parent interval",
            )
            array_path = root / f"case-{case}.npz"
            with array_path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            row = dict(
                source_receipt_hash=r["receipt_hash"],
                frame=frame,
                first_foot_contact_from_observed_prefix=first,
                pressure_side=side,
                stiffness_contraction_fraction=0.5,
                shaping=shaping,
                original_uniform_reference_measures=reference["measures"],
                damping_changed=False,
                actual_changed_joint_indices=np.flatnonzero(
                    np.any(matrix != np.diag(kp), axis=1)
                ).tolist(),
                measures=measures,
                arrays=dict(path=str(array_path), sha256=digest(array_path)),
                donor_state_hash=hash_bytes(donor.tobytes()),
                same_donor_for_both_actual_interventions=True,
                full_horizon_task_outcome_measured=False,
                new_neural_inferences=0,
            )
            write_once(root / f"case-{case}-complete.json", row)
            rows.append(row)
            print(
                "ONE_INTERVAL_PROBE_COMPLETE",
                case,
                side,
                row["measures"],
                flush=True,
            )
        forecasts = sum(t.integrated_steps for t in teachers)
        if boot != 40 or actual != 80 or forecasts != 80:
            raise ValueError("fixed source/actual/forecast budgets differ")
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError("source/input bytes changed")
        result = dict(
            schema="soccer.rsi.directional_contact_compliance_interval_probe.v1",
            declaration_hash=declared["report_hash"],
            rows=rows,
            bootstrap_steps=boot,
            actual_short_interval_steps=actual,
            attempted_bootstrap_steps=attempted_boot,
            attempted_actual_short_interval_steps=attempted_actual,
            forecast_steps=forecasts,
            attempted_forecast_steps=sum(t.attempted_steps for t in teachers),
            new_full_horizon_policy_trials=0,
            new_neural_inferences=0,
            new_optimizer_updates=0,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(root / "complete.json", result)
        print(result["report_hash"], flush=True)
    except BaseException as error:
        write_once(
            root / "failure.json",
            dict(
                type=type(error).__name__,
                message=str(error),
                bootstrap_steps=boot,
                actual_short_interval_steps=actual,
                attempted_bootstrap_steps=attempted_boot,
                attempted_actual_short_interval_steps=attempted_actual,
                forecast_steps=sum(t.integrated_steps for t in teachers),
                attempted_forecast_steps=sum(t.attempted_steps for t in teachers),
                automatic_retry_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
