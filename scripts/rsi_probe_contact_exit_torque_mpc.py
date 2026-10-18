"""Four snapshot-entry, one-control-interval MPC probes. NOT full G1 policies.

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
        or cfg["maximum_forecast_steps"] != 280
        or cfg["bootstrap_steps"] != 40
        or cfg["actual_probe_steps"] != 40
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
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.contact_exit_torque_mpc import ContactExitTorqueMPC
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
            teacher = ContactExitTorqueMPC(
                model,
                world_hash=r["witnesses"]["world"]["sha256"],
                kp=kp,
                kd=kd,
                forecast_budget_steps=70,
            )
            teachers.append(teacher)
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
            plan = teacher.plan(
                data,
                target,
                frame=frame,
                first_foot_contact_frame=first,
                force_history=past[-3:].copy(),
                previous_offset_nm=np.zeros(29),
                prior_nonfoot_contact=bool(np.any(past[:, 2:] > 1)),
            )
            donor = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(model, data, donor, mujoco.mjtState.mjSTATE_INTEGRATION)
            observed = []
            torques = []
            for _ in range(10):
                raw = (target - data.qpos[qi]) * kp - data.qvel[vi] * kd
                requested = raw + plan.offset_nm if np.any(plan.offset_nm) else raw
                tau = np.clip(requested, -limits, limits)
                data.ctrl[ai] = tau
                mujoco.mj_step(model, data)
                actual += 1
                observed.append(
                    np.concatenate((data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7]))
                )
                torques.append(data.actuator_force[ai].copy())
            teacher.verify_actual_interval(data, plan)
            if not np.any(plan.offset_nm):
                equal(
                    observed[-1], trace["canonical_qpos"][frame + 1, 0], "zero-control interval end"
                )
            array_path = root / f"case-{case}.npz"
            with array_path.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    donor_integration_state=donor,
                    actual_qpos_after_substep=np.asarray(observed),
                    actual_actuator_force_nm=np.asarray(torques),
                    chosen_offset_nm=plan.offset_nm,
                    expected_end_state=np.zeros(0)
                    if plan.expected_end_state is None
                    else plan.expected_end_state,
                )
            row = dict(
                source_receipt_hash=r["receipt_hash"],
                frame=frame,
                first_foot_contact_from_observed_prefix=first,
                teacher_contract=teacher.contract(),
                plan=plan.report,
                maximum_requested_offset_nm=float(np.max(abs(plan.offset_nm))),
                selected_forecast_actual_interval_bit_equal=plan.expected_end_state is not None,
                arrays=dict(path=str(array_path), sha256=digest(array_path)),
                donor_state_hash=hash_bytes(donor.tobytes()),
                counters=teacher.counters(),
                full_horizon_task_outcome_measured=False,
                new_neural_inferences=0,
            )
            write_once(root / f"case-{case}-complete.json", row)
            rows.append(row)
            print(
                "ONE_INTERVAL_PROBE_COMPLETE",
                case,
                plan.report["mode"],
                row["maximum_requested_offset_nm"],
                flush=True,
            )
        forecasts = sum(t.counters()["integrated_forecast_steps"] for t in teachers)
        if boot != 40 or actual != 40 or forecasts > 280:
            raise ValueError("fixed source/actual/forecast budgets differ")
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError("source/input bytes changed")
        result = dict(
            schema="soccer.rsi.contact_exit_torque_mpc_interval_probe.v1",
            declaration_hash=declared["report_hash"],
            rows=rows,
            bootstrap_steps=boot,
            actual_short_interval_steps=actual,
            forecast_steps=forecasts,
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
                forecast_steps=sum(t.counters()["integrated_forecast_steps"] for t in teachers),
                automatic_retry_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
