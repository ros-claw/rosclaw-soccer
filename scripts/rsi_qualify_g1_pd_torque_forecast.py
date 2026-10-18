"""Four original PD replays plus 40 owned forecasts; no NN or new policy trial."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

FRAMES = (0, 74, 75, 77, 78, 80, 85, 90, 99, 100)


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
        or cfg["frames"] != list(FRAMES)
        or cfg["new_replay_substep_budget"] != 12000
        or cfg["new_forecast_substep_budget"] != 400
        or cfg.get("hardware_authorized") is not False
        or cfg.get("promotion_authorized") is not False
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("new frozen four-replay/400-forecast-step declaration required")
    pins = {str(args.config): digest(args.config)}
    for folder in (source / "src", source / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecaster
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_json
    from scripts.rsi_atomic_artifacts import write_once

    def sealed(path, key="report_hash"):
        d = json.loads(Path(path).read_text())
        if d[key] != hash_json({k: v for k, v in d.items() if k != key}):
            raise ValueError("source seal differs")
        pins[str(path)] = digest(path)
        return d

    original_config = json.loads(Path(cfg["source_configuration"]).read_text())
    pins[cfg["source_configuration"]] = digest(cfg["source_configuration"])
    upstream = sealed(Path(cfg["source_readback"]))
    old_complete = sealed(Path(original_config["output_root"]) / "complete.json")
    if (
        upstream.get("schema") != "soccer.rsi.swing_release_ablation_terminal_readback.v1"
        or upstream.get("complete_hash") != old_complete["report_hash"]
        or upstream.get("source_native_bodies") != 4
        or upstream.get("whole_input_and_witness_files_verified") is not True
        or upstream.get("reused_endpoints_authenticated_via_prior_complete_terminal_readbacks")
        is not True
    ):
        raise ValueError("complete independent upstream qualification required")
    receipts = []
    for index, item in enumerate(original_config["baseline_receipts"]):
        r = sealed(Path(item["path"]), "receipt_hash")
        if (
            digest(item["path"]) != item["file_hash"]
            or r["receipt_hash"] != item["receipt_hash"]
            or r != old_complete["reused_endpoints"][index]["receipt"]
        ):
            raise ValueError("exact four original endpoints, not replacement results required")
        for w in r["witnesses"].values():
            if digest(w["path"]) != w["sha256"]:
                raise ValueError("original witness bytes changed")
            pins[w["path"]] = w["sha256"]
        receipts.append(r)
    if len(receipts) != 4:
        raise ValueError("all four source endpoints required")
    root.mkdir()
    declaration = dict(
        cfg,
        input_file_hashes=pins,
        upstream_readback_hash=upstream["report_hash"],
        original_receipt_hashes=[r["receipt_hash"] for r in receipts],
        new_neural_inferences=0,
        new_closed_loop_policy_trials=0,
        new_optimizer_updates=0,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(root / "declaration.json", declaration)
    attempted, integrated = 0, 0
    forecasters = []
    rows = []

    def equal(a, b, name):
        if not np.allclose(a, b, rtol=0, atol=1e-8):
            raise ValueError("original PD replay differs: " + name)

    try:
        for case, r in enumerate(receipts):
            write_once(
                root / f"case-{case}-started.json",
                dict(receipt_hash=r["receipt_hash"], replay_attempt_charged=True),
            )
            model = mujoco.MjModel.from_binary_path(r["witnesses"]["world"]["path"])
            commitment = json.loads(Path(r["witnesses"]["commitment"]["path"]).read_text())
            kp = np.asarray(commitment["physics"]["canonical_pd_kp"], dtype=np.float64)
            kd = np.asarray(commitment["physics"]["canonical_pd_kd"], dtype=np.float64)
            f = G1PDTorqueForecaster(
                model,
                expected_world_hash=r["witnesses"]["world"]["sha256"],
                kp=kp,
                kd=kd,
                max_forecast_steps=100,
            )
            forecasters.append(f)
            with np.load(r["witnesses"]["trace"]["path"], allow_pickle=False) as archive:
                trace = dict(archive)
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
            bg = int(model.geom("ball_geom").id)
            bodies = [
                int(model.body(n).id)
                for n in (
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_ankle_pitch_link",
                    "right_ankle_pitch_link",
                    "left_knee_link",
                    "right_knee_link",
                )
            ]
            pelvis = int(model.body("pelvis").id)
            data = mujoco.MjData(model)
            q, v = trace["canonical_qpos"][0, 0], trace["canonical_qvel"][0, 0]
            data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7] = q[:7], q[7:36], q[36:43]
            data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6] = v[:6], v[6:35], v[35:41]
            mujoco.mj_forward(model, data)
            forecasts = []
            for frame in range(300):
                q = np.concatenate((data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7]))
                v = np.concatenate((data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6]))
                equal(q, trace["canonical_qpos"][frame, 0], "frame qpos")
                equal(v, trace["canonical_qvel"][frame, 0], "frame qvel")
                target = trace["joint_target_rad"][frame, 0]
                forecast = f.forecast(data, target, np.zeros(29)) if frame in FRAMES else None
                qs, vs, controls, actuals, heights = [], [], [], [], []
                peaks = np.zeros(6)
                for substep in range(10):
                    raw = (target - data.qpos[qi]) * kp - data.qvel[vi] * kd
                    tau = np.clip(
                        raw, -np.asarray(G1_HARD_TORQUE_LIMITS), np.asarray(G1_HARD_TORQUE_LIMITS)
                    )
                    equal(raw, trace["pd_requested_torque_nm"][frame, 0, substep], "raw PD")
                    equal(tau, trace["torque_nm"][frame, 0, substep], "actual canonical torque")
                    data.ctrl[ai] = tau
                    attempted += 1
                    mujoco.mj_step(model, data)
                    integrated += 1
                    equal(
                        data.actuator_force[ai],
                        trace["actual_actuator_force_nm"][frame, 0, substep],
                        "actuator force",
                    )
                    equal(
                        data.xpos[pelvis, 2],
                        trace["pelvis_z_per_substep_m"][frame, 0, substep],
                        "pelvis",
                    )
                    for index in range(data.ncon):
                        c = data.contact[index]
                        if bg not in (int(c.geom1), int(c.geom2)):
                            continue
                        other = int(c.geom2) if int(c.geom1) == bg else int(c.geom1)
                        body = int(model.geom_bodyid[other])
                        if body in bodies:
                            wrench = np.zeros(6)
                            mujoco.mj_contactForce(model, data, index, wrench)
                            channel = bodies.index(body)
                            peaks[channel] = max(peaks[channel], float(np.linalg.norm(wrench[:3])))
                    qs.append(
                        np.concatenate((data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7]))
                    )
                    vs.append(
                        np.concatenate((data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6]))
                    )
                    controls.append(tau)
                    actuals.append(data.actuator_force[ai].copy())
                    heights.append(float(data.xpos[pelvis, 2]))
                equal(peaks, trace["force_n"][frame, 0], "all six force channels")
                equal(
                    data.xpos[ball],
                    trace["ball_position_after_step_m"][frame, 0],
                    "ball cached after frame",
                )
                if forecast is not None:
                    pairs = (
                        (forecast.canonical_qpos_after_substep, np.asarray(qs)),
                        (forecast.canonical_qvel_after_substep, np.asarray(vs)),
                        (forecast.applied_control_nm, np.asarray(controls)),
                        (forecast.actual_actuator_force_nm, np.asarray(actuals)),
                        (forecast.pelvis_height_solver_epoch_m, np.asarray(heights)),
                    )
                    if any(a.tobytes() != b.tobytes() for a, b in pairs):
                        raise ValueError("selected zero forecast differs from actual frame bytes")
                    state = np.empty_like(forecast.end_integration_state)
                    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
                    if state.tobytes() != forecast.end_integration_state.tobytes():
                        raise ValueError("complete forecast integration state differs")
                    path = root / f"forecast-{case}-{frame}.npz"
                    with path.open("xb") as stream:
                        np.savez_compressed(stream, **vars(forecast))
                    forecasts.append(
                        dict(
                            frame=frame,
                            donor_state_hash=forecast.donor_integration_state_hash,
                            forecast=dict(path=str(path), sha256=digest(path)),
                            all_substep_state_control_and_full_end_state_bit_equal=True,
                        )
                    )
            rows.append(
                dict(
                    source_receipt_hash=r["receipt_hash"],
                    original_outcome=r["outcome"],
                    forecasts=forecasts,
                    replay_substeps=3000,
                    forecast_substeps=f.integrated_steps,
                )
            )
            write_once(root / f"case-{case}-complete.json", rows[-1])
        if (
            attempted != 12000
            or integrated != 12000
            or sum(f.integrated_steps for f in forecasters) != 400
        ):
            raise ValueError("whole original replay and forecast accounting required")
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError("source/input bytes changed")
        result = dict(
            schema="soccer.rsi.g1_pd_torque_forecast_qualification.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            replay_substeps=integrated,
            forecast_substeps=400,
            zero_offset_only=True,
            nonzero_teacher_qualified=False,
            new_neural_inferences=0,
            new_optimizer_updates=0,
            new_closed_loop_policy_trials=0,
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
                attempted_replay_steps=attempted,
                integrated_replay_steps=integrated,
                attempted_forecast_steps=sum(f.attempted_steps for f in forecasters),
                integrated_forecast_steps=sum(f.integrated_steps for f in forecasters),
                automatic_retry_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
