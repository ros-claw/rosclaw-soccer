"""Explicit new native SIM transfer: velocity prior on the unchanged PAiD body."""

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in (
        "body-xml",
        "qualification-root",
        "velocity-reference-root",
        "initial-exam-root",
        "output",
    ):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("body-xml-hash", "qualification-hash", "initial-exam-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen body-transfer source required")

    clean()
    if a.output.exists():
        raise ValueError("unused native transfer output; never automatic retry")
    qr = a.qualification_root / "complete.json"
    ir = a.initial_exam_root / "complete.json"
    if any(
        p.is_symlink() or digest(p) != h
        for p, h in (
            (a.body_xml, a.body_xml_hash),
            (qr, a.qualification_hash),
            (ir, a.initial_exam_hash),
        )
    ):
        raise ValueError("exact source body, model codec and recorded initial exam required")
    qual, initial_exam = [json.loads(p.read_text()) for p in (qr, ir)]
    if (
        qual["status"] != "G1_VELOCITY_POLICY_CPP_CODEC_NN_QUALIFIED_NOT_PHYSICS"
        or not qual["cpp_float32_target_exact"]
        or initial_exam["status"] != "PAID_ROUTER_TRAIN20_WORLDS_CLOSED_PENDING_AUDIT"
        or initial_exam["repeats_exact"] is not True
        or initial_exam["summary"]["stable"] != 20
    ):
        raise ValueError("qualified unchanged velocity codec and original recorded world required")
    policy_dir = a.velocity_reference_root / "deploy/robots/g1/config/policy/velocity/v0"
    onnx = policy_dir / "exported/policy.onnx"
    params = policy_dir / "params/deploy.yaml"
    artifact = next(
        x
        for x in initial_exam["worlds"][4]["artifacts"]
        if x["raw_path"].endswith("/world-initial.npz")
    )
    recorded = a.initial_exam_root / artifact["raw_path"]
    if (
        recorded.is_symlink()
        or not recorded.resolve().is_relative_to(a.initial_exam_root.resolve())
        or digest(recorded) != artifact["raw_hash"]
    ):
        raise ValueError("bound source case4 initial snapshot required")
    pins = {
        str(a.body_xml): a.body_xml_hash,
        str(qr): a.qualification_hash,
        str(ir): a.initial_exam_hash,
        str(recorded): artifact["raw_hash"],
        str(onnx): qual["input_file_hashes"][str(onnx)],
        str(params): qual["input_file_hashes"][str(params)],
    }
    if initial_exam["input_file_hashes"].get(str(a.body_xml)) != a.body_xml_hash:
        raise ValueError("recorded state must belong to exact body, not another MJLab export")
    root = ET.parse(a.body_xml).getroot()
    if root.findall(".//include"):
        raise ValueError("explicit flattened scene required for source asset pinning")
    compiler = root.find("compiler")
    meshdir = (
        Path(compiler.attrib.get("meshdir", "")) if compiler is not None else a.body_xml.parent
    )
    if not meshdir.is_absolute():
        meshdir = a.body_xml.parent / meshdir
    for node in root.iter():
        if "file" in node.attrib:
            if node.tag != "mesh":
                raise ValueError("unrecognized external scene asset; explicit pinning required")
            asset = meshdir / node.attrib["file"]
            pins[str(asset)] = digest(asset)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    clean()
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            protocol="velocity.unchanged_paid_body_software_PD_transfer.v1",
            cases=8,
            repeats=2,
            controls_per_case_at_most=300,
            maximum_native_steps=48000,
            maximum_neural_calls=4800,
            pulse_onset=50,
            pulse_end=150,
            all_initial_ball_world=[3.0, 3.0, 0.11],
            controller_gains="qualified_velocity_deploy_software_PD_not_PAiD_R2_gains",
            dynamics_unchanged=True,
            model_actuator_parameters_unchanged=True,
            no_old_body_certificate_inherited=True,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    steps, calls = 0, 0
    try:
        import mujoco
        import numpy as np

        from rosclaw_soccer.physics.g1_velocity_observation import world_command_to_body
        from rosclaw_soccer.providers.g1.velocity_policy import (
            G1VelocityPolicy,
            observation98,
            phase_delta,
            target29,
        )
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.velocity_torque_transfer import (
            decode_pd_torque,
            require_torque_encoding,
            transfer_cases,
        )

        if mujoco.__version__ != "3.5.0" or mujoco.get_mjcb_control() is not None:
            raise ValueError("pinned callback-free native engine required")
        model = mujoco.MjModel.from_xml_path(str(a.body_xml))
        if model.opt.timestep != 0.002:
            raise ValueError("unchanged native2ms scene required")
        layout = NativeVelocityLayout(model, robot_prefix="", ball_body_name="soccer_ball")
        effort = require_torque_encoding(model, layout)
        write_once(a.output / "body-profile.json", layout.profile())
        binary = a.output / "compiled-body.mjb"
        mujoco.mj_saveModel(model, str(binary), None)
        binary_hash = digest(binary)
        with np.load(recorded, allow_pickle=False) as z:
            recorded_state = {k: z[k].copy() for k in z.files}
        robot_bodies = set()
        for b in range(model.nbody):
            parent = b
            while parent not in (0, layout.pelvis):
                parent = int(model.body_parentid[parent])
            if parent == layout.pelvis:
                robot_bodies.add(b)
        results = []
        for case, (initial, speed) in enumerate(transfer_cases()):
            first_raw, first_result = None, None
            for repeat in range(2):
                actor = G1VelocityPolicy(
                    onnx,
                    params,
                    expected_model_hash=pins[str(onnx)],
                    expected_config_hash=pins[str(params)],
                )
                if actor.contract_hash != qual["policy_contract_hash"]:
                    raise ValueError("unchanged qualified velocity actor required")
                kp, kd = [
                    np.asarray(v, np.float64)
                    for v in (actor.params.stiffness, actor.params.damping)
                ]
                data = mujoco.MjData(model)
                if initial == "recorded_paid":
                    for k, v in recorded_state.items():
                        if k == "time":
                            data.time = float(v)
                        else:
                            getattr(data, k)[:] = v
                else:
                    data.qpos[layout.rq : layout.rq + 7] = [0, 0, 0.793, 1, 0, 0, 0]
                    data.qpos[layout.qi] = actor.params.defaults
                if data.qfrc_applied.any() or data.xfrc_applied.any():
                    raise ValueError("no borrowed external physical assistance allowed")
                data.qpos[layout.bq : layout.bq + 7] = [3, 3, 0.11, 1, 0, 0, 0]
                data.qvel[layout.bv : layout.bv + 6] = 0
                mujoco.mj_forward(model, data)
                initial_state = dict(
                    qpos=data.qpos.copy(),
                    qvel=data.qvel.copy(),
                    ctrl=data.ctrl.copy(),
                    qacc_warmstart=data.qacc_warmstart.copy(),
                    mocap_pos=data.mocap_pos.copy(),
                    mocap_quat=data.mocap_quat.copy(),
                    time=np.array(data.time),
                )
                q, _ = layout.canonical(data)
                w, x, y, z = q[3:7]
                yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                world = (speed * np.cos(yaw), speed * np.sin(yaw), 0.0)
                warning = tuple(int(w.number) for w in data.warning)
                rows, phase = [], phase_delta()
                last = np.zeros(29, np.float32)
                for frame in range(300):
                    mujoco.mj_forward(model, data)
                    q, v = layout.canonical(data)
                    command = world if 50 <= frame < 150 else (0.0, 0.0, 0.0)
                    cmd = world_command_to_body(q, command)
                    omega = layout.angular_velocity_body(data)
                    target = actor.propose(q, v, omega, cmd, frame=frame)
                    calls += 1
                    obs, phase = observation98(q, v, omega, cmd, last, phase, actor.params)
                    if not np.array_equal(obs, actor.last_observation) or not np.array_equal(
                        target, target29(actor.last_action, actor.params)
                    ):
                        raise ValueError("independent qualified observation/action codec differs")
                    last = actor.last_action.copy()
                    row = dict(
                        qpos_before=q,
                        qvel_before=v,
                        observation=obs,
                        action=last,
                        target=target,
                        command_body=cmd,
                        command_world=np.array(command),
                        omega=omega,
                        phase=phase,
                    )
                    native = []
                    for _ in range(10):
                        before_q, before_v = layout.canonical(data)
                        torque = decode_pd_torque(
                            target, before_q[7:36], before_v[6:35], kp, kd, effort
                        )
                        data.ctrl[layout.ai] = torque
                        clock = data.time
                        mujoco.mj_step(model, data)
                        steps += 1
                        if (
                            abs(data.time - clock - 0.002) > 1e-10
                            or tuple(int(w.number) for w in data.warning) != warning
                            or not np.array_equal(data.actuator_force[layout.ai], torque)
                        ):
                            raise ValueError(
                                "native force codec, timing or integration warning changed"
                            )
                        cq, cv = layout.canonical(data)
                        contact_force = 0.0
                        for contact_index, c in enumerate(data.contact):
                            b1, b2 = [int(model.geom_bodyid[g]) for g in c.geom]
                            if (b1 == layout.ball and b2 in robot_bodies) or (
                                b2 == layout.ball and b1 in robot_bodies
                            ):
                                wrench = np.zeros(6)
                                mujoco.mj_contactForce(model, data, contact_index, wrench)
                                contact_force = max(
                                    contact_force, float(np.linalg.norm(wrench[:3]))
                                )
                        native.append(
                            dict(
                                qpos=cq,
                                qvel=cv,
                                torque=torque,
                                force=data.actuator_force[layout.ai].copy(),
                                ball_force=contact_force,
                            )
                        )
                    row.update({k: np.asarray([n[k] for n in native]) for k in native[0]})
                    rows.append(row)
                    uq = 1 - 2 * (row["qpos"][:, 4] ** 2 + row["qpos"][:, 5] ** 2)
                    if row["qpos"][:, 2].min() < 0.5 or uq.min() < 0.6:
                        break
                raw = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
                if any(not np.isfinite(v).all() for v in raw.values()):
                    raise ValueError("finite actual native transfer traces required")
                if (
                    actor.neural_calls != len(rows)
                    or actor.next_frame != len(rows)
                    or not np.array_equal(raw["qpos_before"][1:], raw["qpos"][:-1, -1])
                    or not np.array_equal(raw["qvel_before"][1:], raw["qvel"][:-1, -1])
                ):
                    raise ValueError("complete continuous own neural/native trajectory required")
                upright = 1 - 2 * (raw["qpos"][:, :, 4] ** 2 + raw["qpos"][:, :, 5] ** 2)
                complete = len(rows) == 300
                stable = bool(
                    complete and raw["qpos"][:, :, 2].min() >= 0.65 and upright.min() >= 0.9
                )
                result = dict(
                    case=case,
                    initial=initial,
                    speed_mps=speed,
                    controls=len(rows),
                    body_stable=stable,
                    minimum_height_m=float(raw["qpos"][:, :, 2].min()),
                    minimum_upright=float(upright.min()),
                    pulse_mean_velocity_mps=raw["qvel_before"][100:150, :2].mean(0).tolist()
                    if len(rows) >= 150
                    else None,
                    stop_drift_m=float(
                        np.linalg.norm(raw["qpos_before"][-1, :2] - raw["qpos_before"][150, :2])
                    )
                    if complete
                    else None,
                    maximum_robot_ball_force_N=float(raw["ball_force"].max()),
                    native_steps=len(rows) * 10,
                    neural_calls=len(rows),
                )
                folder = a.output / f"case{case}-repeat{repeat}"
                folder.mkdir()
                np.savez_compressed(folder / "initial.npz", **initial_state)
                np.savez_compressed(folder / "native.npz", **raw)
                write_once(
                    folder / "complete.json",
                    dict(
                        **result,
                        raw_hash=digest(folder / "native.npz"),
                        initial_hash=digest(folder / "initial.npz"),
                    ),
                )
                if first_raw is None:
                    first_raw, first_result = raw, result
                elif result != first_result or any(
                    not np.array_equal(v, first_raw[k]) for k, v in raw.items()
                ):
                    raise ValueError("independent full transfer repeat differs")
                print(json.dumps(dict(**result, repeat=repeat)), flush=True)
            results.append(first_result)
        check_pins(pins)
        clean()
        if digest(binary) != binary_hash:
            raise ValueError("compiled transfer body changed")
        write_once(
            a.output / "complete.json",
            dict(
                status="VELOCITY_PRIOR_PAID_BODY_TRANSFER_CLOSED_NOT_FOOTBALL_QUALIFIED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                results=results,
                compiled_model_hash=binary_hash,
                compiled_model_path=str(binary),
                native_steps=steps,
                neural_calls=calls,
                optimizer_updates=0,
                all_whole_repeats_exact=True,
                unchanged_source_body_hash=a.body_xml_hash,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                native_steps=steps,
                neural_calls=calls,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
