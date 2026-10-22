"""Whole-world independent own98 NN and every2ms torque/body transfer replay."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "qualification-root", "initial-exam-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("producer-head", "expected-head"):
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
            raise ValueError("clean independent replay source required")

    clean()
    if a.output.exists():
        raise ValueError("unused whole-world audit output required")
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    if any(p.is_symlink() for p in (rp, dp)) or (a.root / "failed.json").exists():
        raise ValueError("closed ordinary source transfer required")
    r, d = [json.loads(p.read_text()) for p in (rp, dp)]
    if (
        r["status"] != "VELOCITY_PRIOR_PAID_BODY_TRANSFER_CLOSED_NOT_FOOTBALL_QUALIFIED"
        or r["declaration_hash"] != digest(dp)
        or r["source_head"] != a.producer_head
        or d["source_head"] != a.producer_head
        or r["native_steps"] != 48000
        or r["neural_calls"] != 4800
        or r["all_whole_repeats_exact"] is not True
        or d["protocol"] != "velocity.unchanged_paid_body_software_PD_transfer.v1"
        or d["all_initial_ball_world"] != [3.0, 3.0, 0.11]
        or d["pulse_onset"] != 50
        or d["pulse_end"] != 150
        or d["cases"] != 8
        or d["repeats"] != 2
        or r["input_file_hashes"] != d["input_file_hashes"]
        or any(
            v.get(k) is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("exact full two-repeat eight-case transfer declaration required")
    binary = Path(r["compiled_model_path"])
    if binary.is_symlink() or binary.parent != a.root or digest(binary) != r["compiled_model_hash"]:
        raise ValueError("confined exact compiled simulation body required")
    pins = dict(r["input_file_hashes"])
    for p in (rp, dp, binary):
        pins[str(p)] = digest(p)
    qp, ip = [root / "complete.json" for root in (a.qualification_root, a.initial_exam_root)]
    if any(str(p) not in r["input_file_hashes"] for p in (qp, ip)):
        raise ValueError("source-bound qualified codec and actual initial exam required")
    qual, exam = [json.loads(p.read_text()) for p in (qp, ip)]
    artifact = next(
        x for x in exam["worlds"][4]["artifacts"] if x["raw_path"].endswith("/world-initial.npz")
    )
    recorded = a.initial_exam_root / artifact["raw_path"]
    if (
        recorded.is_symlink()
        or not recorded.resolve().is_relative_to(a.initial_exam_root.resolve())
        or pins.get(str(recorded)) != artifact["raw_hash"]
    ):
        raise ValueError("exact source snapshot required")
    onnx = next(Path(f) for f in r["input_file_hashes"] if f.endswith("/exported/policy.onnx"))
    config = next(Path(f) for f in r["input_file_hashes"] if f.endswith("/params/deploy.yaml"))
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    steps, calls = 0, 0
    try:
        import mujoco
        import numpy as np

        from rosclaw_soccer.physics.g1_velocity_observation import world_command_to_body
        from rosclaw_soccer.providers.g1.velocity_policy import G1VelocityPolicy
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.velocity_torque_transfer import (
            require_torque_encoding,
            transfer_cases,
        )
        from rosclaw_soccer.rsi.velocity_transfer_audit import (
            summarize_velocity_raw,
            validate_velocity_raw,
        )

        if mujoco.__version__ != "3.5.0" or mujoco.get_mjcb_control() is not None:
            raise ValueError("callback-free pinned engine required")
        model = mujoco.MjModel.from_binary_path(str(binary))
        layout = NativeVelocityLayout(model, robot_prefix="", ball_body_name="soccer_ball")
        effort = require_torque_encoding(model, layout)
        robot = set()
        for b in range(model.nbody):
            parent = b
            while parent not in (0, layout.pelvis):
                parent = int(model.body_parentid[parent])
            if parent == layout.pelvis:
                robot.add(b)
        with np.load(recorded, allow_pickle=False) as z:
            recorded_state = {k: z[k].copy() for k in z.files}
        results = []
        for case, (initial, speed) in enumerate(transfer_cases()):
            first_raw = None
            for repeat in range(2):
                folder = a.root / f"case{case}-repeat{repeat}"
                cp, sp, tp = folder / "complete.json", folder / "initial.npz", folder / "native.npz"
                if any(p.is_symlink() for p in (cp, sp, tp)):
                    raise ValueError("ordinary per-world artifacts required")
                source = json.loads(cp.read_text())
                if digest(sp) != source["initial_hash"] or digest(tp) != source["raw_hash"]:
                    raise ValueError("world artifact hashes changed")
                pins.update({str(p): digest(p) for p in (cp, sp, tp)})
                with np.load(tp, allow_pickle=False) as z:
                    raw = {k: z[k].copy() for k in z.files}
                if validate_velocity_raw(raw) != 300:
                    raise ValueError(
                        "complete world required; no padding or partial transfer claim"
                    )
                if first_raw is None:
                    first_raw = raw
                elif any(not np.array_equal(v, first_raw[k]) for k, v in raw.items()):
                    raise ValueError("producer whole repeats differ")
                actor = G1VelocityPolicy(
                    onnx,
                    config,
                    expected_model_hash=pins[str(onnx)],
                    expected_config_hash=pins[str(config)],
                )
                if actor.contract_hash != qual["policy_contract_hash"]:
                    raise ValueError("qualified unchanged98D actor required")
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
                    raise ValueError("no external assistance required")
                data.qpos[layout.bq : layout.bq + 7] = [3, 3, 0.11, 1, 0, 0, 0]
                data.qvel[layout.bv : layout.bv + 6] = 0
                mujoco.mj_forward(model, data)
                with np.load(sp, allow_pickle=False) as z:
                    if set(z.files) != {
                        "qpos",
                        "qvel",
                        "ctrl",
                        "qacc_warmstart",
                        "mocap_pos",
                        "mocap_quat",
                        "time",
                    }:
                        raise ValueError("complete declared initial boundary required")
                    for k in z.files:
                        own = np.array(data.time) if k == "time" else getattr(data, k)
                        if not np.array_equal(own, z[k]):
                            raise ValueError(
                                "source initial state not reconstructible from declaration"
                            )
                q, _ = layout.canonical(data)
                w, x, y, z = q[3:7]
                yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                pulse = (float(speed * np.cos(yaw)), float(speed * np.sin(yaw)), 0.0)
                kp, kd = [
                    np.asarray(v, np.float64)
                    for v in (actor.params.stiffness, actor.params.damping)
                ]
                warning = tuple(int(w.number) for w in data.warning)
                for frame in range(300):
                    mujoco.mj_forward(model, data)
                    q, v = layout.canonical(data)
                    if not np.array_equal(q, raw["qpos_before"][frame]) or not np.array_equal(
                        v, raw["qvel_before"][frame]
                    ):
                        raise ValueError("whole-world state differs before neural observation")
                    command = pulse if 50 <= frame < 150 else (0.0, 0.0, 0.0)
                    cmd = world_command_to_body(q, command)
                    omega = layout.angular_velocity_body(data)
                    target = actor.propose(q, v, omega, cmd, frame=frame)
                    calls += 1
                    if actor.phase != raw["phase"][frame]:
                        raise ValueError("own neural gait phase differs")
                    for k, value in dict(
                        observation=actor.last_observation,
                        action=actor.last_action,
                        target=target,
                        command_body=cmd,
                        command_world=np.asarray(command),
                        omega=omega,
                    ).items():
                        if not np.array_equal(value, raw[k][frame]):
                            raise ValueError("independent own NN or command codec differs: " + k)
                    for sub in range(10):
                        q, v = layout.canonical(data)
                        # Independent expression, not the producer's decoder.
                        torque = np.minimum(
                            effort, np.maximum(-effort, kp * (target - q[7:36]) - kd * v[6:35])
                        )
                        if not np.array_equal(torque, raw["torque"][frame, sub]):
                            raise ValueError("software PD/effort authority differs")
                        data.ctrl[layout.ai] = torque
                        clock = float(data.time)
                        mujoco.mj_step(model, data)
                        steps += 1
                        q, v = layout.canonical(data)
                        if (
                            abs(data.time - clock - 0.002) > 1e-10
                            or tuple(int(w.number) for w in data.warning) != warning
                            or not np.array_equal(q, raw["qpos"][frame, sub])
                            or not np.array_equal(v, raw["qvel"][frame, sub])
                            or not np.array_equal(
                                data.actuator_force[layout.ai], raw["force"][frame, sub]
                            )
                        ):
                            raise ValueError("independent native state/effort replay differs")
                        force = 0.0
                        for j, c in enumerate(data.contact):
                            b1, b2 = [int(model.geom_bodyid[g]) for g in c.geom]
                            if (b1 == layout.ball and b2 in robot) or (
                                b2 == layout.ball and b1 in robot
                            ):
                                wrench = np.zeros(6)
                                mujoco.mj_contactForce(model, data, j, wrench)
                                force = max(force, float(np.linalg.norm(wrench[:3])))
                        if force != raw["ball_force"][frame, sub]:
                            raise ValueError("native contact classification differs")
                if actor.neural_calls != 300:
                    raise ValueError("all own neural calls required")
                own = summarize_velocity_raw(raw, case, initial, speed)
                if own != r["results"][case] or any(source[k] != v for k, v in own.items()):
                    raise ValueError("independently recomputed transfer outcome differs")
                results.append(dict(**own, repeat=repeat))
                print(
                    json.dumps(dict(case=case, repeat=repeat, status="OWN98_WHOLE_NATIVE_CHECKED")),
                    flush=True,
                )
        check_pins(pins)
        clean()
        if steps != 48000 or calls != 4800:
            raise ValueError("entire explicit transfer bank required")
        write_once(
            a.output / "complete.json",
            dict(
                status="VELOCITY_PAID_BODY_OWN98_WHOLE_WORLD_NATIVE_AUDIT_PASS",
                source_head=a.expected_head,
                producer_head=a.producer_head,
                source_hash=digest(rp),
                input_file_hashes=pins,
                results=results,
                own_native_steps=steps,
                own_neural_calls=calls,
                independent_whole_world_replay=True,
                optimizer_updates=0,
                football_skill_proven=False,
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
                own_native_steps=steps,
                own_neural_calls=calls,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
