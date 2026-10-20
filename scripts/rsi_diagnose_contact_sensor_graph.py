"""Bounded sensor-vs-raw-solver diagnostic, stock graph/eager x three repeats.

No policy learning, no library mutation, no retuned thresholds. Every result,
including failed calibration, is preserved. Fixtures are NOT football skill.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("qualification-root", "reference-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("qualification-unit", "qualification-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and frozen diagnostic source required")
    wait_for_terminal_collection(a.qualification_unit)
    path = a.qualification_root / "complete.json"
    if digest(path) != a.qualification_hash:
        raise ValueError("immutable original qualification required for diagnosis")
    qual = json.loads(path.read_text())
    pins = dict(qual["input_file_hashes"])
    pins[str(path)] = a.qualification_hash
    for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            modes=["graph", "eager"],
            repeats_per_mode=3,
            worlds=8,
            substeps_per_fixture=10,
            maximum_gpu_world_steps=480,
            maximum_native_steps=480,
            maximum_trace_buffer_mib=64,
            sensor_protocol_and_tolerances_unchanged=True,
            library_mutation=False,
            new_optimizer_steps=0,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )
    env = None
    results = []
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

        sys.path.insert(0, str(a.reference_root))
        from scripts.rsi_ball_aware_sensor_calibration import calibrate
        from scripts.rsi_mjlab_ball_aware_backend import (
            BallAwareMotionEnv,
            make_ball_aware_training_cfg,
        )

        torch.set_num_threads(1)
        config = next(Path(k) for k in pins if k.endswith("/policy/velocity/v0/params/deploy.yaml"))
        params = VelocityPolicyParameters.load(config, expected_hash=pins[str(config)])
        for mode in ("graph", "eager"):
            cfg = make_ball_aware_training_cfg(
                parameters=params,
                command_mode="reference_position_velocity",
                command_point="pelvis",
                motion_dir=a.reference_root / "src/assets/soccer/motions/shooter",
                num_envs=8,
                seed=1601,
                balanced_initial_sources=True,
                football_rewards=True,
            )
            cfg.commands["motion"].sampling_mode = "uniform"
            env = BallAwareMotionEnv(cfg=cfg, device="cuda:0", render_mode=None)
            if not env.sim.use_cuda_graph:
                raise ValueError("explicit stock graph support required for paired diagnosis")
            env.sim.use_cuda_graph = mode == "graph"
            model = env.sim.mj_model
            if model.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL:
                raise ValueError("diagnostic raw decoder is explicitly pyramidal only")
            layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
            feet = [
                model.body("robot/" + side + "_ankle_roll_link").id for side in ("left", "right")
            ]
            sensor_adrs = [
                int(model.sensor("ball_" + side + "_foot_ball_force").adr[0])
                for side in ("left", "right")
            ]
            for repeat in range(3):
                env.reset(seed=1601)
                folder = a.output / f"{mode}-repeat{repeat}"
                folder.mkdir()
                result, passed = calibrate(env, layout, folder, trace=True)
                with np.load(folder / "sensor-calibration.npz", allow_pickle=False) as z:
                    primitive = np.zeros((8, 10, 2))
                    bad_addresses = 0
                    for step in range(10):
                        n = int(z["trace_nacon"][step, 0])
                        for cid in range(n):
                            g1, g2 = z["trace_contact_geom"][step, cid]
                            b1, b2 = model.geom_bodyid[[g1, g2]]
                            other = b2 if b1 == layout.ball else b1 if b2 == layout.ball else -1
                            if other not in feet:
                                continue
                            world = int(z["trace_contact_worldid"][step, cid])
                            dim = int(z["trace_contact_dim"][step, cid])
                            adr = int(z["trace_contact_efc_address"][step, cid, 0])
                            nefc = int(z["trace_nefc"][step, world])
                            width = 1 if dim == 1 else 2 * (dim - 1)
                            if adr < 0:
                                continue
                            if adr + width > nefc:
                                bad_addresses += 1
                                continue
                            pyramid = z["trace_efc_force"][step, world, adr : adr + width]
                            force = np.zeros(3)
                            force[0] = pyramid.sum()
                            if dim > 1:
                                force[1] = (pyramid[0] - pyramid[1]) * z["trace_contact_friction"][
                                    step, cid, 0
                                ]
                                force[2] = (pyramid[2] - pyramid[3]) * z["trace_contact_friction"][
                                    step, cid, 1
                                ]
                            side = feet.index(other)
                            primitive[world, step, side] = max(
                                primitive[world, step, side], np.linalg.norm(force)
                            )
                    sensed = np.stack(
                        [
                            np.linalg.norm(z["trace_sensordata"][:, :, adr : adr + 3], axis=2).T
                            for adr in sensor_adrs
                        ],
                        axis=-1,
                    )
                    result.update(
                        raw_solver_primitive_vs_cpu_max_error_n=float(
                            abs(primitive - z["native_force"]).max()
                        ),
                        raw_sensor_vs_primitive_max_error_n=float(abs(sensed - primitive).max()),
                        raw_sensor_vs_history_max_error_n=float(
                            abs(sensed - z["gpu_force_history"]).max()
                        ),
                        contact_addresses_outside_active_constraints=bad_addresses,
                    )
                    np.savez_compressed(
                        folder / "independent-force-decode.npz", primitive=primitive, sensor=sensed
                    )
                result.update(
                    mode=mode,
                    repeat=repeat,
                    calibration_passed=passed,
                    source_raw_hash=digest(folder / "sensor-calibration.npz"),
                    decode_hash=digest(folder / "independent-force-decode.npz"),
                )

                def finite(value):
                    if isinstance(value, dict):
                        return {k: finite(v) for k, v in value.items()}
                    if isinstance(value, list):
                        return [finite(v) for v in value]
                    if isinstance(value, float) and not math.isfinite(value):
                        return str(value)
                    return value

                result = finite(result)
                write_once(folder / "complete.json", result)
                results.append(result)
                print(json.dumps(result, allow_nan=False), flush=True)
            env.close()
            env = None
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="CONTACT_SENSOR_DIAGNOSTIC_NOT_QUALIFICATION",
                source_head=a.expected_head,
                input_file_hashes=pins,
                results=results,
                gpu_world_steps=480,
                native_steps=480,
                new_optimizer_steps=0,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                results=results,
                last_fixture_counts=getattr(env, "_ball_fixture_counts", {}),
                automatic_retry_authorized=False,
            ),
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
