"""Fixed one-body native feasibility study, NOT expanded-domain qualification.

Six declared commands, paired fresh re-execution, and exact old/new codec
overlap at two legacy points. No learned policy, promotion or football claim.
"""

import argparse
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("anchor", "reference-source", "core-source-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("anchor-hash", "reference-hash", "core-expected-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--heading-hold", action="store_true")
    a = p.parse_args()
    from scripts.rsi_active_core_source import pin_active_core
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    path = a.anchor / "complete.json"
    if a.output.exists() or path.is_symlink() or digest(path) != a.anchor_hash:
        raise ValueError("new envelope output and exact native identity anchor required")
    anchor = json.loads(path.read_text())
    if (
        anchor["status"] != "NATIVE8_CONTACT_TEACHER_AUTHORITY_MEASURED_NOT_LEARNED"
        or not anchor["fresh_reexecution_exact"]
        or len(anchor["records"]) != 4
        or not all(r["healthy_full_horizon"] for r in anchor["records"])
        or not all(all(r["teacher_parent_identity"].values()) for r in anchor["records"])
    ):
        raise ValueError("complete actual parent identity anchor required")
    pins = dict(anchor["input_file_hashes"])
    pins[str(path)] = a.anchor_hash
    if a.reference_source.is_symlink() or digest(a.reference_source) != a.reference_hash:
        raise ValueError("exact separate training-source reference required")
    pins[str(a.reference_source)] = a.reference_hash
    pin_active_core(a.core_source_root, a.core_expected_head, pins)
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
    from rosclaw_soccer.providers.g1.experimental_velocity_envelope import (
        ExperimentalVelocityEnvelopeStream,
    )
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.sim.contracts import hash_json

    def unique(suffix):
        found = [Path(k) for k in pins if k.endswith(suffix)]
        if len(found) != 1:
            raise ValueError("unique exact anchor asset required: " + suffix)
        return found[0]

    scene = unique("/cached-standard/declared-scene.xml")
    model_path = unique("/velocity/v0/exported/policy.onnx")
    config = unique("/velocity/v0/params/deploy.yaml")
    motion = unique("/policy/robonaldo/model/freekick_motion.npz")
    with np.load(motion) as data:
        height = float(data["body_pos_w"][0, 0, 2])
    prefix = "athlete/"
    xml = shared_native_scene(scene.read_text(), prefixes=(prefix,))
    courses = (
        ("standing", (0.0, 0.0, 0.0), True),
        ("forward-1", (1.0, 0.0, 0.0), True),
        ("forward-1p5", (1.5, 0.0, 0.0), False),
        ("forward-2", (2.0, 0.0, 0.0), False),
        ("lateral-positive-0p8", (0.0, 0.8, 0.0), False),
        ("lateral-negative-0p8", (0.0, -0.8, 0.0), False),
    )
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="native_one_body_experimental_velocity_heading_hold.v2"
            if a.heading_hold
            else "native_one_body_experimental_velocity_envelope.v1",
            source_head=a.expected_head,
            courses=courses,
            horizon_controls=600,
            control_hz=50,
            physics_hz=500,
            warmup_controls=50,
            maximum_command_change_per_control=0.05,
            heading_feedback_gain=2.0 if a.heading_hold else 0.0,
            heading_reference_rad=0.0,
            maximum_worlds=16,
            maximum_native_steps=96000,
            assessment_last_controls=250,
            point_tracking_tolerance_mps=0.25,
            point_heading_tolerance_rad=0.35,
            physical_health_contract="unchanged_SharedNativeMotorWorld",
            ball_initial_position_m=[30, 30, 0.11],
            no_football_interaction_claim=True,
            original_domain_and_yaml_unchanged=True,
            all_domain_points_qualified=False,
            hardware_authorized=False,
        ),
    )
    rows, charged = [], 0
    try:
        for name, requested, overlap in courses:
            for kind in ("legacy", "experimental") if overlap else ("experimental",):
                reference = None
                for repeat in (0, 1):
                    m = mujoco.MjModel.from_xml_string(xml)
                    d = mujoco.MjData(m)
                    kwargs = dict(
                        expected_model_hash=pins[str(model_path)],
                        expected_config_hash=pins[str(config)],
                        horizon_controls=600,
                    )
                    if kind == "experimental":
                        stream = ExperimentalVelocityEnvelopeStream(
                            model_path,
                            config,
                            reference_source=a.reference_source,
                            expected_reference_hash=a.reference_hash,
                            **kwargs,
                        )
                    else:
                        stream = ContinuousVelocityStream(model_path, config, **kwargs)
                    lo = NativeVelocityLayout(m, robot_prefix=prefix, ball_body_name="soccer_ball")
                    d.qpos[lo.rq : lo.rq + 7] = [0, 0, height, 1, 0, 0, 0]
                    d.qpos[lo.qi] = np.asarray(stream.params.defaults)
                    d.qpos[lo.bq : lo.bq + 7] = [30, 30, 0.11, 1, 0, 0, 0]
                    mujoco.mj_forward(m, d)
                    effort = 0.85 * np.asarray(G1_HARD_TORQUE_LIMITS)
                    w = SharedNativeMotorWorld(
                        m,
                        d,
                        players={
                            prefix: NativePDParameters(
                                np.asarray(stream.params.stiffness),
                                np.asarray(stream.params.damping),
                                effort,
                            )
                        },
                        ball_body_name="soccer_ball",
                    )
                    raw = {k: [] for k in ("qpos", "qvel", "torque", "ball_forces")}
                    controls = {
                        k: []
                        for k in (
                            "commands",
                            "observations98",
                            "targets29",
                            "velocity_body_after",
                            "yaw_after",
                        )
                    }
                    command = np.zeros(3, np.float32)
                    for frame in range(600):
                        q, v = lo.canonical(d)
                        destination = np.zeros(3) if frame < 50 else np.array(requested)
                        if a.heading_hold and frame >= 50:
                            qw, qx, qy, qz = q[3:7]
                            measured_yaw = math.atan2(
                                2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)
                            )
                            error = math.atan2(math.sin(-measured_yaw), math.cos(-measured_yaw))
                            destination[2] = np.clip(2 * error, -1, 1)
                        command = (
                            command.astype(np.float64) + np.clip(destination - command, -0.05, 0.05)
                        ).astype(np.float32)
                        target = stream.propose(
                            q, v, lo.angular_velocity_body(d), command, frame=frame
                        )
                        step = w.advance({prefix: target})
                        charged += 10
                        for key in raw:
                            raw[key].append(step[key])
                        q, v = lo.canonical(d)
                        qw, qx, qy, qz = q[3:7]
                        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                        c, s = math.cos(yaw), math.sin(yaw)
                        for key, value in (
                            ("commands", command),
                            ("observations98", stream.last_observation),
                            ("targets29", target),
                            (
                                "velocity_body_after",
                                np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]]),
                            ),
                            ("yaw_after", np.array(yaw)),
                        ):
                            controls[key].append(value.copy())
                        if w.failed:
                            break
                    arrays = {k: np.concatenate(v) for k, v in raw.items()}
                    arrays.update({k: np.asarray(v) for k, v in controls.items()})
                    exact, legacy_exact = None, None
                    if repeat:
                        with np.load(a.output / reference["path"]) as previous:
                            exact = {
                                k: bool(np.array_equal(v, previous[k])) for k, v in arrays.items()
                            }
                        if not all(exact.values()):
                            raise ValueError("experimental source/fresh re-execution changed")
                    if overlap and kind == "experimental":
                        with np.load(a.output / f"{name}-legacy-{repeat}.npz") as old:
                            legacy_exact = {
                                k: bool(np.array_equal(v, old[k])) for k, v in arrays.items()
                            }
                        if not all(legacy_exact.values()):
                            raise ValueError("new codec changed old-domain native physics")
                    healthy = w.control_steps == 600 and not w.failed
                    mean_velocity = arrays["velocity_body_after"][-250:].mean(axis=0)
                    tracking_error = float(np.linalg.norm(mean_velocity - np.array(requested[:2])))
                    max_heading = float(np.max(abs(arrays["yaw_after"][-250:])))
                    force_bound = bool(
                        np.max(abs(arrays["torque"][:, lo.ai]) / effort) <= 1 + 1e-12
                    )
                    path = a.output / f"{name}-{kind}-{repeat}.npz"
                    np.savez_compressed(path, **arrays)
                    artifact = dict(path=path.name, hash=digest(path))
                    if repeat == 0:
                        reference = artifact
                    row = dict(
                        course=name,
                        kind=kind,
                        repeat=repeat,
                        profile_hash=str(stream.contract_hash),
                        body_binding_hash=str(hash_json(lo.profile())),
                        controls=w.control_steps,
                        native_steps=w.native_steps,
                        neural_calls=stream.neural_calls,
                        healthy_full_horizon=healthy,
                        assessment_complete=healthy,
                        assessment_tail_controls_used=min(250, w.control_steps),
                        mean_available_tail_body_velocity_mps=mean_velocity.tolist(),
                        mean_tracking_error_mps=tracking_error,
                        max_last250_heading_error_rad=max_heading,
                        point_feasible=healthy
                        and force_bound
                        and tracking_error <= 0.25
                        and max_heading <= 0.35,
                        actual_effort_bound_pass=force_bound,
                        no_ball_contact=not bool(np.any(arrays["ball_forces"] > 1)),
                        fresh_reexecution_exact=exact,
                        legacy_overlap_exact=legacy_exact,
                        artifact=artifact,
                    )
                    rows.append(row)
                    write_once(a.output / f"record-{name}-{kind}-{repeat}.json", row)
                    print(json.dumps(row), flush=True)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="EXPERIMENTAL_VELOCITY_POINTS_MEASURED_NOT_A_DOMAIN_QUALIFICATION",
                source_head=a.expected_head,
                input_file_hashes=pins,
                records=rows,
                native_steps=charged,
                optimizer_updates=0,
                original_qualified_domain_unchanged=True,
                full_command_domain_qualified=False,
                football_skill_proven=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                completed=len(rows),
                known_native_steps=charged,
                blind_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
