"""Reintegrate a recorded cold-start candidate, including its failure.

Saved targets are only used for verification. This is not policy inference,
an independent episode, a team pass, or permission to run real hardware.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

from rosclaw_soccer.rsi.cftrack_motion_reference import load_cftrack_motion
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.omnicontact_initial_velocity import reference_initial_velocities
from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters, SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import check_pins, digest
from scripts.rsi_native_fresh_common import clean, source_pins


def validate_raw(raw: dict, controls: int) -> None:
    if type(controls) is not int or not 1 <= controls <= 600:
        raise ValueError("bounded actual control count required")
    for key, shape in (
        ("qpos", (controls * 10, 43)),
        ("qvel", (controls * 10, 41)),
        ("torque", (controls * 10, 29)),
        ("ball_forces", (controls * 10, 1, 3)),
        ("joint_targets", (controls, 29)),
    ):
        if key not in raw or raw[key].shape != shape or not np.isfinite(raw[key]).all():
            raise ValueError("complete finite source history required: " + key)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("qualification", "external-source", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("qualification-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--factorial-protocol", type=Path)
    p.add_argument("--causal-protocol", type=Path)
    p.add_argument("--projection-protocol", type=Path)
    a = p.parse_args()
    clean(a.expected_head)
    complete = a.qualification / "complete.json"
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or complete.is_symlink()
        or digest(complete) != a.qualification_hash
    ):
        raise ValueError("unused output and exact original candidate required")
    original = json.loads(complete.read_text())
    projected = original.get("projected_target", False)
    if (
        type(projected) is not bool
        or projected != (a.projection_protocol is not None)
        or (
            projected
            and (
                a.projection_protocol.is_symlink()
                or digest(a.projection_protocol) != original.get("projection_protocol_hash")
            )
        )
    ):
        raise ValueError("source-bound explicit target projection protocol required")
    causal = original.get("causal_ball_reference", False)
    if (
        type(causal) is not bool
        or causal != (a.causal_protocol is not None)
        or (
            causal
            and (
                a.causal_protocol.is_symlink()
                or digest(a.causal_protocol) != original.get("causal_protocol_hash")
            )
        )
    ):
        raise ValueError("source-bound explicit causal protocol required")
    ground = original.get("ground_reference_ball", False)
    no_spin = original.get("zero_reference_ball_spin", False)
    factorial = ground or no_spin
    if (
        type(ground) is not bool
        or type(no_spin) is not bool
        or factorial != (a.factorial_protocol is not None)
        or (
            factorial
            and (
                a.factorial_protocol.is_symlink()
                or digest(a.factorial_protocol) != original.get("factorial_protocol_hash")
            )
        )
    ):
        raise ValueError("source-bound explicit factorial protocol required")
    if (
        original.get("status") != "CFTRACK_NATIVE_BASELINE_RECORDED_NOT_PROMOTED"
        or original.get("runtime_failure") is not None
        or (
            original.get("reference_ball_initial_velocity") is not True
            and original.get("reference_all_initial_velocities") is not True
        )
        or (
            original.get("reference_ball_initial_velocity") is True
            and original.get("reference_all_initial_velocities") is True
        )
        or original.get("reference_source") != "NPZmotion"
        or original.get("promotion_authorized") is not False
    ):
        raise ValueError("actual unpromoted moving-ball source required")
    pins = dict(original["input_file_hashes"])
    pins[str(complete)] = a.qualification_hash
    raw_path = a.qualification / original["artifact"]["path"]
    if (
        raw_path.parent != a.qualification
        or raw_path.is_symlink()
        or digest(raw_path) != original["artifact"]["hash"]
    ):
        raise ValueError("exact confined actual source trajectory required")
    pins[str(raw_path)] = original["artifact"]["hash"]
    source_pins(pins)
    check_pins(pins)
    motion_path = Path(original["motion_source"])
    if str(motion_path) not in pins:
        raise ValueError("source-bound derived reference required")
    motion = load_cftrack_motion(motion_path)
    external = a.external_source.resolve()
    config = external / "policy/omnicontact/config/OmniContact.yaml"
    if str(config) not in pins:
        raise ValueError("source-bound CFtrack PD configuration required")
    config_data = yaml.safe_load(config.read_text())
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("one source-bound standard native scene required")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    if (
        compiled_model_hash(model)
        != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("exact compiled native body required")
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    state = mujoco.MjData(model)
    order = np.asarray(config_data["lab2mj"], dtype=int)
    state.qpos[layout.rq : layout.rq + 3] = motion["body_pos_w"][0, 0]
    state.qpos[layout.rq + 3 : layout.rq + 7] = motion["body_quat_w"][0, 0]
    state.qpos[layout.qi] = motion["joint_pos"][0, order]
    state.qpos[layout.bq : layout.bq + 3] = motion["object_pos_w"][0]
    state.qpos[layout.bq + 2] = max(0.11, float(state.qpos[layout.bq + 2]))
    state.qpos[layout.bq + 3 : layout.bq + 7] = motion["object_quat_w"][0]
    if original.get("reference_all_initial_velocities") is True:
        initial = reference_initial_velocities(motion, order)
        state.qvel[layout.rv : layout.rv + 3] = initial["root_linear"]
        state.qvel[layout.rv + 3 : layout.rv + 6] = initial["root_angular_local"]
        state.qvel[layout.vi] = initial["joints"]
        state.qvel[layout.bv : layout.bv + 3] = initial["ball_linear"]
        state.qvel[layout.bv + 3 : layout.bv + 6] = initial["ball_angular_local"]
    else:
        state.qvel[layout.bv : layout.bv + 3] = motion["object_lin_vel_w"][0]
    if factorial:
        from rosclaw_soccer.rsi.initial_ball_factorial import initial_ball_factorial

        position, spin = initial_ball_factorial(
            state.qpos[layout.bq : layout.bq + 3],
            state.qvel[layout.bv + 3 : layout.bv + 6],
            ground_ball=ground,
            zero_spin=no_spin,
        )
        state.qpos[layout.bq : layout.bq + 3] = position
        state.qvel[layout.bv + 3 : layout.bv + 6] = spin
    mujoco.mj_forward(model, state)
    params = NativePDParameters(
        np.array(config_data["kp_lab"], np.float32)[order],
        np.array(config_data["kd_lab"], np.float32)[order],
        0.85 * np.array(G1_HARD_TORQUE_LIMITS),
    )
    world = SharedNativeMotorWorld(
        model, state, players={"a/": params}, ball_body_name="soccer_ball"
    )
    with np.load(raw_path, allow_pickle=False) as archive:
        raw = {key: archive[key].copy() for key in archive.files}
    controls = original["controls"]
    validate_raw(raw, controls)
    a.output.mkdir(parents=True)
    matched, failure = 0, None
    try:
        for tick in range(controls):
            current = world.advance({"a/": raw["joint_targets"][tick]})
            for key in ("qpos", "qvel", "torque", "ball_forces"):
                if not np.array_equal(current[key], raw[key][tick * 10 : (tick + 1) * 10]):
                    raise ValueError(f"physical replay mismatch at {tick}: {key}")
            matched += 1
            if current["body_failed"]["a/"] and tick != controls - 1:
                raise ValueError("unexpected early body failure")
        if bool(world.failed) != original["body_failed"]:
            raise ValueError("terminal body failure differs from source")
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc))
    check_pins(pins)
    if factorial and digest(a.factorial_protocol) != original.get("factorial_protocol_hash"):
        raise ValueError("factorial protocol changed during replay")
    if causal and digest(a.causal_protocol) != original.get("causal_protocol_hash"):
        raise ValueError("causal protocol changed during replay")
    if projected and digest(a.projection_protocol) != original.get("projection_protocol_hash"):
        raise ValueError("projection protocol changed during replay")
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_PARTIAL_MOVING_SAVED_TARGET_REPLAY_EXACT"
            if matched == controls and failure is None
            else "CFTRACK_PARTIAL_MOVING_SAVED_TARGET_REPLAY_FAILED",
            source_head=a.expected_head,
            qualification_hash=a.qualification_hash,
            controls=controls,
            compared_controls=matched,
            native_steps=world.native_steps,
            recorded_source_body_failed=original["body_failed"],
            replay_body_failed=bool(world.failed),
            error=failure,
            recorded_targets_verification_only=True,
            independently_recomputed_neural_policy=False,
            task_success_not_implied=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
