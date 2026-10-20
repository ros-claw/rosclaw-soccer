"""From0 continuous native body, external velocity history -> SONIC reference.

Uses velocity deployment PD for the ENTIRE body. This is an explicitly new
control system, not reuse of an old SONIC gain/motor certificate.
"""

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from scripts.rsi_probe_external_sonic_history import CaptureSession


class CountedCapture(CaptureSession):
    def __init__(self, session):
        super().__init__(session)
        self.attempts = 0

    def run(self, outputs, feed):
        self.attempts += 1
        return super().run(outputs, feed)


def historical_entry(pins):
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.native_entry_calibration import NativeCalibratedEntry

    prior = json.loads(
        next(Path(k) for k in pins if k.endswith("/entry-affordance.json")).read_text()
    )
    d = json.loads(
        next(
            Path(k)
            for k in pins
            if k.endswith("/rsi-native-entry-calibration-pilot-20261020/declaration.json")
        ).read_text()
    )
    parent = SkillEntryAffordance(
        tuple(prior["ball_offset_xy_m"]),
        prior["historical_frame"],
        prior["historical_forward_speed_mps"],
        prior["historical_artifact_hash"],
    )
    if (
        parent.contract_hash != prior["contract_hash"]
        or parent.contract_hash != d["fit"]["parent_affordance_hash"]
    ):
        raise ValueError("exact historical affordance lineage required")
    return NativeCalibratedEntry(
        tuple(d["entry_ball_offset_xy_m"]),
        parent.historical_frame,
        parent.historical_forward_speed_mps,
        parent.historical_artifact_hash,
        parent.contract_hash,
        d["fit"]["motion_configuration_hash"],
        d["fit_hash"],
    )


def run(
    a,
    pins,
    probe,
    params,
    model,
    config,
    world,
    baseline,
    course,
    source,
    sonic_root,
    *,
    body_factory=None,
    handoff_callback=None,
    reviewer=None,
):
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.physics.g1_velocity_observation import (
        G1VelocityObservation,
        world_command_to_body,
    )
    from rosclaw_soccer.providers.g1.sonic_external_history import (
        attach_external_history,
        prepare_external_history,
    )
    from rosclaw_soccer.providers.g1.sonic_reference_preparation import prepare_sonic_reference
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.providers.g1.sonic_wrist_quarantine import quarantine_wrist_reference
    from rosclaw_soccer.providers.g1.velocity_policy import G1VelocityPolicy
    from rosclaw_soccer.rsi.ball_entry_teacher import ball_entry_decision
    from rosclaw_soccer.rsi.switched_velocity_sonic_review import review_switched
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_probe_compact_source import load_arrays

    entry = historical_entry(pins)
    sensor = G1VelocityObservation(world, expected_world_hash=pins[str(a.world)])
    velocity = G1VelocityPolicy(
        model, config, expected_model_hash=pins[str(model)], expected_config_hash=pins[str(config)]
    )
    tracker = G1SonicRunupController(sonic_root, inference_threads=1)
    tracker._encoder, tracker._decoder = (
        CountedCapture(tracker._encoder),
        CountedCapture(tracker._decoder),
    )
    body = (G1TargetEpisode if body_factory is None else body_factory)(
        world,
        world_hash=pins[str(a.world)],
        kp=np.array(params.stiffness),
        kd=np.array(params.damping),
        defaults=np.array(params.defaults),
        course=course,
        record_all_ball_contacts=True,
    )
    physical, prefix = [], []
    handoff, receipt = None, None
    try:
        for f in range(300):
            q, v = body.canonical()
            decision = ball_entry_decision(q, v, frame=f, affordance=entry)
            if handoff is None and decision.ready:
                if f != probe["first_ready_global_frame"]:
                    raise ValueError(
                        "same frozen from0 actor must reproduce qualified entry boundary"
                    )
                handoff = f
                h = prepare_external_history(
                    np.concatenate(
                        [r["target_body_canonical_qpos_after_substep"][:, -1] for r in prefix[-10:]]
                    ),
                    np.concatenate(
                        [r["target_body_canonical_qvel_after_substep"][:, -1] for r in prefix[-10:]]
                    ),
                    np.concatenate([r["joint_target_rad"] for r in prefix[-10:]]),
                    defaults=tracker.default_angles,
                    scale=tracker._action_scale,
                )
                source_copy, _ = quarantine_wrist_reference(source, q[:36])
                prepared = prepare_sonic_reference(
                    source_copy,
                    q[:36],
                    source_frequency_hz=120,
                    entry_intervals=50,
                    source_evidence_hash=pins[str(a.csv)],
                )
                receipt = attach_external_history(
                    tracker,
                    h,
                    prepared.reference,
                    current_qpos=q,
                    current_qvel=v,
                    padded_reference_frames=309,
                )
                if receipt != probe["history_receipt"]:
                    raise ValueError(
                        "actual continuously reached history/reference must match NN probe"
                    )
                for k in baseline:
                    if not np.array_equal(np.concatenate([r[k] for r in prefix]), baseline[k][:f]):
                        raise ValueError(
                            "all22 prefix fields must reproduce original measured-entry body"
                        )
                write_once(a.output / "handoff.json", receipt)
                if handoff_callback is not None:
                    handoff_callback(body, f)
            if handoff is None:
                omega = sensor.angular_velocity_body(q, v)
                cmd = decision.command
                bc = world_command_to_body(q, cmd)
                target = velocity.propose(q, v, omega, bc, frame=f)
            else:
                local = f - handoff
                state = SimpleNamespace(qpos=q.copy(), qvel=v.copy())
                target = (
                    tracker.update(state, local)
                    if local < tracker.config.execution_frames
                    else tracker.update_recovery_extension(
                        state, local - tracker.config.execution_frames
                    )
                )
                if local == 0:
                    qualified = load_arrays(a.probe / "session0.npz")
                    enc, dec = tracker._encoder.calls[-1], tracker._decoder.calls[-1]
                    actual = dict(
                        encoder_input=next(iter(enc[0].values())),
                        encoder_output=enc[1][0],
                        decoder_input=next(iter(dec[0].values())),
                        decoder_output=dec[1][0],
                        joint_target=target,
                    )
                    if any(not np.array_equal(value, qualified[k]) for k, value in actual.items()):
                        raise ValueError(
                            "first actual policy inference differs from qualified probe"
                        )
            row = body.step(f, target)
            physical.append({k: value.copy() for k, value in row.items()})
            if handoff is None:
                row.update(
                    velocity_policy_observation=velocity.last_observation[None].copy(),
                    velocity_policy_action_hg=velocity.last_action[None].copy(),
                    command_body_mps=bc[None].copy(),
                    command_world_mps=np.array([cmd]),
                    angular_velocity_body_radps=omega[None].copy(),
                    gait_phase=np.array([velocity.phase], dtype=np.float32),
                    entry_target_root_xy_m=np.array([decision.target_root_xy_m]),
                    entry_error_m=np.array([decision.horizontal_error_m]),
                    entry_ready=np.array([decision.ready]),
                    entry_timed_out=np.array([decision.timed_out]),
                )
                prefix.append(row)
            else:
                qnext, vnext = body.canonical()
                tracker.observe(SimpleNamespace(qpos=qnext, qvel=vnext))
        if handoff is None:
            raise ValueError("same frozen actor failed the already qualified readiness boundary")
        native = {k: np.concatenate([r[k] for r in physical]) for k in physical[0]}
        neural = dict(
            encoder_input=np.concatenate(
                [next(iter(c[0].values())) for c in tracker._encoder.calls]
            ),
            encoder_output=np.concatenate([c[1][0] for c in tracker._encoder.calls]),
            decoder_input=np.concatenate(
                [next(iter(c[0].values())) for c in tracker._decoder.calls]
            ),
            decoder_output=np.concatenate([c[1][0] for c in tracker._decoder.calls]),
        )
        if (
            velocity.neural_calls != handoff
            or velocity.attempted_neural_calls != handoff
            or tracker._encoder.attempts != 300 - handoff
            or tracker._decoder.attempts != 300 - handoff
        ):
            raise ValueError("exact actual successful and attempted neural call counts required")
        result = (review_switched if reviewer is None else reviewer)(
            native, neural, params, tracker, world, handoff=handoff
        )
        np.savez_compressed(a.output / "native300.npz", **native)
        np.savez_compressed(
            a.output / "velocity-prefix.npz",
            **{k: np.concatenate([r[k] for r in prefix]) for k in prefix[0]},
        )
        np.savez_compressed(a.output / "sonic-neural.npz", **neural)
        result.update(
            body_report=body.review(),
            velocity_neural_calls=handoff,
            sonic_encoder_neural_calls=300 - handoff,
            sonic_decoder_neural_calls=300 - handoff,
            native_steps=3000,
            optimizer_steps=0,
            full22_prefix_exact=True,
            first_encoder_decoder_target_probe_exact=True,
            history_receipt=receipt,
            parameter_contract_hash=params.contract_hash,
            artifacts={
                name: digest(a.output / name)
                for name in ("native300.npz", "velocity-prefix.npz", "sonic-neural.npz")
            },
        )
        check_pins(pins)
        return result
    except Exception as error:
        if physical:
            np.savez_compressed(
                a.output / "partial-native.npz",
                **{k: np.concatenate([r[k] for r in physical]) for k in physical[0]},
            )
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                handoff_frame=handoff,
                actual_integrated_steps=body.integrated_steps,
                actual_attempted_steps=body.attempted_steps,
                actual_velocity_nn_calls=velocity.neural_calls,
                attempted_velocity_nn_calls=velocity.attempted_neural_calls,
                attempted_encoder_nn_calls=tracker._encoder.attempts,
                attempted_decoder_nn_calls=tracker._decoder.attempts,
                completed_encoder_nn_calls=len(tracker._encoder.calls),
                completed_decoder_nn_calls=len(tracker._decoder.calls),
                automatic_retry_authorized=False,
            ),
        )
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("produce", "reproduce"), required=True)
    for k in ("probe", "entry", "core-source", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("expected-head", "probe-unit", "expected-probe-hash"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--producer", type=Path)
    p.add_argument("--producer-unit")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import (
        source_suffix_hash_index,
        wait_for_terminal_collection,
    )
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("clean frozen motor pilot and unused output required")
    wait_for_terminal_collection(a.probe_unit)
    if digest(a.probe / "complete.json") != a.expected_probe_hash:
        raise ValueError("exact real closed neural bridge probe required")
    probe = json.loads((a.probe / "complete.json").read_text())
    if (
        probe["status"] != "EXTERNAL_SONIC_HISTORY_NN_PROBED_NOT_MOTOR_QUALIFIED"
        or probe["new_neural_calls"] != 4
        or not probe["two_independent_cpu_sessions_exact"]
        or not probe["first_target_within_native_joint_limits"]
        or not probe["all_four_parent_episodes_reviewed"]
    ):
        raise ValueError("complete history/proposal qualification required, not motor admission")
    pins = dict(probe["input_file_hashes"])
    pins.update({str(f): digest(f) for f in a.probe.rglob("*") if f.is_file()})
    index = source_suffix_hash_index(pins)
    for relative in (
        "src/rosclaw_soccer/providers/g1/sonic_external_history.py",
        "src/rosclaw_soccer/providers/g1/velocity_policy.py",
        "src/rosclaw_soccer/providers/g1/sonic_runup.py",
    ):
        if index.get(relative) != {digest(Path(relative))}:
            raise ValueError("actual qualified bridge and both frozen policies required")
    for name in ("complete.json", "declaration.json", "measured_entry/native300.npz"):
        if pins.get(str(a.entry / name)) != digest(a.entry / name):
            raise ValueError("exact native from0 baseline required")
    producer = None
    if a.mode == "reproduce":
        if a.producer is None or not a.producer_unit:
            raise ValueError("actual whole producer required")
        wait_for_terminal_collection(a.producer_unit, wait=True)
        producer = json.loads((a.producer / "complete.json").read_text())
        if (
            producer["status"] != "VELOCITY_SONIC_COMMON_PD_SWITCH_PRODUCED"
            or producer["source_head"] != a.expected_head
            or producer["native_steps"] != 3000
        ):
            raise ValueError("whole same-source native predecessor required")
        pins.update({str(f): digest(f) for f in a.producer.rglob("*") if f.is_file()})
    check_pins(pins)
    import mujoco
    import numpy as np

    from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
    from scripts.rsi_probe_compact_source import exact_artifact, load_arrays

    config = next(Path(k) for k in pins if k.endswith("/velocity/v0/params/deploy.yaml"))
    model = next(Path(k) for k in pins if k.endswith("/velocity/v0/exported/policy.onnx"))
    a.world = next(Path(k) for k in pins if k.endswith("/attempt-0004/compiled_model.mjb"))
    a.csv = next(Path(k) for k in pins if k.endswith("/BGI_Shooting_00001.csv"))
    sonic_root = next(Path(k).parent for k in pins if k.endswith("/planner_sonic.onnx"))
    params = VelocityPolicyParameters.load(config, expected_hash=pins[str(config)])
    baseline = load_arrays(a.entry / "measured_entry/native300.npz")
    course = tuple(json.loads((a.entry / "declaration.json").read_text())["course"])
    source = np.loadtxt(a.csv, delimiter=",", skiprows=1, dtype=np.float64)[466:767]
    pins.update(
        {
            str(f): digest(f)
            for dr in (Path.cwd() / "src", Path.cwd() / "scripts", a.core_source)
            for f in dr.rglob("*.py")
        }
    )
    world = mujoco.MjModel.from_binary_path(str(a.world))
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            mode=a.mode,
            maximum_native_steps=3000,
            maximum_neural_calls=600,
            optimizer_steps=0,
            initial_course=course,
            independent_initial_worlds=1,
            physical_state_reset=False,
            fixed_kp=params.stiffness,
            fixed_kd=params.damping,
            defaults=params.defaults,
            common_pd_is_velocity_not_sonic=True,
            gain_switch=False,
            original_full_entry_research_gate_unchanged=True,
            official_exam=False,
            hardware_authorized=False,
        ),
    )
    result = run(a, pins, probe, params, model, config, world, baseline, course, source, sonic_root)
    if producer:
        expected = {k: v for k, v in producer["result"].items() if k != "artifacts"}
        if (
            json.loads(json.dumps({k: v for k, v in result.items() if k != "artifacts"}))
            != expected
        ):
            raise ValueError("whole NN/native switched motor repeat metadata differs")
        for name in result["artifacts"]:
            exact_artifact(a.output / name, a.producer / name)
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="VELOCITY_SONIC_COMMON_PD_SWITCH_REPRODUCED"
            if producer
            else "VELOCITY_SONIC_COMMON_PD_SWITCH_PRODUCED",
            source_head=a.expected_head,
            result=result,
            native_steps=3000,
            new_optimizer_steps=0,
            goal_complete=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
