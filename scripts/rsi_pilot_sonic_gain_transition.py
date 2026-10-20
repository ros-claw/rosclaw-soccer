"""Constant-parity gated native ten-tick PD compatibility test, SIM only."""

import argparse
import json
import subprocess
from pathlib import Path


def numerical_result(row):
    return json.loads(json.dumps({k: v for k, v in row.items() if k != "artifacts"}))


def compare_constant_artifacts(actual, expected):
    import numpy as np

    from scripts.rsi_probe_compact_source import load_arrays

    for name in ("native300.npz", "velocity-prefix.npz", "sonic-neural.npz"):
        a, b = load_arrays(actual / name), load_arrays(expected / name)
        extra = set() if name == "sonic-neural.npz" else {"held_pd_kp", "held_pd_kd"}
        if set(a) != set(b) | extra or any(not np.array_equal(a[k], v) for k, v in b.items()):
            raise ValueError("constant protocol must reproduce every old physical/neural field")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("qualify", "produce", "reproduce"), required=True)
    for k in ("baseline", "baseline-repeat", "entry", "probe", "core-source", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in (
        "expected-head",
        "baseline-unit",
        "baseline-repeat-unit",
        "expected-baseline-hash",
        "expected-baseline-repeat-hash",
    ):
        p.add_argument("--" + k, required=True)
    for k in ("qualification", "producer"):
        p.add_argument("--" + k, type=Path)
        p.add_argument("--" + k + "-unit")
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
        raise ValueError("unused output and frozen new gain-protocol source required")
    pins = {}
    parents = []
    for root, unit, h, status in (
        (
            a.baseline,
            a.baseline_unit,
            a.expected_baseline_hash,
            "VELOCITY_SONIC_COMMON_PD_SWITCH_PRODUCED",
        ),
        (
            a.baseline_repeat,
            a.baseline_repeat_unit,
            a.expected_baseline_repeat_hash,
            "VELOCITY_SONIC_COMMON_PD_SWITCH_REPRODUCED",
        ),
    ):
        wait_for_terminal_collection(unit)
        if digest(root / "complete.json") != h:
            raise ValueError("exact two whole actual original native baselines required")
        c = json.loads((root / "complete.json").read_text())
        d = json.loads((root / "declaration.json").read_text())
        if (
            c["status"] != status
            or c["native_steps"] != 3000
            or c["source_head"] != d["source_head"]
            or c["result"]["body_report"]["integrated_steps"] != 3000
            or c["result"]["velocity_neural_calls"] != 30
            or c["result"]["sonic_encoder_neural_calls"] != 270
            or c["result"]["sonic_decoder_neural_calls"] != 270
        ):
            raise ValueError("closed whole from0 H300 baseline required")
        for k, v in d["input_file_hashes"].items():
            if k in pins and pins[k] != v:
                raise ValueError("conflicting original baseline inputs")
            pins[k] = v
        pins.update({str(f): digest(f) for f in root.rglob("*") if f.is_file()})
        parents.append(c)
    if numerical_result(parents[0]["result"]) != numerical_result(parents[1]["result"]):
        raise ValueError("original two complete baselines must agree")
    index = source_suffix_hash_index(pins)
    for rel in (
        "src/rosclaw_soccer/providers/g1/sonic_runup.py",
        "src/rosclaw_soccer/providers/g1/sonic_external_history.py",
        "src/rosclaw_soccer/providers/g1/velocity_policy.py",
        "src/rosclaw_soccer/physics/g1_target_episode.py",
        "src/rosclaw_soccer/rsi/ball_entry_teacher.py",
    ):
        if index.get(rel) != {digest(Path(rel))}:
            raise ValueError(
                "original integrator and both qualified policy/teacher sources required"
            )
    for path in (
        a.entry / "measured_entry/native300.npz",
        a.entry / "declaration.json",
        a.probe / "complete.json",
        a.probe / "session0.npz",
    ):
        if pins.get(str(path)) != digest(path):
            raise ValueError("actual original entry/NN bridge inputs required")
    if a.mode != "qualify":
        if a.qualification is None or not a.qualification_unit:
            raise ValueError("actual complete constant-gain parity must precede new gain trial")
        wait_for_terminal_collection(a.qualification_unit, wait=True)
        q = json.loads((a.qualification / "complete.json").read_text())
        if (
            q["status"] != "GAIN_SCHEDULE_CONSTANT_NATIVE_PARITY"
            or q["source_head"] != a.expected_head
            or q["native_steps"] != 3000
            or q["baseline_hash"] != a.expected_baseline_hash
            or not q["constant_all_original_fields_exact"]
        ):
            raise ValueError("same new source whole constant qualifier required")
        pins.update({str(f): digest(f) for f in a.qualification.rglob("*") if f.is_file()})
    producer = None
    if a.mode == "reproduce":
        if a.producer is None or not a.producer_unit:
            raise ValueError("explicit real ramp predecessor required")
        wait_for_terminal_collection(a.producer_unit, wait=True)
        producer = json.loads((a.producer / "complete.json").read_text())
        if (
            producer["status"] != "GAIN_SCHEDULE_LINEAR10_NATIVE_PRODUCED"
            or producer["source_head"] != a.expected_head
            or producer["native_steps"] != 3000
        ):
            raise ValueError("same whole ramp producer required")
        pins.update({str(f): digest(f) for f in a.producer.rglob("*") if f.is_file()})
    check_pins(pins)
    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_gain_scheduled_episode import (
        G1GainSchedule,
        G1GainScheduledEpisode,
    )
    from rosclaw_soccer.providers.g1.sonic_runup import _sonic_control_parameters
    from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
    from rosclaw_soccer.rsi.gain_scheduled_switch_review import review_gain_switch
    from scripts.rsi_pilot_velocity_sonic_switch import run
    from scripts.rsi_probe_compact_source import exact_artifact, load_arrays

    # Recheck full baseline arrays, rather than trusting repeated summary text.
    for name in ("native300.npz", "velocity-prefix.npz", "sonic-neural.npz"):
        exact_artifact(a.baseline / name, a.baseline_repeat / name)
        for root, c in ((a.baseline, parents[0]), (a.baseline_repeat, parents[1])):
            if digest(root / name) != c["result"]["artifacts"][name]:
                raise ValueError("original native/neural artifact hash mismatch")
    config = next(Path(k) for k in pins if k.endswith("/velocity/v0/params/deploy.yaml"))
    model = next(Path(k) for k in pins if k.endswith("/velocity/v0/exported/policy.onnx"))
    a.world = next(Path(k) for k in pins if k.endswith("/attempt-0004/compiled_model.mjb"))
    a.csv = next(Path(k) for k in pins if k.endswith("/BGI_Shooting_00001.csv"))
    sonic_root = next(Path(k).parent for k in pins if k.endswith("/planner_sonic.onnx"))
    params = VelocityPolicyParameters.load(config, expected_hash=pins[str(config)])
    final_kp, final_kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
    probe = json.loads((a.probe / "complete.json").read_text())
    schedule = G1GainSchedule(
        params.stiffness,
        params.damping,
        tuple(float(x) for x in final_kp),
        tuple(float(x) for x in final_kd),
        probe["first_ready_global_frame"],
        "constant" if a.mode == "qualify" else "linear10",
    )
    course = tuple(json.loads((a.entry / "declaration.json").read_text())["course"])
    baseline = load_arrays(a.entry / "measured_entry/native300.npz")
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
            gain_schedule=schedule.report(),
            gain_schedule_hash=schedule.contract_hash,
            maximum_native_steps=3000,
            maximum_neural_calls=600,
            new_optimizer_steps=0,
            unchanged_initial_course=course,
            old_research_gate_unchanged=True,
            physical_state_or_solver_history_reset=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    try:

        def factory(w, **kwargs):
            return G1GainScheduledEpisode(w, schedule=schedule, **kwargs)

        def reviewer(raw, neural, params, tracker, world, *, handoff):
            return review_gain_switch(
                raw, neural, params, tracker, world, handoff=handoff, schedule=schedule
            )

        result = run(
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
            body_factory=factory,
            handoff_callback=lambda body, f: body.activate(f),
            reviewer=reviewer,
        )
        if a.mode == "qualify":
            if numerical_result(result) != numerical_result(parents[0]["result"]):
                raise ValueError("new constant protocol changes whole old outcome/metadata")
            compare_constant_artifacts(a.output, a.baseline)
        if producer:
            if numerical_result(result) != numerical_result(producer["result"]):
                raise ValueError("whole independent gain-ramp NN/native repeat differs")
            for name in result["artifacts"]:
                exact_artifact(a.output / name, a.producer / name)
        check_pins(pins)
        status = {
            "qualify": "GAIN_SCHEDULE_CONSTANT_NATIVE_PARITY",
            "produce": "GAIN_SCHEDULE_LINEAR10_NATIVE_PRODUCED",
            "reproduce": "GAIN_SCHEDULE_LINEAR10_NATIVE_REPRODUCED",
        }[a.mode]
        write_once(
            a.output / "complete.json",
            dict(
                status=status,
                source_head=a.expected_head,
                baseline_hash=a.expected_baseline_hash,
                result=result,
                native_steps=3000,
                actual_neural_calls=result["velocity_neural_calls"]
                + result["sonic_encoder_neural_calls"]
                + result["sonic_decoder_neural_calls"],
                constant_all_original_fields_exact=a.mode == "qualify",
                new_optimizer_steps=0,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        if not (a.output / "failure.json").exists():
            write_once(
                a.output / "failure.json",
                dict(
                    error=repr(error),
                    native_neural_counts_in_episode_if_executed=True,
                    automatic_retry_authorized=False,
                ),
            )
        raise


if __name__ == "__main__":
    main()
