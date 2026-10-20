"""Audit the complete entry trial, then four NN calls at its actual first entry.

No dynamics, no native state adoption, no gain switch. A first-target or codec
check is not a qualification of a continuously switched physical motor.
"""

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace


class CaptureSession:
    def __init__(self, session):
        self.session = session
        self.calls = []

    def get_inputs(self):
        return self.session.get_inputs()

    def run(self, outputs, feed):
        result = self.session.run(outputs, feed)
        self.calls.append(({k: v.copy() for k, v in feed.items()}, [v.copy() for v in result]))
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("producer", "repeat", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in (
        "expected-head",
        "producer-unit",
        "repeat-unit",
        "expected-producer-hash",
        "expected-repeat-hash",
    ):
        p.add_argument("--" + k, required=True)
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
        raise ValueError("clean frozen history probe and unused output required")
    pins = {
        str(f): digest(f)
        for dr in (Path.cwd() / "src", Path.cwd() / "scripts")
        for f in dr.rglob("*.py")
    }
    records = []
    for root, unit, h, status in (
        (a.producer, a.producer_unit, a.expected_producer_hash, "G1_VELOCITY_ENTRY_PRODUCED"),
        (a.repeat, a.repeat_unit, a.expected_repeat_hash, "G1_VELOCITY_ENTRY_REPRODUCED"),
    ):
        wait_for_terminal_collection(unit, wait=True)
        if digest(root / "complete.json") != h:
            raise ValueError("exact actual whole native entry parents required")
        c = json.loads((root / "complete.json").read_text())
        d = json.loads((root / "declaration.json").read_text())
        if (
            c["status"] != status
            or c["native_steps"] != 6000
            or c["neural_calls"] != 600
            or c["source_head"] != d["source_head"]
            or len(c["results"]) != 2
            or c["optimizer_steps"] != 0
            or d["arms"] != ["idle", "measured_entry"]
        ):
            raise ValueError("whole two-arm H300 trial required")
        for path, value in d["input_file_hashes"].items():
            if path in pins and pins[path] != value:
                raise ValueError("conflicting numerical input source")
            pins[path] = value
        pins.update({str(f): digest(f) for f in root.rglob("*") if f.is_file()})
        records.append((root, c, d))
    index = source_suffix_hash_index(pins)
    for rel in (
        "src/rosclaw_soccer/rsi/velocity_entry_evidence.py",
        "src/rosclaw_soccer/rsi/ball_entry_teacher.py",
        "src/rosclaw_soccer/providers/g1/sonic_runup.py",
    ):
        if index.get(rel) != {digest(Path(rel))}:
            raise ValueError("same original tracker, entry teacher and physical review required")
    check_pins(pins)
    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_velocity_observation import G1VelocityObservation
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.providers.g1.sonic_external_history import (
        attach_external_history,
        prepare_external_history,
    )
    from rosclaw_soccer.providers.g1.sonic_reference_preparation import prepare_sonic_reference
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.providers.g1.sonic_wrist_quarantine import quarantine_wrist_reference
    from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.native_entry_calibration import NativeCalibratedEntry
    from rosclaw_soccer.rsi.velocity_entry_evidence import review_entry
    from scripts.rsi_collect_velocity_policy_response import compare_velocity_result
    from scripts.rsi_probe_compact_source import exact_artifact, load_arrays

    config = next(Path(k) for k in pins if k.endswith("/velocity/v0/params/deploy.yaml"))
    worldfile = next(Path(k) for k in pins if k.endswith("/attempt-0004/compiled_model.mjb"))
    csv = next(Path(k) for k in pins if k.endswith("/BGI_Shooting_00001.csv"))
    sonic_root = next(Path(k).parent for k in pins if k.endswith("/planner_sonic.onnx"))
    priorpath = next(Path(k) for k in pins if k.endswith("/entry-affordance.json"))
    calibpath = next(
        Path(k)
        for k in pins
        if k.endswith("/rsi-native-entry-calibration-pilot-20261020/declaration.json")
    )
    prior, calib = json.loads(priorpath.read_text()), json.loads(calibpath.read_text())
    parent = SkillEntryAffordance(
        tuple(prior["ball_offset_xy_m"]),
        prior["historical_frame"],
        prior["historical_forward_speed_mps"],
        prior["historical_artifact_hash"],
    )
    if (
        parent.contract_hash != prior["contract_hash"]
        or parent.contract_hash != calib["fit"]["parent_affordance_hash"]
    ):
        raise ValueError("exact historical affordance ancestor required")
    entry = NativeCalibratedEntry(
        tuple(calib["entry_ball_offset_xy_m"]),
        parent.historical_frame,
        parent.historical_forward_speed_mps,
        parent.historical_artifact_hash,
        parent.contract_hash,
        calib["fit"]["motion_configuration_hash"],
        calib["fit_hash"],
    )
    params = VelocityPolicyParameters.load(config, expected_hash=pins[str(config)])
    world = mujoco.MjModel.from_binary_path(str(worldfile))
    sensor = G1VelocityObservation(world, expected_world_hash=pins[str(worldfile)])
    raw_entry = None
    for root, c, d in records:
        for i, arm in enumerate(("idle", "measured_entry")):
            row = c["results"][i]
            if (
                row != json.loads((root / arm / "complete.json").read_text())
                or digest(root / arm / "native300.npz") != row["native_hash"]
            ):
                raise ValueError("whole raw episode commitment required")
            raw = load_arrays(root / arm / "native300.npz")
            summary = review_entry(raw, params, sensor, entry, tuple(d["course"]), arm)
            if any(row.get(k) != v for k, v in summary.items()):
                raise ValueError("entry result differs from recomputed physical evidence")
            if root == a.producer:
                compare_velocity_result(row, records[1][1]["results"][i])
                exact_artifact(root / arm / "native300.npz", a.repeat / arm / "native300.npz")
                if arm == "measured_entry":
                    raw_entry = raw
    if raw_entry is None:
        raise ValueError("actual measured entry recording required")
    ready = np.flatnonzero(raw_entry["entry_ready"])
    if len(ready) == 0 or ready[0] < 10:
        raise ValueError("actual ready boundary and ten real previous frames required")
    f = int(ready[0])
    source = np.loadtxt(csv, delimiter=",", skiprows=1, dtype=np.float64)[466:767]
    q, v = raw_entry["canonical_qpos"][f].copy(), raw_entry["canonical_qvel"][f].copy()
    qh = raw_entry["target_body_canonical_qpos_after_substep"][f - 10 : f, -1].copy()
    vh = raw_entry["target_body_canonical_qvel_after_substep"][f - 10 : f, -1].copy()
    th = raw_entry["joint_target_rad"][f - 10 : f].copy()
    if not np.array_equal(qh[-1], q) or not np.array_equal(vh[-1], v):
        raise ValueError("past-only ten actual post-states must end at current boundary")
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            first_ready_global_frame=f,
            actual_history_post_frame_indices=list(range(f - 9, f + 1)),
            previous_target_global_frame_indices=list(range(f - 10, f)),
            source_window=[466, 767],
            source_frequency_hz=120,
            entry_intervals=50,
            source_wrist_quarantine=True,
            maximum_neural_calls=4,
            new_native_steps=0,
            new_optimizer_steps=0,
            history_probe_not_physical_motor_qualification=True,
            hardware_authorized=False,
        ),
    )
    outputs, receipts = [], []
    try:
        prepared_source, quarantine = quarantine_wrist_reference(source, q[:36])
        prepared = prepare_sonic_reference(
            prepared_source,
            q[:36],
            source_frequency_hz=120,
            entry_intervals=50,
            source_evidence_hash=pins[str(csv)],
        )
        limits = world.jnt_range[[world.joint(n).id for n in G1_DDS_JOINT_NAMES]]
        for trial in range(2):
            c = G1SonicRunupController(sonic_root, inference_threads=1)
            h = prepare_external_history(
                qh, vh, th, defaults=c.default_angles, scale=c._action_scale
            )
            receipt = attach_external_history(
                c,
                h,
                prepared.reference,
                current_qpos=q,
                current_qvel=v,
                padded_reference_frames=309,
            )
            c._encoder, c._decoder = CaptureSession(c._encoder), CaptureSession(c._decoder)
            target = c.update(SimpleNamespace(qpos=q.copy(), qvel=v.copy()), 0)
            if len(c._encoder.calls) != 1 or len(c._decoder.calls) != 1:
                raise ValueError(
                    "exact one encoder and decoder inference per fresh session required"
                )
            enc_feed, enc_out = c._encoder.calls[0]
            dec_feed, dec_out = c._decoder.calls[0]
            arrays = dict(
                encoder_input=next(iter(enc_feed.values())),
                encoder_output=enc_out[0],
                decoder_input=next(iter(dec_feed.values())),
                decoder_output=dec_out[0],
                joint_target=target,
                past_post_qpos=qh,
                past_post_qvel=vh,
                past_applied_targets=th,
                reference=c.reference.copy(),
            )
            np.savez_compressed(a.output / f"session{trial}.npz", **arrays)
            outputs.append(arrays)
            receipts.append(receipt)
        if receipts[0] != receipts[1] or any(
            not np.array_equal(v, outputs[1][k]) for k, v in outputs[0].items()
        ):
            raise ValueError("full independent CPU sessions must repeat exactly")
        check_pins(pins)
        target = outputs[0]["joint_target"]
        outside = [
            G1_DDS_JOINT_NAMES[i]
            for i in np.flatnonzero((target < limits[:, 0]) | (target > limits[:, 1]))
        ]
        write_once(
            a.output / "complete.json",
            dict(
                status="EXTERNAL_SONIC_HISTORY_NN_PROBED_NOT_MOTOR_QUALIFIED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                new_neural_calls=4,
                new_native_steps=0,
                new_optimizer_steps=0,
                reviewed_parent_native_steps=12000,
                reviewed_parent_neural_calls=1200,
                first_ready_global_frame=f,
                all_four_parent_episodes_reviewed=True,
                two_independent_cpu_sessions_exact=True,
                history_receipt=receipts[0],
                history_contract_hash=h.contract_hash,
                source_preparation_hash=prepared.contract_hash,
                quarantine_contract_hash=quarantine.contract_hash,
                first_target_within_native_joint_limits=not outside,
                first_target_outside_joint_names=outside,
                maximum_first_target_change_rad=float(np.max(abs(target - th[-1]))),
                native_motor_or_pd_qualification=False,
                football_skill_proven=False,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                new_native_steps=0,
                nn_outputs_completed=len(outputs) * 2,
                automatic_retry_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
