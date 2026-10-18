"""Reconstruct causal pre-parent inputs on all quarantined historical traces.

No fitting, admission, new Foundation inference, physics or deployable teacher.
An explicit optional flag tests an untrained Core GRU numerically, not in physics.
The observations did not drive those old episodes: this is an offline input
audit, not proof of a closed-loop student or a sensor-qualified real actor.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi import pre_parent_reference_features as inputs
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-readback", type=Path, required=True)
    parser.add_argument("--study-complete", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zero-gru-seed", type=int)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    if (
        args.output.exists()
        or len(args.study_complete) != 4
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("unused output, four complete studies and clean frozen source required")
    pins = {}
    if args.zero_gru_seed is not None and not 0 <= args.zero_gru_seed < 2**32:
        raise ValueError("bounded explicit zero-GRU initialization seed required")
    zero_parameters, limits = None, None
    if args.zero_gru_seed is not None:
        import mujoco
        import rosclaw.growth.causal_residual_memory as memory

        from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
        from rosclaw_soccer.rsi import reference_student_decoder as decoder_module
        from rosclaw_soccer.rsi.contact_reference_labels import reference_labels

        zero_parameters = memory.initial_parameters(135, 12, seed=args.zero_gru_seed)

    def digest(path):
        with Path(path).open("rb") as stream:
            return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()

    def sealed(path):
        value = json.loads(path.read_text())
        if value["report_hash"] != hash_json(
            {k: v for k, v in value.items() if k != "report_hash"}
        ):
            raise ValueError("exact original logical document seal required")
        pins[str(path)] = digest(path)
        return value

    labels = sealed(args.reference_readback)
    if (
        labels.get("schema") != "soccer.rsi.contact_reference_label_reconstruction.v1"
        or labels.get("historical_endpoints") != 18
        or len(labels.get("rows", [])) != 18
        or labels.get("recorded_frames_reconstructed") != 5400
        or labels.get("same_state_and_same_tick") is not True
        or any(
            labels.get(k) is not False
            for k in (
                "counterfactual_trajectories_subtracted",
                "training_dataset_exported",
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError(
            "whole quarantined same-state label audit, not admitted teachers, required"
        )
    for path, h in labels["input_file_hashes"].items():
        if digest(path) != h:
            raise ValueError("original reference audit input bytes changed")
        pins[path] = h
    for relative in (
        "scripts/rsi_audit_reference_actor_inputs.py",
        "src/rosclaw_soccer/rsi/pre_parent_reference_features.py",
        "src/rosclaw_soccer/rsi/contact_reference_labels.py",
        "src/rosclaw_soccer/rsi/step_motor_features.py",
        "src/rosclaw_soccer/rsi/stochastic_step_execution.py",
    ):
        pins[str(source / relative)] = digest(source / relative)
    if zero_parameters is not None:
        for module in (memory, decoder_module):
            pins[module.__file__] = digest(module.__file__)
    schemas = {
        "soccer.rsi.mesh_public_draw_complete.v1": 4,
        "soccer.rsi.mesh_contact_catalog_complete.v1": 16,
        "soccer.rsi.mesh_option4_teacher_screen_complete.v1": 18,
        "soccer.rsi.mesh_surface_swing_pilot_complete.v1": 7,
    }
    seen, receipts = set(), {}
    for path in args.study_complete:
        if pins.get(str(path)) != digest(path):
            raise ValueError("whole original study must belong to authenticated reference audit")
        study = sealed(path)
        schema = study["schema"]
        if schema not in schemas or schema in seen or len(study["results"]) != schemas[schema]:
            raise ValueError("four distinct whole original studies required")
        seen.add(schema)
        receipts.update({r["receipt_hash"]: r for r in study["results"]})
    rows = []
    for item in labels["rows"]:
        receipt = receipts[item["source_receipt_hash"]]
        witness = receipt["witnesses"]["trace"]
        if (
            pins.get(witness["path"]) != witness["sha256"]
            or digest(witness["path"]) != witness["sha256"]
        ):
            raise ValueError("exact original reference-audited physical trace required")
        with np.load(witness["path"], allow_pickle=False) as archive:
            trace = dict(archive)
        arrays = inputs.trace_observations(trace)
        x, y = arrays["observation"], arrays["same_state_reference_target_rad"]
        rows.append(
            dict(
                case_id=item["case_id"],
                source_receipt_hash=item["source_receipt_hash"],
                original_outcome=item["original_outcome"],
                provisional_success_trace=item["provisional_success_trace"],
                reconstructed_input_hash=hash_bytes(x.tobytes()),
                same_state_label_hash=hash_bytes(y.tobytes()),
                input_shape=list(x.shape),
                label_shape=list(y.shape),
                maximum_absolute_input=float(np.max(np.abs(x))),
                nonzero_target_frames=int(np.any(y != 0, axis=1).sum()),
                historical_episode_executed_this_student=False,
                dataset_admitted=False,
            )
        )
        if zero_parameters is not None:
            world = receipt["witnesses"]["world"]
            commitment = receipt["witnesses"]["commitment"]
            if any(
                pins.get(w["path"]) != w["sha256"] or digest(w["path"]) != w["sha256"]
                for w in (world, commitment)
            ):
                raise ValueError("original world and parent identity witness bytes required")
            if limits is None:
                model = mujoco.MjModel.from_binary_path(world["path"])
                if (model.nq, model.nv, model.nu) != (43, 41, 29):
                    raise ValueError("exact canonical compiled G1 joint limits required")
                limits = np.stack([model.jnt_range[model.joint(n).id] for n in G1_DDS_JOINT_NAMES])
                world_hash = world["sha256"]
            elif world["sha256"] != world_hash:
                raise ValueError("zero numeric replay must not mix physical worlds")
            parent = json.loads(Path(commitment["path"]).read_text())["step_model_hash"]
            decoder = decoder_module.ReferenceStudentDecoder(
                zero_parameters,
                mean=np.zeros(135),
                scale=np.ones(135),
                world_hash=world["sha256"],
                parent_model_hash=parent,
            )
            navigation = reference_labels(trace)["navigation_joint_target_rad"]
            observed, hidden = [], []
            for frame in range(300):
                record = decoder.advance(
                    trace, frame=frame, navigation_target=navigation[frame, 0], joint_limits=limits
                )
                if (
                    record["pre_parent_joint_target_rad"].tobytes()
                    != navigation[frame, 0].tobytes()
                    or np.any(record["applied_reference_increment_rad"] != 0)
                    or np.any(record["requested_reference_increment_rad"] != 0)
                ):
                    raise ValueError("zero GRU must preserve every actual navigation target bit")
                observed.append(record["input_features"])
                hidden.append(record["recurrent_hidden_state"])
            rows[-1]["zero_gru_numeric_replay"] = dict(
                decoder_contract=decoder.contract(),
                observed_features_hash=hash_bytes(np.stack(observed).tobytes()),
                hidden_sequence_hash=hash_bytes(np.stack(hidden).tobytes()),
                gru_numeric_inferences=300,
                all_navigation_targets_bit_equal=True,
                actual_parent_inferences_repeated=0,
                actual_body_steps=0,
                full_original_teacher_noninterference_claimed=False,
                closed_loop_student_qualified=False,
            )
    if len({r["source_receipt_hash"] for r in rows}) != 18:
        raise ValueError("eighteen distinct endpoints required; replays cannot inflate data")
    if any(digest(path) != h for path, h in pins.items()):
        raise ValueError("source or original witness bytes changed during input audit")
    result = dict(
        schema="soccer.rsi.pre_parent_reference_input_reconstruction.v1",
        source_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        original_reference_readback_hash=labels["report_hash"],
        input_file_hashes=pins,
        feature_names=list(inputs.FEATURE_NAMES),
        rows=rows,
        reconstructed_frames=5400,
        historical_endpoints=18,
        independent_courses=6,
        reused_original_causal_state_encoder=True,
        current_teacher_target_in_inputs=False,
        future_outcome_or_family_id_in_inputs=False,
        force_clock="previous completed frame; frame0 zero",
        previous_action_clock="previous applied reference increment, not parent motor residual",
        navigation_target_clock=(
            "actual float64 committed navigation target BEFORE contact increment"
        ),
        appended_remaining_time_fraction=True,
        action_dimensions=12,
        parent_motor_must_recompute_on_new_reference=True,
        simulator_kinematic_inputs_not_real_sensor_qualification=True,
        training_dataset_exported=False,
        teacher_qualified=False,
        new_optimizer_updates=0,
        new_neural_inferences=0,
        new_physical_steps=0,
        closed_loop_student_improvement_claimed=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    if zero_parameters is not None:
        result.update(
            schema="soccer.rsi.pre_parent_reference_zero_gru_numeric_replay.v1",
            zero_gru_initial_seed=args.zero_gru_seed,
            new_neural_inferences=5400,
            new_gru_numeric_inferences=5400,
            new_foundation_inferences=0,
            compiled_world_models_loaded=1,
            mujoco_data_allocations=0,
            all5400_navigation_targets_bit_equal=True,
            zero_output_preserves_navigation_not_full_teacher=True,
            no_training_only_initial_parameters=True,
        )
        result["report_hash"] = hash_json({k: v for k, v in result.items() if k != "report_hash"})
    write_once(args.output, result)
    print("PRE_PARENT_INPUT_AUDIT_COMPLETE_NOT_TRAINING", result["report_hash"], flush=True)


if __name__ == "__main__":
    main()
