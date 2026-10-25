"""Qualify copied-policy 64-control predictions on actual G1; no task gain claim."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("effect-review", "reference-record", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--effect-review-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--pre-proposal-96", action="store_true")
    p.add_argument("--pre-proposal-128", action="store_true")
    p.add_argument("--synergy-model", type=Path)
    p.add_argument("--synergy-model-hash")
    p.add_argument("--whole-joint-reference", action="store_true")
    p.add_argument("--aim-goal-y", type=float)
    p.add_argument("--executed-synergy", action="store_true")
    p.add_argument("--executed-task-axis", type=int)
    p.add_argument("--target-feasible-axes", action="store_true")
    a = p.parse_args()
    if a.pre_proposal_96 and a.pre_proposal_128:
        raise ValueError("one explicit forecast qualification protocol required")
    pre_proposal = a.pre_proposal_96 or a.pre_proposal_128
    aim_command = a.aim_goal_y is not None
    task_axis = a.executed_task_axis is not None
    target_conditioning = a.executed_synergy or task_axis
    if task_axis and (
        not 0 <= a.executed_task_axis < 29
        or not a.pre_proposal_128
        or not a.target_feasible_axes
        or a.executed_synergy
        or a.synergy_model is not None
        or a.synergy_model_hash is not None
        or a.whole_joint_reference
        or aim_command
    ):
        raise ValueError("separate128 per-axis actual-task direction, without PCA/reference mixing")
    if a.target_feasible_axes and not target_conditioning:
        raise ValueError("per-axis target feasibility is an explicit executed-target study")
    if a.executed_synergy and (
        not a.pre_proposal_128 or a.synergy_model is None or a.whole_joint_reference or aim_command
    ):
        raise ValueError("separate128 executed-target qualification and fitted subspace required")
    if aim_command and (
        not a.pre_proposal_128
        or a.synergy_model is not None
        or a.whole_joint_reference
        or not -2.0 <= a.aim_goal_y <= 2.0
        or a.aim_goal_y == 0.0
    ):
        raise ValueError("separate bounded nonzero aim-command128 qualification required")
    if a.whole_joint_reference and (not a.pre_proposal_128 or a.synergy_model is not None):
        raise ValueError(
            "whole-joint reference requires separate128 protocol without fitted PCA mixing"
        )
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or digest(a.effect_review) != a.effect_review_hash:
        raise ValueError("unused output and exact closed local-pilot effect readback required")
    review = json.loads(a.effect_review.read_text())
    if review["status"] != "TASK_SPACE_NATIVE_EFFECT_READBACK_COMPLETE":
        raise ValueError("closed physical effect analysis required")
    pins = dict(review["input_file_hashes"])
    pins[str(a.effect_review)] = a.effect_review_hash
    synergy_model = None
    feedback_complement = False
    if a.synergy_model is not None:
        if not pre_proposal or digest(a.synergy_model) != a.synergy_model_hash:
            raise ValueError("exact synergy model and explicit pre-proposal protocol required")
        synergy_model = json.loads(a.synergy_model.read_text())
        feedback_complement = (
            synergy_model["status"] == "ACTUAL_SUCCESSFUL_MOTOR_FEEDBACK_COMPLEMENT_FITTED"
        )
        if (
            synergy_model["status"] != "PUBLIC_EXECUTED_MOTION_SYNERGIES_FITTED"
            and not feedback_complement
        ):
            raise ValueError("executed public motion subspace required")
        if feedback_complement and not a.pre_proposal_128:
            raise ValueError("new feedback complement requires explicit128 qualification")
        if feedback_complement and a.executed_synergy:
            raise ValueError("executed-target study uses only the original qualified subspace")
        for key, value in synergy_model["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting motion model source/asset binding")
            pins[key] = value
        pins[str(a.synergy_model)] = a.synergy_model_hash
        model_artifact = a.synergy_model.parent / synergy_model["artifact"]["path"]
        expected_artifact = synergy_model["artifact"]["hash"]
        if (
            model_artifact.is_symlink()
            or not model_artifact.resolve().is_relative_to(a.synergy_model.parent.resolve())
            or digest(model_artifact) != expected_artifact
            or (str(model_artifact) in pins and pins[str(model_artifact)] != expected_artifact)
        ):
            raise ValueError("confined exact fitted model artifact required before inference")
        pins[str(model_artifact)] = expected_artifact
    if pins.get(str(a.reference_record)) != digest(a.reference_record):
        raise ValueError("explicit pinned original trajectory required")
    reference = json.loads(a.reference_record.read_text())
    if reference["arm"] != "onnx":
        raise ValueError("original unmodified ONNX controller required")
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        paths = [Path(k) for k in pins if k.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError("unique qualified research asset required: " + suffix)
        return paths[0]

    import mujoco
    import numpy as np
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.aim_goal_command import AimGoalCommand, condition_aim_goal
    from rosclaw_soccer.rsi.executed_target_synergy import (
        ExecutedTargetSynergy,
        apply_executed_target_synergy,
    )
    from rosclaw_soccer.rsi.experimental_velocity_evidence import admit_heading_evidence
    from rosclaw_soccer.rsi.motion_synergy_reference import (
        MotionSynergyPlan,
        condition_synergy_reference,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_nn_forecast import SharedNNForecaster
    from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode
    from rosclaw_soccer.rsi.whole_joint_reference import (
        WholeJointReferencePlan,
        condition_whole_joint_reference,
    )

    cfgs = [
        Path(k)
        for k in pins
        if k.endswith(".json")
        and Path(k).stat().st_size < 2048
        and json.loads(Path(k).read_text()).get("schema")
        == "soccer.native_single_finisher_experimental_approach.v1"
    ]
    if len(cfgs) != 1:
        raise ValueError("unique qualified experimental approach required")
    fast, _ = admit_heading_evidence(cfgs[0], pins[str(cfgs[0])])
    vm, vc = one("/velocity/v0/exported/policy.onnx"), one("/velocity/v0/params/deploy.yaml")
    assets = dict(
        velocity_model=vm,
        velocity_config=vc,
        velocity_model_hash=pins[str(vm)],
        velocity_config_hash=pins[str(vc)],
        **fast,
    )
    assets["experimental_fast"] = reference["experimental_fast_approach"]
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(
            one("/cached-standard/declared-scene.xml").read_text(), prefixes=("a/",)
        )
    )
    model_hash = compiled_model_hash(model)
    types = _load_robonaldo(one("/policy/robonaldo/FreeKick.py").parents[2])
    instances = []
    scheduled_frames = (80, 100, 150, 220)
    frames = (
        (reference["entry_control"], *scheduled_frames)
        if aim_command or target_conditioning
        else scheduled_frames
    )
    if len(set(frames)) != len(frames):
        raise ValueError("distinct declared measured-entry and scheduled probes required")
    horizon = 128 if a.pre_proposal_128 else 96 if a.pre_proposal_96 else 64
    forecast_budget = len(frames) * horizon * 10
    basis_isaac = None
    if synergy_model is not None:
        with np.load(
            a.synergy_model.parent / synergy_model["artifact"]["path"], allow_pickle=False
        ) as raw:
            basis_hg = raw["basis_hg"]
        if basis_hg.shape != (6, 29):
            raise ValueError("rank6 motion subspace required")
        basis_isaac = np.zeros_like(basis_hg)
        basis_isaac[:, types[3]] = basis_hg

    class Probe(SharedRoboNaldoPlayer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.predictor = SharedNNForecaster(
                self,
                expected_policy_type=types[2],
                expected_model_hash=model_hash,
                native_step_budget=forecast_budget,
            )
            self.predictions = []
            self.target_plan = None
            self.target_edits = []
            self.original_reference = None
            instances.append(self)

        def install_measured_entry(self, samples, *, reference_frame):
            receipt = super().install_measured_entry(samples, reference_frame=reference_frame)
            if aim_command:
                command = AimGoalCommand((8.0, float(a.aim_goal_y), 0.11))
                self.predictions.append(
                    self.predictor.forecast(
                        controls=horizon, before_proposal=True, aim_goal=command
                    )
                )
                condition_aim_goal(self.policy, command)
                self.goal = self.policy.target_pos_w.copy()
            if target_conditioning:
                self.original_reference = {
                    k: getattr(self.policy, k).copy()
                    for k in (
                        "motion_joint_pos",
                        "motion_joint_vel",
                        "motion_body_pos",
                        "motion_body_quat",
                    )
                }
                if task_axis:
                    from rosclaw_soccer.rsi.actuated_task_response import actuated_task_plan

                    delta = np.zeros(29, np.float64)
                    delta[a.executed_task_axis] = 0.03
                    self.target_plan = actuated_task_plan(self.world.control_steps, delta)
                else:
                    self.target_plan = ExecutedTargetSynergy(
                        self.world.control_steps,
                        tuple(tuple(float(x) for x in r) for r in basis_hg),
                        ((0.01, 0.0, 0.0, 0.0, 0.0, 0.0), (-0.01, 0.0, 0.0, 0.0, 0.0, 0.0)),
                        projection_mode="per_axis" if a.target_feasible_axes else "whole_tick",
                    )
                self.predictions.append(
                    self.predictor.forecast(
                        controls=horizon, before_proposal=True, target_plan=self.target_plan
                    )
                )
            elif basis_isaac is not None:
                plan = MotionSynergyPlan(
                    reference_frame,
                    tuple(tuple(float(x) for x in r) for r in basis_isaac),
                    ((0.01, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                )
                condition_synergy_reference(self.policy, plan)
            if a.whole_joint_reference:
                condition_whole_joint_reference(
                    self.policy,
                    WholeJointReferencePlan(
                        reference_frame,
                        tuple(int(i) for i in types[3]),
                        ((0.001,) * 29, (-0.001,) * 29),
                    ),
                )
            return receipt

        def propose(self):
            if pre_proposal and self.world.control_steps in scheduled_frames:
                self.predictions.append(
                    self.predictor.forecast(
                        controls=horizon, before_proposal=True, target_plan=self.target_plan
                    )
                )
            target, receipt = super().propose()
            if self.target_plan is not None:
                target, edit = apply_executed_target_synergy(self, self.target_plan)
                self.target_edits.append(edit)
            if not pre_proposal and self.world.control_steps in frames:
                self.predictions.append(self.predictor.forecast(controls=horizon))
            return target, receipt

    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            model_hash=model_hash,
            probes=frames,
            controls_per_probe=horizon,
            before_proposal=pre_proposal,
            reference_conditioning="OFF_original_arrays"
            if target_conditioning
            else "whole_joint_reference_nonzero"
            if a.whole_joint_reference
            else "feedback_complement_nonzero"
            if feedback_complement
            else "synergy_nonzero"
            if synergy_model is not None
            else "OFF_original_arrays",
            actual_worlds=1,
            actual_native_step_ceiling=3600,
            forecast_native_step_ceiling=forecast_budget,
            every_future_target_and_native_array_must_match=True,
            no_changed_targets=synergy_model is None
            and not a.whole_joint_reference
            and not aim_command
            and not target_conditioning,
            command_goal_conditioning="entry_nonzero_world_aim" if aim_command else "OFF",
            executed_target_conditioning=target_conditioning,
            actuated_task_axis=a.executed_task_axis,
            actuated_task_probe_rad=0.03 if task_axis else None,
            fitted_pca_basis_used=synergy_model is not None,
            executed_target_projection_mode="per_axis" if a.target_feasible_axes else "whole_tick",
            task_scoring_goal_world_point=[8.0, 0.0, 0.11],
            desired_command_world_point=[8.0, a.aim_goal_y, 0.11]
            if aim_command
            else [8.0, 0.0, 0.11],
            copied_history_no_enter_or_reset=True,
            optimizer_updates=0,
            teacher_effectiveness_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    row, arrays = evaluate_episode(
        model=model,
        assets=assets,
        policy_types=types,
        course=tuple(reference["course"]),
        parameters=reference["parameters"],
        hard_limits=np.array(G1_HARD_TORQUE_LIMITS),
        player_type=Probe,
    )
    if task_axis:
        # A failed new qualification must retain its charged actual trajectory.
        actual_unvalidated = a.output / "actual-unvalidated.npz"
        np.savez_compressed(actual_unvalidated, **arrays)
        unvalidated_predictions = []
        for j, predicted in enumerate(instances[0].predictions if instances else []):
            path = a.output / f"forecast-unvalidated-{j:02d}.npz"
            np.savez_compressed(
                path,
                **{
                    k: predicted[k]
                    for k in (
                        "qpos",
                        "qvel",
                        "torque",
                        "ball_forces",
                        "joint_targets",
                        "policy_ticks",
                    )
                },
            )
            unvalidated_predictions.append(
                dict(
                    artifact=path.name,
                    hash=digest(path),
                    donor_control_frame=predicted["donor_control_frame"],
                    controls=predicted["controls"],
                    body_failed=predicted["body_failed"],
                )
            )
        write_once(
            a.output / "executed-unvalidated.json",
            dict(
                actual_record=row,
                artifact=actual_unvalidated.name,
                hash=digest(actual_unvalidated),
                target_edits=instances[0].target_edits if instances else [],
                predictions=unvalidated_predictions,
                actual_native_steps=row["native_steps"],
                optimizer_updates=0,
                physical_interface_qualified=False,
                hardware_authorized=False,
            ),
        )
        if row["controls"] != 360 or not row["healthy_full_horizon"]:
            raise ValueError("task-direction qualification requires a healthy full actual horizon")
    if len(instances) != 1 or row["entry_control"] is None:
        raise ValueError("actual admitted neural skill required")
    probe = instances[0]
    if tuple(r["donor_control_frame"] for r in probe.predictions) != frames:
        raise ValueError("all preregistered NN forecast boundaries required")
    if aim_command and not probe.predictions[0]["aim_command_conditioned"]:
        raise ValueError("entry forecast must apply the nonzero command on an unchanged donor")
    if target_conditioning and (
        not probe.predictions[0]["executed_target_plan"]
        or not any(e["applied"] for e in probe.target_edits)
        or any(e["reference_modified"] for e in probe.target_edits)
        or any(
            not np.array_equal(value, getattr(probe.policy, key))
            for key, value in probe.original_reference.items()
        )
    ):
        raise ValueError("entry-conditioned actually executed multiframe target edits required")
    for predicted in probe.predictions:
        start = predicted["donor_control_frame"]
        if predicted["controls"] != horizon or predicted["body_failed"]:
            raise ValueError("original healthy full forecast future not reproduced")
        for key in ("qpos", "qvel", "torque", "ball_forces"):
            if not np.array_equal(predicted[key], arrays[key][start * 10 : (start + horizon) * 10]):
                raise ValueError("closed-loop NN forecast differs from actual future: " + key)
        if not np.array_equal(
            predicted["joint_targets"], arrays["joint_targets"][start : start + horizon]
        ):
            raise ValueError("future NN calls do not reproduce actual future PD targets")
        artifact = a.output / f"forecast-{start:03d}.npz"
        np.savez_compressed(
            artifact,
            **{
                k: predicted[k]
                for k in ("qpos", "qvel", "torque", "ball_forces", "joint_targets", "policy_ticks")
            },
        )
        pins[str(artifact)] = digest(artifact)
    with np.load(a.reference_record.parent / reference["artifact"]["path"]) as old:
        same = set(old.files) == set(arrays) and all(
            np.array_equal(old[k], v) for k, v in arrays.items()
        )
        unconditioned = (
            synergy_model is None
            and not a.whole_joint_reference
            and not aim_command
            and not target_conditioning
        )
        if unconditioned and not same:
            raise ValueError("owned forecasts changed original actual whole trajectory")
        if not unconditioned and same:
            raise ValueError(
                "nonzero reference qualification did not change actual whole trajectory"
            )
    artifact = a.output / "actual.npz"
    np.savez_compressed(artifact, **arrays)
    pins[str(artifact)] = digest(artifact)
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status=(
                "G1_ACTUATED_TASK_DIRECTION_NN_FUTURES_EXACT"
                if task_axis
                else "G1_EXECUTED_TARGET_SYNERGY_NN_FUTURES_EXACT"
                if a.executed_synergy
                else "G1_AIM_GOAL_COMMAND_NN_FUTURES_EXACT"
                if aim_command
                else "G1_WHOLE_JOINT_REFERENCE_NN_FUTURES_EXACT"
                if a.whole_joint_reference
                else f"G1_COPIED_NN_CLOSED_LOOP_{horizon}_CONTROL_FORECASTS_EXACT"
                if synergy_model is None
                else "G1_FEEDBACK_COMPLEMENT_REFERENCE_NN_FUTURES_EXACT"
                if feedback_complement
                else "G1_SYNERGY_REFERENCE_NN_FUTURES_EXACT"
            ),
            before_proposal=pre_proposal,
            reference_conditioning="OFF_original_arrays"
            if target_conditioning
            else "whole_joint_reference_nonzero"
            if a.whole_joint_reference
            else "feedback_complement_nonzero"
            if feedback_complement
            else "synergy_nonzero"
            if synergy_model is not None
            else "OFF_original_arrays",
            source_head=a.expected_head,
            command_goal_conditioning="entry_nonzero_world_aim" if aim_command else "OFF",
            executed_target_conditioning=target_conditioning,
            actuated_task_axis=a.executed_task_axis,
            actuated_task_probe_rad=0.03 if task_axis else None,
            fitted_pca_basis_used=synergy_model is not None,
            executed_target_projection_mode="per_axis" if a.target_feasible_axes else "whole_tick",
            executed_target_edits=probe.target_edits,
            reference_arrays_unchanged_by_target_plan=bool(target_conditioning),
            future_target_semantics="original_NN_plus_executed_target_plan"
            if target_conditioning
            else "original_NN",
            task_scoring_goal_world_point=[8.0, 0.0, 0.11],
            desired_command_world_point=[8.0, a.aim_goal_y, 0.11]
            if aim_command
            else [8.0, 0.0, 0.11],
            entry_command_forecast_conditioned=bool(
                aim_command and probe.predictions[0].get("aim_command_conditioned")
            ),
            input_file_hashes=pins,
            actual_record=row,
            actual_worlds=1,
            actual_native_steps=row["native_steps"],
            forecasts=probe.predictor.counters(),
            all_future_nn_targets_and_native_arrays_exact=True,
            actual_whole_trajectory_unchanged=synergy_model is None
            and not a.whole_joint_reference
            and not aim_command
            and not target_conditioning,
            teacher_effectiveness_proven=False,
            optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
