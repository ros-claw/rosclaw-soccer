"""One current-state generated contact reference through original NN/native PD."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("composed-complete", "geometry-complete", "reference-record", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("composed-hash", "geometry-hash", "reference-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_teacher_physics_context import load_walk_kick_context

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused native contact-reference output required")
    for path, commitment in (
        (a.composed_complete, a.composed_hash),
        (a.geometry_complete, a.geometry_hash),
        (a.reference_record, a.reference_hash),
    ):
        if path.is_symlink() or digest(path) != commitment:
            raise ValueError("exact closed prerequisites and consumed original failure required")
    composed = json.loads(a.composed_complete.read_text())
    geometry = json.loads(a.geometry_complete.read_text())
    reference = json.loads(a.reference_record.read_text())
    if (
        composed["status"] != "CURRENT_WALK_TO_KICK_NN_AND_NATIVE_FUTURE_EXACT"
        or geometry["status"] != "CONTACT_REFERENCE_ENDPOINT_GEOMETRY_RECORDED_NOT_PHYSICS"
        or geometry.get("actual_first_contact_foot_used_for_OFFLINE_foot_selection") is not True
    ):
        raise ValueError("body/NN interface and explicit-foot geometry prerequisite required")
    if (
        reference["arm"] != "B"
        or reference["course_index"] != 8
        or reference["outcome"]["task_pass"] is not False
    ):
        raise ValueError("single preregistered consumed failure8, not a generalization claim")
    pins = dict(composed["input_file_hashes"])
    for path, h in (
        (a.composed_complete, a.composed_hash),
        (a.geometry_complete, a.geometry_hash),
        (a.reference_record, a.reference_hash),
    ):
        pins[str(path)] = h
    original = a.reference_record.parent / reference["artifact"]["path"]
    if (
        original.parent != a.reference_record.parent
        or original.is_symlink()
        or digest(original) != reference["artifact"]["hash"]
    ):
        raise ValueError("original same-course raw commitment required")
    pins[str(original)] = reference["artifact"]["hash"]
    source_pins(pins)
    check_pins(pins)
    context = load_walk_kick_context(pins)
    context["assets"]["experimental_fast"] = reference["experimental_fast_approach"]
    import numpy as np

    from rosclaw_soccer.rsi.contact_reference_generator import generate_contact_reference
    from rosclaw_soccer.rsi.contact_reference_plan import condition_contact_reference
    from rosclaw_soccer.rsi.shared_nn_forecast import SharedNNForecaster, copy_sim_policy
    from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode
    from rosclaw_soccer.sim.contracts import hash_json

    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.current_nn_contact_reference_native_qualification.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            actual_world_ceiling=1,
            actual_native_step_ceiling=3600,
            forecast_native_step_ceiling=2560,
            future_nn_call_ceiling=256,
            original_constructor_warmups_source_bound=5,
            reference_joint_offset_limit_rad=0.35,
            additional_reference_velocity_limit_rad_s=4.0,
            reference_duration_controls=48,
            actual_joint_or_torque_caps_modified=False,
            actual_reference_FORCED_ONLY_for_interface_qualification_if_valid=True,
            recorded_future_arrays_used_for_control=False,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    instances = []

    def save_future(name, future):
        path = a.output / (name + ".npz")
        np.savez_compressed(path, **{k: v for k, v in future.items() if isinstance(v, np.ndarray)})
        pins[str(path)] = digest(path)
        path = path.with_suffix(".json")
        write_once(path, {k: v for k, v in future.items() if not isinstance(v, np.ndarray)})
        pins[str(path)] = digest(path)

    class QualificationPlayer(SharedRoboNaldoPlayer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.forecaster = SharedNNForecaster(
                self,
                expected_policy_type=type(self.policy),
                expected_model_hash=compiled_model_hash(self.world.model),
                native_step_budget=2560,
            )
            self.future = None
            self.installation = None
            instances.append(self)

        def install_measured_entry(self, samples, *, reference_frame):
            receipt = super().install_measured_entry(samples, reference_frame=reference_frame)
            plan, self.generation, self.parent = generate_contact_reference(self, self.forecaster)
            save_future("parent-future", self.parent)
            if plan is not None:
                limits = self.world.model.jnt_range[self.layout.joints].astype(np.float64)
                preview = condition_contact_reference(
                    copy_sim_policy(self.policy), plan, limits_hg=limits
                )
                self.installation = preview
                if preview["applied"]:
                    self.future = self.forecaster.forecast(
                        controls=128, before_proposal=True, reference_plan=plan
                    )
                    save_future("contact-future", self.future)
                    actual_install = condition_contact_reference(
                        self.policy, plan, limits_hg=limits
                    )
                    if hash_json(actual_install) != hash_json(preview):
                        raise ValueError("prospective and actual reference installation differ")
            path = a.output / "before-actual-reference.json"
            write_once(
                path,
                dict(
                    generation=self.generation,
                    installation=self.installation,
                    recorded_future_used=False,
                    actual_reference_installed=self.future is not None,
                ),
            )
            pins[str(path)] = digest(path)
            return receipt

    row, actual = evaluate_episode(
        model=context["model"],
        assets=context["assets"],
        policy_types=context["types"],
        course=tuple(reference["course"]),
        parameters=reference["parameters"],
        hard_limits=context["hard_limits"],
        player_type=QualificationPlayer,
    )
    path = a.output / "actual-unvalidated.npz"
    np.savez_compressed(path, **actual)
    pins[str(path)] = digest(path)
    path = a.output / "actual-unvalidated.json"
    write_once(path, row)
    pins[str(path)] = digest(path)
    if len(instances) != 1 or row["entry_control"] != reference["entry_control"]:
        raise ValueError("one same measured actual entry required")
    player = instances[0]
    entry = row["entry_control"]
    with np.load(original, allow_pickle=False) as old:
        for key in ("qpos", "qvel", "torque", "ball_forces"):
            if not np.array_equal(actual[key][: entry * 10], old[key][: entry * 10]):
                raise ValueError("original pre-entry actual path changed")
            if player.parent["controls"] == 128 and not np.array_equal(
                player.parent[key], old[key][entry * 10 : (entry + 128) * 10]
            ):
                raise ValueError(
                    "own original NN parent future differs from original actual source"
                )
        changed = int(
            np.any(
                actual["joint_targets"] != old["joint_targets"][: len(actual["joint_targets"])],
                axis=1,
            ).sum()
        )
        unchanged = (
            all(np.array_equal(actual[k], old[k]) for k in actual)
            if len(actual["qpos"]) == len(old["qpos"])
            else False
        )
    status = "REFERENCE_ABSTAINED_NO_NEW_ACTION"
    compared = 0
    if player.future is not None:
        compared = min(player.future["controls"], row["controls"] - entry)
        for key in ("qpos", "qvel", "torque", "ball_forces"):
            if not np.array_equal(
                player.future[key][: compared * 10],
                actual[key][entry * 10 : (entry + compared) * 10],
            ):
                raise ValueError(
                    "new reference original-NN/native future differs from actual: " + key
                )
        if not np.array_equal(
            player.future["joint_targets"][:compared],
            actual["joint_targets"][entry : entry + compared],
        ):
            raise ValueError("conditioned NN target/history differs fromactual")
        if changed == 0:
            raise ValueError("nonzero reference installed without any measured target effect")
        status = (
            "CONTACT_REFERENCE_NATIVE_NN_EXACT"
            if compared == 128 and row["healthy_full_horizon"]
            else "CONTACT_REFERENCE_PARTIAL_PHYSICAL_FAILURE_NOT_QUALIFIED"
        )
    elif not unchanged:
        raise ValueError("abstention changed the original whole actual path")
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status=status,
            source_head=a.expected_head,
            input_file_hashes=pins,
            actual_worlds=1,
            actual_native_steps=row["native_steps"],
            actual_record=row,
            original_outcome=reference["outcome"],
            generation=player.generation,
            installation=player.installation,
            compared_native_controls=compared,
            forecasts=player.forecaster.counters(),
            changed_target_controls=changed,
            no_action_path_bitexact_original=unchanged,
            neural_optimizer_updates=0,
            qualification_is_not_generalized_teacher_or_learning_gain=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
