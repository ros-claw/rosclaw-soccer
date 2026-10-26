"""One G1 actual geometry-feedback rollout and prospective full-NN parity."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("composed-complete", "reference-record", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("composed-hash", "reference-hash", "expected-head"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_teacher_physics_context import load_walk_kick_context

    clean(args.expected_head)
    if args.output.exists():
        raise ValueError("unused qualification output required")
    for path, commitment in (
        (args.composed_complete, args.composed_hash),
        (args.reference_record, args.reference_hash),
    ):
        if path.is_symlink() or digest(path) != commitment:
            raise ValueError("exact closed original proof and public failure required")
    proof = json.loads(args.composed_complete.read_text())
    reference = json.loads(args.reference_record.read_text())
    if (
        proof["status"] != "CURRENT_WALK_TO_KICK_NN_AND_NATIVE_FUTURE_EXACT"
        or reference["arm"] != "B"
        or reference["outcome"]["task_pass"] is not False
    ):
        raise ValueError("qualified body/control interface and consumed public failure required")
    pins = dict(proof["input_file_hashes"])
    pins[str(args.composed_complete)] = args.composed_hash
    pins[str(args.reference_record)] = args.reference_hash
    original = args.reference_record.parent / reference["artifact"]["path"]
    if (
        original.parent != args.reference_record.parent
        or digest(original) != reference["artifact"]["hash"]
    ):
        raise ValueError("same local original raw trajectory required")
    pins[str(original)] = reference["artifact"]["hash"]
    source_pins(pins)
    check_pins(pins)
    context = load_walk_kick_context(pins)
    context["assets"]["experimental_fast"] = reference["experimental_fast_approach"]
    import numpy as np

    from rosclaw_soccer.rsi.ball_clearance_control import BallClearanceController, BallClearancePlan
    from rosclaw_soccer.rsi.shared_nn_forecast import SharedNNForecaster
    from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode

    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            protocol="soccer.measured_clearance_actual_nn_parity.v1",
            source_head=args.expected_head,
            input_file_hashes=pins,
            actual_world_ceiling=1,
            actual_native_step_ceiling=3600,
            forecast_native_step_ceiling=1280,
            future_nn_call_ceiling=128,
            activation_clearance_m=0.08,
            target_offset_limit_rad=0.06,
            requested_offset_slew_per_control_rad=0.008,
            active_controls=128,
            fade_controls=8,
            actual_policy_constructor_warmup_calls_source_bound=5,
            prospective_selection_gate_bypassed_only_for_interface_qualification=True,
            task_effectiveness_proven=False,
            optimizer_updates=0,
            hardware_authorized=False,
        ),
    )
    instances = []

    class QualificationPlayer(SharedRoboNaldoPlayer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.controller = BallClearanceController(self, BallClearancePlan())
            self.forecaster = SharedNNForecaster(
                self,
                expected_policy_type=type(self.policy),
                expected_model_hash=compiled_model_hash(self.world.model),
                native_step_budget=1280,
            )
            self.edits = []
            instances.append(self)

        def install_measured_entry(self, samples, *, reference_frame):
            receipt = super().install_measured_entry(samples, reference_frame=reference_frame)
            self.future = self.forecaster.forecast(
                controls=128, before_proposal=True, clearance_plan=BallClearancePlan()
            )
            path = args.output / "future.npz"
            np.savez_compressed(
                path, **{k: v for k, v in self.future.items() if isinstance(v, np.ndarray)}
            )
            pins[str(path)] = digest(path)
            write_once(
                args.output / "future.json",
                {k: v for k, v in self.future.items() if not isinstance(v, np.ndarray)},
            )
            return receipt

        def propose(self):
            _, receipt = super().propose()
            target, edit = self.controller.apply()
            self.edits.append(edit)
            return target, dict(receipt, proposed_target=target.copy(), target_edit=edit)

    row, actual = evaluate_episode(
        model=context["model"],
        assets=context["assets"],
        policy_types=context["types"],
        course=tuple(reference["course"]),
        parameters=reference["parameters"],
        hard_limits=context["hard_limits"],
        player_type=QualificationPlayer,
    )
    path = args.output / "actual-unvalidated.npz"
    np.savez_compressed(path, **actual)
    pins[str(path)] = digest(path)
    write_once(args.output / "actual-unvalidated.json", row)
    if len(instances) != 1:
        raise ValueError("one actual skill required")
    player = instances[0]
    write_once(args.output / "actual-edits.json", dict(edits=player.edits))
    if player.future["controls"] != 128 or player.future["body_failed"]:
        raise ValueError("complete healthy interface prediction required before expansion")
    entry = row["entry_control"]
    if entry != reference["entry_control"] or row["native_steps"] != 3600:
        raise ValueError("same actual admission and full physical horizon required")
    with np.load(original, allow_pickle=False) as baseline:
        for key in ("qpos", "qvel", "torque", "ball_forces"):
            if not np.array_equal(actual[key][: entry * 10], baseline[key][: entry * 10]):
                raise ValueError("pre-skill actual prefix changed")
        changed = int(np.any(actual["joint_targets"] != baseline["joint_targets"], axis=1).sum())
    for key in ("qpos", "qvel", "torque", "ball_forces"):
        if not np.array_equal(player.future[key], actual[key][entry * 10 : (entry + 128) * 10]):
            raise ValueError("measured-feedback owned future differs from actual: " + key)
    if not np.array_equal(
        player.future["joint_targets"], actual["joint_targets"][entry : entry + 128]
    ):
        raise ValueError("actual corrected NN targets differ from prospective targets")
    applied = sum(bool(e.get("applied", False)) for e in player.edits)
    if applied == 0 or changed == 0:
        raise ValueError("nonzero original-codec physical target changes required")
    check_pins(pins)
    clean(args.expected_head)
    write_once(
        args.output / "complete.json",
        dict(
            status="MEASURED_CLEARANCE_G1_ACTUAL_NN_FUTURE_EXACT",
            source_head=args.expected_head,
            input_file_hashes=pins,
            actual_record=row,
            original_outcome=reference["outcome"],
            original_nonfoot_peak_n=reference["native_nonfoot_peak_force_n"],
            forecasts=player.forecaster.counters(),
            actual_native_steps=row["native_steps"],
            applied_clearance_controls=applied,
            changed_target_controls_vs_original=changed,
            task_effectiveness_proven=False,
            optimizer_updates=0,
            qualification_is_not_teacher_selection_or_learning_gain=True,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
