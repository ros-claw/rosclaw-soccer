"""Finite six-state public paired teacher pilot; old passing branches stay exact."""

import argparse
import json
from pathlib import Path

INDICES = (0, 1, 6, 8, 21, 30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("interface-audit", "source-audit", "source-root", "backbone-model", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("interface-audit-hash", "source-audit-hash", "expected-head"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--case-index", type=int, choices=INDICES, required=True)
    a = parser.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_teacher_physics_context import load_teacher_context

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused declared pilot output required")
    for path, commitment in (
        (a.interface_audit, a.interface_audit_hash),
        (a.source_audit, a.source_audit_hash),
    ):
        if path.is_symlink() or digest(path) != commitment:
            raise ValueError("exact independently closed interface and source teacher required")
    audit = json.loads(a.interface_audit.read_text())
    old_audit = json.loads(a.source_audit.read_text())
    if (
        audit["status"] != "ACTUAL_MEASURED_CLEARANCE_NN547_GEOMETRY_CODEC_VERIFIED"
        or audit["actual_nn547_readbacks"] != 128
        or audit["new_physical_steps"] != 0
    ):
        raise ValueError("actual-state neural/codec/geometry qualification required")
    if old_audit["status"] != "MOTION_SYNERGY_TEACHER_RAW_CEM_FUTURES_AND_FULL_EXAM_VERIFIED":
        raise ValueError("independent old teacher raw audit required")
    pins = dict(audit["input_file_hashes"])
    pins[str(a.interface_audit)] = a.interface_audit_hash
    pins[str(a.source_audit)] = a.source_audit_hash
    paths = [
        Path(p)
        for p, h in pins.items()
        if h == audit["qualification_hash"] and p.endswith("/complete.json")
    ]
    if len(paths) != 1 or json.loads(paths[0].read_text()).get("anticipatory_velocity") is not True:
        raise ValueError("separate explicitly anticipatory qualification required")
    case = a.source_root / f"course-{a.case_index:02d}"
    baseline_path = case / "C.json"
    baseline = json.loads(baseline_path.read_text())
    for path in (baseline_path, case / baseline["artifact"]["path"], a.backbone_model):
        if path.is_symlink() or old_audit["input_file_hashes"].get(str(path)) != digest(path):
            raise ValueError("source case/basis differs from independently audited teacher")
        pins[str(path)] = digest(path)
    model_record = json.loads(a.backbone_model.read_text())
    artifact = a.backbone_model.parent / model_record["artifact"]["path"]
    if old_audit["input_file_hashes"].get(str(artifact)) != digest(artifact):
        raise ValueError("original fixed motion basis required")
    pins[str(artifact)] = digest(artifact)
    source_pins(pins)
    check_pins(pins)
    context = load_teacher_context(pins, a.backbone_model)
    context["assets"]["experimental_fast"] = baseline["experimental_fast_approach"]
    import numpy as np

    from rosclaw_soccer.rsi.clearance_kick_player import AnticipatoryClearanceKickPlayer
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode

    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.anticipatory_clearance_finite_teacher_pilot.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            case_index=a.case_index,
            complete_bank_indices=INDICES,
            historical_protection_indices=[0, 30],
            old_failed_indices=[1, 6, 8, 21],
            actual_world_ceiling=1,
            actual_native_step_ceiling=3600,
            maximum_base_candidates=37,
            maximum_new_candidates=1,
            forecast_native_step_ceiling=48640,
            future_nn_call_ceiling=4864,
            original_constructor_warmups_source_bound=5,
            old_passing_branch_must_remain_bitexact=True,
            new_nn_optimizer_updates=0,
            no_automatic_search_expansion=True,
            task_exam_modified=False,
            hardware_authorized=False,
        ),
    )
    instances = []

    class PilotPlayer(AnticipatoryClearanceKickPlayer):
        basis_hg = context["basis"]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.saved = False
            instances.append(self)

        def plan_entry(self):
            super().plan_entry()
            if self.saved:
                return
            self.saved = True
            # These are produced/persisted before the first actual kick NN call.
            for j, (name, predicted) in enumerate(self.candidate_predictions):
                path = a.output / f"forecast-{j:02d}.npz"
                np.savez_compressed(
                    path, **{k: v for k, v in predicted.items() if isinstance(v, np.ndarray)}
                )
                pins[str(path)] = digest(path)
                path = path.with_suffix(".json")
                write_once(
                    path,
                    dict(
                        name=name,
                        **{k: v for k, v in predicted.items() if not isinstance(v, np.ndarray)},
                    ),
                )
                pins[str(path)] = digest(path)
            path = a.output / "before-actual-decision.json"
            write_once(
                path,
                dict(
                    events=self.events,
                    clearance_events=self.clearance_events,
                    clearance_enabled=self.clearance_enabled,
                ),
            )
            pins[str(path)] = digest(path)

    row, actual = evaluate_episode(
        model=context["model"],
        assets=context["assets"],
        policy_types=context["types"],
        course=tuple(baseline["course"]),
        parameters=baseline["parameters"],
        hard_limits=context["hard_limits"],
        player_type=PilotPlayer,
    )
    path = a.output / "actual-unvalidated.npz"
    np.savez_compressed(path, **actual)
    pins[str(path)] = digest(path)
    path = a.output / "actual-unvalidated.json"
    write_once(path, row)
    pins[str(path)] = digest(path)
    if len(instances) != 1:
        raise ValueError("one actual private teacher required")
    player = instances[0]
    path = a.output / "actual-clearance-edits.json"
    write_once(path, dict(edits=player.clearance_edits))
    pins[str(path)] = digest(path)
    if row["entry_control"] != baseline["entry_control"]:
        raise ValueError("same measured approach/entry required")
    with np.load(case / baseline["artifact"]["path"], allow_pickle=False) as old:
        unchanged = set(old.files) == set(actual) and all(
            np.array_equal(v, old[k]) for k, v in actual.items()
        )
    if not player.clearance_enabled and not unchanged:
        raise ValueError("abstaining or protected teacher changed the old whole actual path")
    if baseline["outcome"]["task_pass"] and (player.clearance_enabled or not unchanged):
        raise ValueError("old passing branch was replaced")
    if player.clearance_enabled:
        predicted = player.selected_prediction
        if predicted is None:
            raise ValueError("selected actual feedback lacks prospective raw evidence")
        entry = row["entry_control"]
        for key in ("qpos", "qvel", "torque", "ball_forces"):
            if not np.array_equal(predicted[key], actual[key][entry * 10 : (entry + 128) * 10]):
                raise ValueError("selected feedback prediction differs fromactual: " + key)
        if not np.array_equal(
            predicted["joint_targets"], actual["joint_targets"][entry : entry + 128]
        ):
            raise ValueError("selected actual target/history path differs fromprediction")
    base, new = player.forecaster.counters(), player.clearance_forecaster.counters()
    requested = base["attempted_native_steps"] + new["attempted_native_steps"]
    integrated = base["integrated_native_steps"] + new["integrated_native_steps"]
    if requested > 48640 or base["future_nn_calls"] + new["future_nn_calls"] > 4864:
        raise ValueError("declared pilot forecast/NN budget exceeded")
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="FINITE_CLEARANCE_TEACHER_ACTUAL_CASE_COMPLETE",
            source_head=a.expected_head,
            input_file_hashes=pins,
            case_index=a.case_index,
            actual_worlds=1,
            actual_native_steps=row["native_steps"],
            old_outcome=baseline["outcome"],
            actual_record=row,
            clearance_enabled=player.clearance_enabled,
            clearance_events=player.clearance_events,
            old_branch_or_abstention_actual_bitexact=unchanged,
            rescue=not baseline["outcome"]["task_pass"] and row["outcome"]["task_pass"],
            regression=baseline["outcome"]["task_pass"] and not row["outcome"]["task_pass"],
            forecasts=dict(
                base=base,
                clearance=new,
                attempted_native_steps=requested,
                integrated_native_steps=integrated,
            ),
            optimizer_updates=0,
            pilot_is_not_generalization_or_learning_proof=True,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
