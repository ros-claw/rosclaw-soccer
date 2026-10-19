"""Independent whole12 navigation/raw audit and36000 recorded-target steps.

Recorded PD targets, not an independently reimplemented neural policy. No
Foundation allocation, optimizer, teacher promotion or new unique course.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.navigation_recovery_parity import review_navigation_recovery
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_navigation_recovery_inputs import ACTIONS, CONTEXTS, require_scope
from scripts.rsi_read_full_task_zero import same_arrays
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def read_campaign(root, *, complete_file_hash):
    """Authenticate the whole finite campaign before any new physics."""
    pins = {}

    def pin(name, expected=None):
        path = Path(name)
        value = digest(path)
        if (
            path.is_symlink()
            or (expected is not None and value != expected)
            or (str(path) in pins and pins[str(path)] != value)
        ):
            raise ValueError("immutable exact navigation proof required")
        pins[str(path)] = value

    def sealed(name, expected=None):
        pin(name, expected)
        value = json.loads(Path(name).read_text())
        if value["report_hash"] != hash_json(
            {k: v for k, v in value.items() if k != "report_hash"}
        ):
            raise ValueError("navigation proof seal mismatch")
        return value

    def arrays(witness):
        pin(witness["path"], witness["sha256"])
        with np.load(witness["path"], allow_pickle=False) as archive:
            return dict(archive)

    complete = sealed(root / "complete.json", complete_file_hash)
    declaration = sealed(root / "declaration.json")
    require_scope(declaration["config"])
    if (
        complete["schema"] != "soccer.rsi.navigation_recovery4_complete.v1"
        or declaration["schema"] != "soccer.rsi.navigation_recovery4_declaration.v1"
        or complete["declaration_hash"] != declaration["report_hash"]
        or Path(declaration["config"]["output_root"]) != root
        or [j["index"] for j in declaration["jobs"]] != CONTEXTS
        or [r["index"] for r in complete["rows"]] != CONTEXTS
        or declaration["prediction_actions"] != ["baseline", "baseline", *ACTIONS[1:]]
        or any(
            complete.get(k) is not False
            for k in (
                "teacher_qualified",
                "training_authorized",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or type(complete["new_optimizer_updates"]) is not int
        or complete["new_optimizer_updates"] != 0
    ):
        raise ValueError("whole four fixed consumed contexts and no promotion required")
    for name, value in declaration["input_file_hashes"].items():
        pin(name, value)
    stage = sealed(declaration["config"]["stage_readback"])
    native = sealed(stage["effective_cohort_spec"]["producer_complete"])
    cases = []
    prediction_steps = 0
    for group, job in zip(complete["rows"], declaration["jobs"], strict=True):
        i = job["index"]
        if len(group["actuals"]) != 3 or len(group["predictions"]) != 4:
            raise ValueError("all three actual arms and four predictions required")
        if type(job["source_index"]) is not int or not 0 <= job["source_index"] < 6:
            raise ValueError("original source binding required")
        parent = native["results"][job["source_index"]]
        world = parent["witnesses"]["world"]
        cw = parent["witnesses"]["commitment"]
        pin(world["path"], world["sha256"])
        pin(cw["path"], cw["sha256"])
        commitment = json.loads(Path(cw["path"]).read_text())
        if commitment["motor_policy_hash"] != job["original_policy_hash"]:
            raise ValueError("same original sampled motor policy required")
        gold = arrays(job["gold_arrays"])
        for ai, (action, row) in enumerate(zip(ACTIONS, group["actuals"], strict=True)):
            if (
                row != sealed(root / f"case-{i}-arm-{ai}-complete.json")
                or row["index"] != i
                or row["arm_index"] != ai
                or row["action_name"] != action
                or row["course"] != job["course"]
                or row["source_case_hash"] != job["source_case_hash"]
                or type(row["new_optimizer_updates"]) is not int
                or row["new_optimizer_updates"] != 0
                or any(
                    row[k] is not False
                    for k in (
                        "teacher_qualified",
                        "training_authorized",
                        "promotion_authorized",
                        "hardware_authorized",
                    )
                )
            ):
                raise ValueError("all ordered same-course actual rows required")
            trace = arrays(row["arrays"])
            control = review_navigation_recovery(trace, gold, action_name=action)
            outcome = label_first_touch(trace).to_dict()
            start = control["start_frame"]
            if outcome != row["outcome"] or row["matching_prediction_indices"] != (
                [0, 1] if ai == 0 else [ai + 1]
            ):
                raise ValueError("whole M0 and exact predicted arm binding required")
            main = row["main_review"]
            if (
                any(
                    type(main[k]) is not int or main[k] != 3000
                    for k in ("native_attempted_steps", "native_integrated_steps")
                )
                or main.get("constant_forward_approach") is not True
                or main.get("original_precontact_forward_law_unchanged") is not False
            ):
                raise ValueError("whole declared actual body and original approach flag required")
            lease = main.get("born_navigation_recovery")
            if ai == 0:
                if lease is not None:
                    raise ValueError("baseline must not install a lease")
            elif (
                not isinstance(lease, dict)
                or lease.get("start_frame") != start
                or lease.get("action_name") != action
                or lease.get("forward_cap_m_s") != (0.7 if ai == 1 else 0.0)
                or lease.get("end_exclusive") != 300
                or lease.get("contract_hash")
                != hash_json({k: v for k, v in lease.items() if k != "contract_hash"})
            ):
                raise ValueError("same born-at-current lease receipt required")
            for bi in row["matching_prediction_indices"]:
                pred = group["predictions"][bi]
                if (
                    pred != sealed(root / f"case-{i}-branch-{bi}-prediction.json")
                    or pred["index"] != i
                    or pred["branch_index"] != bi
                    or pred["action_name"] != action
                    or pred["start_frame"] != start
                    or pred["predicted_outcome"] != outcome
                    or pred["persisted_before_donor_continues"] is not True
                    or pred["recorded_future_used_for_prediction"] is not False
                ):
                    raise ValueError("separately sealed prospective same-lease prediction required")
                same_arrays(arrays(pred["arrays"]), {k: a[start:] for k, a in trace.items()})
                review = pred["review"]
                expected_calls = sum(f % 20 == 0 for f in range(start, 300))
                if (
                    review["action_name"] != action
                    or review["schema"] != "soccer.rsi.full_task_branch_review.v1"
                    or review["start_frame"] != start
                    or review["new_planner_onnx_calls"] != expected_calls
                    or review["future_observations_from_own_simulation"] is not True
                    or review["recorded_future_targets_or_states_consumed"] is not False
                    or review["prefix_steps_counted_as_new"] is not False
                    or any(
                        review.get(k) is not False
                        for k in (
                            "teacher_qualified",
                            "hardware_authorized",
                            "promotion_authorized",
                        )
                    )
                    or type(review["new_optimizer_updates"]) is not int
                    or review["new_optimizer_updates"] != 0
                    or review["recomputed_foundation_encoder_decoder_pairs"] != 300 - start
                    or any(
                        type(review[k]) is not int or review[k] != (300 - start) * 10
                        for k in ("new_native_attempted_steps", "new_native_integrated_steps")
                    )
                ):
                    raise ValueError(
                        "only actual new remainder steps and scheduled planner calls required"
                    )
                prediction_steps += (300 - start) * 10
            cases.append((row, world, commitment, control))
        choice = sealed(root / f"case-{i}-prospective-choice.json")
        selected = max(
            [0, 2, 3],
            key=lambda n: (
                group["predictions"][n]["predicted_outcome"]["safe"],
                group["predictions"][n]["predicted_outcome"]["training_reward"],
                -n,
            ),
        )
        if (
            choice["index"] != i
            or choice["selected_branch"] != selected
            or choice["selected_action"] != group["predictions"][selected]["action_name"]
            or choice["selection_uses_only_measured_past_and_own_simulated_future"] is not True
            or choice["advisory_only"] is not True
            or choice["teacher_qualified"] is not False
            or choice["hardware_authorized"] is not False
        ):
            raise ValueError("advisory choice must follow only its own prospective ranking")
    totals = dict(
        new_main_native_steps=36000,
        new_prediction_native_steps=prediction_steps,
        combined_new_native_steps=36000 + prediction_steps,
    )
    if prediction_steps != 14320 or any(
        type(complete.get(k)) is not int or complete[k] != v for k, v in totals.items()
    ):
        raise ValueError("whole50320-step source accounting required")
    return complete, pins, cases, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--complete-file-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or (args.root / "failure.json").exists():
        raise ValueError("unused output and no source failure required")
    require_terminal(args.unit)
    complete, pins, cases, arrays = read_campaign(
        args.root, complete_file_hash=args.complete_file_hash
    )
    if args.preflight:
        print("NAVIGATION_RECOVERY_WHOLE12_RAW_READY_NO_NEW_PHYSICS", flush=True)
        return
    args.output.mkdir()
    declaration = dict(
        schema="soccer.rsi.navigation_recovery4_verification_declaration.v1",
        producer_complete_hash=complete["report_hash"],
        input_file_hashes=pins,
        physical_step_budget=36000,
        new_nn_inferences=0,
        new_optimizer_updates=0,
        teacher_qualified=False,
        hardware_authorized=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(args.output / "declaration.json", declaration)
    rows = []
    audit = index = None
    deadline = time.monotonic() + 1800
    try:
        import mujoco

        from rosclaw_soccer.physics.recorded_early_episode import RecordedEarlyEpisode
        from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController

        for index, (row, world, commitment, control) in enumerate(cases):
            audit = None
            if time.monotonic() >= deadline:
                raise RuntimeError("bounded recorded PD verification expired")
            ph = commitment["physics"]
            model = mujoco.MjModel.from_binary_path(world["path"])
            audit = RecordedEarlyEpisode(
                model,
                world_hash=world["sha256"],
                kp=np.asarray(ph["canonical_pd_kp"], dtype=np.float64),
                kd=np.asarray(ph["canonical_pd_kd"], dtype=np.float64),
                course=tuple(row["course"]),
                defaults=np.asarray(G1SonicRunupController.default_angles, dtype=np.float64),
            )
            review = audit.verify_unforced_control(arrays(row["arrays"]))
            if review["actual_physical_steps"] != 3000 or any(
                not np.isfinite(e) or e > 1e-8 for e in review["maximum_absolute_errors"].values()
            ):
                raise ValueError("whole finite recorded-target physical alignment required")
            result = dict(
                index=index,
                source_case_hash=row["report_hash"],
                physical_review=review,
                control_review=control,
            )
            result["report_hash"] = hash_json(result)
            write_once(args.output / f"case-{index}-complete.json", result)
            rows.append(result)
            print("NAVIGATION_RECOVERY_RECORDED_PHYSICS_COMPLETE", index, flush=True)
        if len(rows) != 12 or any(digest(n) != h for n, h in pins.items()):
            raise ValueError("whole12 input chain changed during physics audit")
        result = dict(
            schema="soccer.rsi.navigation_recovery4_verification_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            actual_physical_steps=36000,
            new_nn_inferences=0,
            new_optimizer_updates=0,
            independent_neural_policy_qualified=False,
            temporal_execution_order_independently_attested=False,
            teacher_qualified=False,
            training_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(args.output / "complete.json", result)
    except BaseException as error:
        write_once(
            args.output / "failure.json",
            dict(
                current_case=index,
                completed_cases=len(rows),
                current_attempted_steps=0 if audit is None else audit.attempted_steps,
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
