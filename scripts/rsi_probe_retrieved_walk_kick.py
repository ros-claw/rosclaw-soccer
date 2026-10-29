"""Evaluate a frozen public-teacher retrieval baseline on held-out public cases."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-audit", "experimental-approach-config", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-audit-hash", "experimental-approach-config-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--shard", type=int, choices=range(4), required=True)
    p.add_argument("--prospective", action="store_true")
    p.add_argument("--extended", action="store_true")
    p.add_argument("--outcome-model-root", type=Path)
    p.add_argument("--outcome-model-hash")
    a = p.parse_args()
    if a.extended and not a.prospective:
        raise ValueError("extended bank requires prospective evaluation")
    if bool(a.outcome_model_root) != bool(a.outcome_model_hash):
        raise ValueError("exact outcome model root and receipt hash required together")

    from rosclaw_soccer.rsi.experimental_velocity_evidence import admit_heading_evidence
    from rosclaw_soccer.rsi.walk_kick_retrieval import (
        nearest_teacher,
        outcome_model_teacher,
        prospective_courses,
    )
    from rosclaw_soccer.rsi.walk_kick_teacher_plan import search_course_bank
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    courses = search_course_bank(expanded_public=True)
    bank_hash = preflight_receiving_courses(courses)
    prospective = prospective_courses(extended=a.extended) if a.prospective else ()
    prospective_hash = preflight_receiving_courses(prospective) if prospective else None
    if prospective and set(c.course for c in prospective) & set(c.course for c in courses):
        raise ValueError("new courses must not overlap public donor bank")
    if a.extended and set(c.course for c in prospective) & {
        c.course for c in prospective_courses()
    }:
        raise ValueError("extended bank must not overlap initial prospective study")
    if (
        a.output.exists()
        or a.source_audit.is_symlink()
        or digest(a.source_audit) != a.source_audit_hash
    ):
        raise ValueError("unused output and exact public audit required")
    source = json.loads(a.source_audit.read_text())
    if (
        source.get("status")
        != "PUBLIC40_CEM_ALL760_RAW_LABELS_AND_REEXECUTION_AUTHENTICATED_NOT_PROMOTED"
        or source.get("course_bank_hash") != bank_hash
        or [r["course_index"] for r in source["results"]] != list(range(40))
        or source.get("promotion_authorized") is not False
    ):
        raise ValueError("complete unpromoted public teacher audit required")
    pins = dict(source["input_file_hashes"])
    pins[str(a.source_audit)] = a.source_audit_hash
    model_report = None
    if a.outcome_model_root:
        model_receipt = a.outcome_model_root / "complete.json"
        if model_receipt.is_symlink() or digest(model_receipt) != a.outcome_model_hash:
            raise ValueError("exact outcome-model receipt required")
        model_report = json.loads(model_receipt.read_text())
        if (
            model_report.get("status") != "GROUPED_OUTCOME_MODEL_NEEDS_PHYSICS_NOT_SKILL"
            or model_report.get("source_audit_hash") != a.source_audit_hash
            or [x["seed"] for x in model_report["records"]] != [981, 982, 983]
            or model_report.get("promotion_authorized") is not False
        ):
            raise ValueError("same source and all three unpromoted models required")
        pins[str(model_receipt)] = a.outcome_model_hash
        for rec in model_report["records"]:
            checkpoint = a.outcome_model_root / f"model-{rec['seed']}.pt"
            if checkpoint.is_symlink() or digest(checkpoint) != rec["checkpoint_hash"]:
                raise ValueError("exact frozen outcome checkpoint required")
            pins[str(checkpoint)] = rec["checkpoint_hash"]
    fast_assets, point_pins = admit_heading_evidence(
        a.experimental_approach_config, a.experimental_approach_config_hash
    )
    for key, value in point_pins.items():
        if key in pins and pins[key] != value:
            raise ValueError("inconsistent asset provenance")
        pins[key] = value
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        matches = [Path(k) for k in pins if k.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError("unique authenticated asset required: " + suffix)
        return matches[0]

    # Fixed indices before training. Test outcomes are never used to choose donors.
    test_indices = tuple(range(0, 40, 5))
    selected = {}
    eligible = []
    for i, course in enumerate(courses):
        if i in test_indices:
            eligible.append(False)
            continue
        summary = source["results"][i]
        path = one(f"/course-{i:02d}/{summary['selected']}.json")
        row = json.loads(path.read_text())
        artifact = path.parent / row["artifact"]["path"]
        if (
            row["course"] != list(course.course)
            or artifact.parent != path.parent
            or artifact.is_symlink()
            or digest(artifact) != row["artifact"]["hash"]
        ):
            raise ValueError("selected teacher must match authenticated physical history")
        selected[i] = row
        eligible.append(bool(row["outcome"]["task_pass"] and row["healthy_full_horizon"]))

    import mujoco
    import numpy as np
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode
    from rosclaw_soccer.sim.contracts import hash_json

    features = np.asarray([c.course for c in courses])
    eligible = np.asarray(eligible, dtype=bool)
    if not eligible.any():
        raise ValueError("nonempty successful training donors required")
    learned_models = []
    if model_report:
        import torch

        from rosclaw_soccer.rsi.first_touch_outcome_model import FirstTouchOutcomeModel

        for rec in model_report["records"]:
            checkpoint = torch.load(
                a.outcome_model_root / f"model-{rec['seed']}.pt",
                map_location="cpu",
                weights_only=True,
            )
            if checkpoint.get("seed") != rec["seed"]:
                raise ValueError("outcome model seed mismatch")
            predictor = FirstTouchOutcomeModel().eval()
            predictor.load_state_dict(checkpoint["state_dict"], strict=True)
            learned_models.append(predictor)
    vm, vc = one("/velocity/v0/exported/policy.onnx"), one("/velocity/v0/params/deploy.yaml")
    assets = dict(
        velocity_model=vm,
        velocity_config=vc,
        velocity_model_hash=pins[str(vm)],
        velocity_config_hash=pins[str(vc)],
    )
    assets.update(fast_assets)
    types = _load_robonaldo(one("/policy/robonaldo/FreeKick.py").parents[2])
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(
            one("/cached-standard/declared-scene.xml").read_text(), prefixes=("a/",)
        )
    )
    evaluation_courses = prospective if a.prospective else courses
    evaluation_indices = tuple(range(len(prospective))) if a.prospective else test_indices
    assignments = [i for i in evaluation_indices if evaluation_indices.index(i) % 4 == a.shard]
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            course_bank_hash=bank_hash,
            prospective_course_bank_hash=prospective_hash,
            train_indices=[i for i in range(40) if i not in test_indices],
            test_indices=test_indices,
            assigned_indices=assignments,
            donor_indices=np.flatnonzero(eligible).tolist(),
            frozen_metric_scales=[0.5, 0.3, 0.8],
            training_labels_privileged=True,
            public_test_previously_consumed=not a.prospective,
            prospective_new_course_bank=a.prospective,
            extended_course_bank=a.extended,
            independent_fresh=False,
            no_actor_gradient=True,
            outcome_model_hash=a.outcome_model_hash,
            outcome_model_is_safety_gate=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    rows = []
    for i in assignments:
        donor = nearest_teacher(evaluation_courses[i].course, features, eligible)
        donors = {"retrieval": donor}
        if learned_models:
            donor_ids = np.flatnonzero(eligible)
            candidates = []
            for index in donor_ids:
                params = selected[int(index)]["parameters"]
                candidates.append(
                    [
                        *evaluation_courses[i].course,
                        params["speed_mps"],
                        params["entry_gap_m"],
                        params["reference_frame"],
                        params["root_lateral_offset_m"],
                    ]
                )
            candidate_tensor = torch.tensor(candidates, dtype=torch.float32)
            with torch.no_grad():
                scores = np.stack([m(candidate_tensor).sigmoid().numpy() for m in learned_models])
            donors["outcome-model"] = outcome_model_teacher(scores, donor_ids)
        arms = (
            ("parent", "retrieval", "outcome-model") if learned_models else ("parent", "retrieval")
        )
        for arm in arms:
            chosen = selected[donors[arm]] if arm != "parent" else None
            params = (
                dict(
                    speed_mps=1.0, entry_gap_m=0.9, reference_frame=None, root_lateral_offset_m=0.0
                )
                if arm == "parent"
                else chosen["parameters"]
            )
            own_assets = dict(
                assets,
                experimental_fast=False
                if arm == "parent"
                else chosen["experimental_fast_approach"],
            )
            result, arrays = evaluate_episode(
                model=model,
                assets=own_assets,
                policy_types=types,
                course=evaluation_courses[i].course,
                parameters=params,
                hard_limits=np.asarray(G1_HARD_TORQUE_LIMITS),
            )
            outcome = relabel_arrays(
                arrays,
                controls=result["controls"],
                ball_address=result["ball_qpos_address"],
                root_address=result["robot_qpos_address"],
            )
            if hash_json(outcome) != hash_json(result["outcome"]):
                raise ValueError("independent native outcome reconstruction mismatch")
            path = a.output / f"course-{i:02d}-{arm}.npz"
            np.savez_compressed(path, **arrays)
            rows.append(
                dict(
                    course_index=i,
                    donor_index=donors[arm] if arm != "parent" else None,
                    arm=arm,
                    result=result,
                    raw_hash=digest(path),
                )
            )
            write_once(a.output / f"course-{i:02d}-{arm}.json", rows[-1])
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="PUBLIC_RETRIEVED_TEACHER_EXPLORATORY_NOT_SKILL",
            source_head=a.expected_head,
            rows=rows,
            prospective_new_course_bank=a.prospective,
            extended_course_bank=a.extended,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
