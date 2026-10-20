"""Qualify fixed24 actual student30 + teacher history, NOT recovery or learning."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "core-source", "near-review", "near-plan", "shadow", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "expected-head",
        "expected-inputs-hash",
        "expected-near-review-hash",
        "shadow-unit",
        "expected-shadow-source",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--producer", type=Path)
    p.add_argument("--producer-unit")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_motion_portfolio_common import admit
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    # The live predecessor may finish while old lineage is hashed. No model or
    # physical body is allocated until its actual terminal0 and receipt pass.
    spec, _, pins, _ = admit(a)
    wait_for_terminal_collection(a.shadow_unit, wait=True)
    shadow = json.loads((a.shadow / "complete.json").read_text())
    if (
        shadow["source_head"] != a.expected_shadow_source
        or shadow["status"] != "SHADOW_WARM20_TEACHER_NEURAL_NATIVE_REPRODUCED"
        or shadow["all18_array_fields_byte_exact_to_original"] is not True
        or shadow["new_native_steps"] != 62000
        or len(shadow["shadow_history_checks"]) != 2
        or any(
            c
            != dict(
                shadow_encoder_calls=20,
                shadow_decoder_calls=20,
                checked_own_proprioception_frames=20,
                actual_consumed_history930_byte_exact=True,
            )
            for c in shadow["shadow_history_checks"]
        )
        or digest(a.near_review / "complete.json") != a.expected_near_review_hash
    ):
        raise ValueError("actual shadow20 full native/history parity prerequisite required")
    review = json.loads((a.near_review / "complete.json").read_text())
    if review["status"] != "CATEGORICAL_NEAR96_FOUR_ARM_PRIMITIVES_AUTHENTICATED":
        raise ValueError("full consumed near96 raw audit required")
    for key, value in review["input_file_hashes"].items():
        if key in pins and pins[key] != value:
            raise ValueError("conflicting historical lineage")
        pins[key] = value
    for f in (a.near_review / "complete.json", a.shadow / "complete.json"):
        pins[str(f)] = digest(f)
    if pins.get(str(a.near_plan / "complete.json")) != digest(a.near_plan / "complete.json"):
        raise ValueError("original fixed near96/roster declaration required")
    plan = json.loads((a.near_plan / "complete.json").read_text())
    roster = plan["roster"]
    if [r["variant"] for r in roster] != ["anchor", "seed17", "seed29", "seed43"]:
        raise ValueError("fixed previously evaluated categorical roster required")
    from dataclasses import asdict

    from rosclaw_soccer.rsi.teacher_coverage_cohort import teacher_coverage_indices
    from rosclaw_soccer.training.categorical_motion_holdout import (
        categorical_motion_holdout_courses,
    )
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

    courses = categorical_motion_holdout_courses()
    if preflight_receiving_courses(courses) != plan["preflight_hash"]:
        raise ValueError("complete original consumed near96 population required")
    indices = teacher_coverage_indices()
    traces = {}
    for i in indices:
        matched = [Path(k) for k in pins if k.endswith(f"/course{i:03d}/seed43/native.npz")]
        if len(matched) != 1:
            raise ValueError("one byte-bound recorded seed43 endpoint per case required")
        traces[i] = matched[0]
    prior_paths = [Path(k) for k in pins if Path(k).name == "entry-affordance.json"]
    if len(prior_paths) != 1:
        raise ValueError("one original affordance required")
    if bool(a.producer) != bool(a.producer_unit):
        raise ValueError("paired actual producer root/unit required")
    if a.producer:
        wait_for_terminal_collection(a.producer_unit, wait=True)
        producer = json.loads((a.producer / "complete.json").read_text())
        if (
            producer["status"] != "STUDENT30_SHADOW_HISTORY_NATIVE_QUALIFIED"
            or producer["source_head"] != a.expected_head
            or producer["indices"] != list(indices)
        ):
            raise ValueError("all24 same-source warm qualification producer required")
        pins.update({str(f): digest(f) for f in a.producer.rglob("*") if f.is_file()})
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            indices=list(indices),
            cases=[asdict(courses[i]) for i in indices],
            student_seed=43,
            actual_intervention_frame=30,
            maximum_native_steps=24 * 300,
            student_neural_recomputed=True,
            foundation_actions0_to19_replayed=True,
            teacher_advice_executed=False,
            recovery_executed=False,
            separate_process_same_implementation_repeat=bool(a.producer),
            optimizer_steps=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    import mujoco
    import numpy as np
    import torch

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.sonic_navigation import (
        G1SonicNavigation,
        SonicNavigationConfig,
    )
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupConfig, G1SonicRunupController
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.categorical_motion import create_categorical_motion
    from rosclaw_soccer.rsi.foundation_motor_codec import FoundationMotorCodec
    from rosclaw_soccer.rsi.optimizer_lineage_audit import exact_tensor_tree
    from scripts.rsi_motor_student_rollout import exact_array
    from scripts.rsi_online_kinematic_motion import load_model
    from scripts.rsi_warm_student_intervention import warm_student30

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parent = load_model(spec, Path(roster[0]["path"])).eval().requires_grad_(False)
    before = {k: v.clone() for k, v in parent.state_dict().items()}
    model = create_categorical_motion(parent)
    cp = torch.load(roster[3]["path"], map_location="cpu", weights_only=True)
    if (
        cp["model_kind"] != "categorical_fsq32_motion64"
        or cp["anchor_hash"] != roster[0]["hash"]
        or cp["world_hash"] != digest(Path(spec["world"]))
    ):
        raise ValueError("same categorical43/parent/world identities required")
    model.load_state_dict(cp["state_dict"], strict=True)
    for k, v in parent.state_dict().items():
        exact_tensor_tree(v, before[k], path=k)
    model.eval().requires_grad_(False)
    world = mujoco.MjModel.from_binary_path(spec["world"])
    root = Path(spec["model_root"])
    muscle_backend = G1SonicRunupController(
        root, G1SonicRunupConfig(model_variant="low_latency"), inference_threads=1
    )
    stored = json.loads(prior_paths[0].read_text())
    prior = SkillEntryAffordance(
        tuple(stored["ball_offset_xy_m"]),
        stored["historical_frame"],
        stored["historical_forward_speed_mps"],
        stored["historical_artifact_hash"],
    )
    if prior.contract_hash != stored["contract_hash"]:
        raise ValueError("original teacher affordance contract changed")
    rows = []
    for i in indices:
        folder = a.output / f"course{i:03d}"
        folder.mkdir()
        body = G1TargetEpisode(
            world,
            world_hash=digest(Path(spec["world"])),
            kp=muscle_backend.kp.copy(),
            kd=muscle_backend.kd.copy(),
            course=courses[i].course,
            defaults=G1SonicRunupController.default_angles.copy(),
            record_all_ball_contacts=True,
        )
        nav = G1SonicNavigation(
            root,
            courses[i].agent_id,
            SonicNavigationConfig(
                maximum_frames=300,
                model_variant="low_latency",
                inference_threads=1,
                record_default_planner_events=True,
            ),
        )
        progress = {}
        try:
            with np.load(traces[i], allow_pickle=False) as z:
                expected = {k: z[k].copy() for k in z.files}
            model.calls = 0
            actual, labels = warm_student30(
                model,
                FoundationMotorCodec(muscle_backend),
                nav,
                body,
                prior,
                expected,
                progress=progress,
            )
            for name, fields in (("actual-student30.npz", actual), ("shadow-labels30.npz", labels)):
                np.savez_compressed(folder / name, **fields)
                if a.producer:
                    with np.load(a.producer / folder.name / name, allow_pickle=False) as z:
                        if set(z.files) != set(fields):
                            raise ValueError("all repeated field identities required")
                        for k, v in fields.items():
                            exact_array(v, z[k])
            row = dict(
                course_index=i,
                actual_student_native_steps=body.integrated_steps,
                actual_attempted_steps=body.attempted_steps,
                actual_student_prefix30_byte_exact=True,
                shadow_actual_history930_byte_exact=True,
                new_student_policy_calls=30,
                new_student_muscle_decoder_calls=10,
                shadow_encoder_calls=30,
                shadow_decoder_calls=30,
                recovery_executed=False,
                optimizer_steps=0,
            )
            write_once(folder / "complete.json", row)
            rows.append(row)
            write_once(a.output / f"progress{i:03d}.json", row)
            print("ACTUAL_STUDENT30_SHADOW_HISTORY", i, body.integrated_steps, flush=True)
        except Exception as error:
            for name, data in progress.items():
                if data:
                    np.savez_compressed(
                        folder / ("partial-" + name + ".npz"),
                        **{k: np.stack([r[k] for r in data]) for k in data[0]},
                    )
            failure = dict(
                course_index=i,
                error=repr(error),
                native_steps=body.integrated_steps,
                attempted_steps=body.attempted_steps,
                automatic_retry_authorized=False,
            )
            write_once(folder / "failure.json", failure)
            write_once(a.output / "failure.json", failure)
            raise
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="STUDENT30_SHADOW_HISTORY_NATIVE_QUALIFIED",
            source_head=a.expected_head,
            indices=list(indices),
            results=rows,
            new_native_steps=sum(r["actual_student_native_steps"] for r in rows),
            separate_process_same_implementation_repeat=bool(a.producer),
            recovery_executed=False,
            optimizer_steps=0,
            goal_complete=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
